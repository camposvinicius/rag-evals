"""Tests for the LLM-as-judge plumbing — no network, no boto3.

A FakeClient stands in for Bedrock: the contract is just
``complete(prompt) -> str``, so the parsing, prompting, and aggregation
logic is fully testable offline.
"""

import json

import pytest

from ragevals.datasets import Document
from ragevals.generation import generate_answer
from ragevals.metrics.judge import JudgeError, judge_faithfulness, parse_verdict
from ragevals.runner import run_generation_eval


class FakeClient:
    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class TestParseVerdict:
    def test_clean_json(self):
        verdict = parse_verdict('{"score": 0.8, "unsupported_claims": ["x"]}')
        assert verdict["score"] == 0.8
        assert verdict["unsupported_claims"] == ["x"]

    def test_json_wrapped_in_markdown_fence(self):
        raw = 'Here is my evaluation:\n```json\n{"score": 1.0, "unsupported_claims": []}\n```'
        assert parse_verdict(raw)["score"] == 1.0

    def test_missing_score_is_error(self):
        with pytest.raises(JudgeError, match="score missing or out of range"):
            parse_verdict('{"unsupported_claims": []}')

    def test_score_out_of_range_is_error(self):
        with pytest.raises(JudgeError, match="out of range"):
            parse_verdict('{"score": 3.5}')

    def test_no_json_is_error(self):
        with pytest.raises(JudgeError, match="No JSON object"):
            parse_verdict("The answer looks fine to me!")

    def test_missing_unsupported_claims_defaults_to_empty(self):
        assert parse_verdict('{"score": 1.0}')["unsupported_claims"] == []


class TestJudgeFaithfulness:
    def test_prompt_contains_question_answer_and_contexts(self):
        client = FakeClient('{"score": 1.0, "unsupported_claims": []}')
        judge_faithfulness(client, "What is X?", "X is Y.", ["X is Y.", "Z is W."])
        prompt = client.prompts[0]
        assert "What is X?" in prompt
        assert "X is Y." in prompt
        assert "[2] Z is W." in prompt


class TestGenerateAnswer:
    def test_grounding_instruction_and_contexts_present(self):
        client = FakeClient("  X is Y.  ")
        answer = generate_answer(client, "What is X?", ["X is Y."])
        assert answer == "X is Y."
        assert "ONLY the source documents" in client.prompts[0]


class TestRunGenerationEval:
    def test_adds_generation_block_and_aggregates(self):
        docs = [Document(id="d1", title="T", text="X is Y.")]
        run_doc = {
            "config": {"k": 1, "num_queries": 1},
            "aggregates": {"recall@1": 1.0},
            "per_query": [
                {
                    "qa_id": "q1",
                    "question": "What is X?",
                    "retrieved_ids": ["d1"],
                    "relevant_ids": ["d1"],
                    "scores": {"recall@1": 1.0},
                }
            ],
        }
        generator = FakeClient("X is Y.")
        judge = FakeClient(json.dumps({"score": 0.9, "unsupported_claims": []}))

        result = run_generation_eval(run_doc, docs, generator, judge)

        assert result["generation_aggregates"] == {"faithfulness": 0.9}
        block = result["per_query"][0]["generation"]
        assert block["answer"] == "X is Y."
        assert block["faithfulness"] == 0.9
        # retrieval aggregates untouched -> regression gate unaffected
        assert result["aggregates"] == {"recall@1": 1.0}
