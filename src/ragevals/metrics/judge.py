"""LLM-as-judge: score generated answers against the retrieved context.

v1 ships one judge metric — **faithfulness**: is every claim in the answer
supported by the retrieved documents? This is the metric that catches
hallucination, which is the failure mode that actually hurts in production.

Provider support: AWS Bedrock (via boto3's ``converse`` API). The client
resolves credentials/region through the standard AWS chain (env vars,
``AWS_PROFILE``, instance roles) — nothing is hardcoded here. Anthropic and
OpenAI HTTP clients are welcome additions; the ``LLMClient`` protocol is
one method.
"""

from __future__ import annotations

import json
import re
from typing import Protocol


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str:
        """Send a single-turn prompt, return the model's text response."""
        ...


class BedrockClient:
    """Minimal Bedrock client over the ``converse`` API.

    Region and credentials come from the standard AWS resolution chain;
    the model id is injected explicitly (no default — model choice is a
    cost and quality decision the caller must own).
    """

    def __init__(self, model_id: str, max_tokens: int = 512):
        import boto3  # imported lazily so unit tests don't need boto3

        if not model_id:
            raise ValueError("model_id is required")
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._client = boto3.client("bedrock-runtime")

    def complete(self, prompt: str) -> str:
        response = self._client.converse(
            modelId=self._model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": self._max_tokens, "temperature": 0.0},
        )
        return response["output"]["message"]["content"][0]["text"]


_FAITHFULNESS_PROMPT = """\
You are evaluating whether an answer is faithful to its source documents.

Question:
{question}

Source documents:
{contexts}

Answer to evaluate:
{answer}

An answer is faithful when every factual claim it makes is directly supported
by the source documents. Ignore style; judge only factual support.

Respond with ONLY a JSON object, no other text:
{{"score": <float 0.0-1.0, fraction of claims supported>,
 "unsupported_claims": [<each unsupported claim, as a short string>]}}"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class JudgeError(RuntimeError):
    """Raised when the judge's response cannot be parsed."""


def parse_verdict(raw: str) -> dict:
    """Extract the JSON verdict from a model response.

    Models occasionally wrap JSON in markdown fences or prose despite
    instructions; we extract the outermost JSON object rather than failing
    on cosmetic noise — but invalid JSON or a missing/out-of-range score is
    a hard error. A judge you cannot parse is a judge you cannot trust.
    """
    match = _JSON_RE.search(raw)
    if not match:
        raise JudgeError(f"No JSON object found in judge response: {raw[:200]!r}")
    try:
        verdict = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise JudgeError(f"Judge returned invalid JSON: {raw[:200]!r}") from exc

    score = verdict.get("score")
    if not isinstance(score, int | float) or not 0.0 <= float(score) <= 1.0:
        raise JudgeError(f"Judge score missing or out of range: {verdict!r}")

    verdict["score"] = float(score)
    verdict.setdefault("unsupported_claims", [])
    return verdict


def judge_faithfulness(
    client: LLMClient, question: str, answer: str, contexts: list[str]
) -> dict:
    """Score one answer's faithfulness against its retrieved contexts."""
    prompt = _FAITHFULNESS_PROMPT.format(
        question=question,
        contexts="\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts)),
        answer=answer,
    )
    return parse_verdict(client.complete(prompt))
