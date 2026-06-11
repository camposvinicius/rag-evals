# rag-evals

A CI-friendly evaluation harness for RAG pipelines: retrieval metrics, an
optional LLM-as-judge for answer faithfulness, and a regression gate that
fails your build when quality drops.

> Most RAG demos never answer one question: **how do you know it didn't get
> worse after your last change?** This project treats evals like software
> tests — run them on every commit, compare against a committed baseline,
> fail CI on regression.

## How it works

```
corpus.jsonl ──┐
               ├─► retriever (BM25) ──► retrieval metrics ──► run.json ──► regression gate
qa.jsonl ──────┘                                                              │
                                                                   committed baseline.json
        optional, credentialed:
        retrieved docs ──► grounded generation ──► LLM-as-judge (faithfulness)
```

- **`ragevals run`** retrieves for every annotated question and scores
  recall@k, precision@k, hit-rate@k, MRR and nDCG@k. With
  `--with-generation` it also generates a grounded answer per query and has
  an LLM judge score its faithfulness to the retrieved context. With
  `--retrieved` it scores results exported from **your own RAG stack**
  instead of running the built-in retriever.
- **`ragevals check`** compares a run against `baselines/baseline.json` and
  exits 1 if any retrieval metric drops more than the configured tolerance.
- **`ragevals diff`** compares two runs per query — which questions got
  worse, what was retrieved before and after.
- **`ragevals update-baseline`** promotes a good run to be the new baseline —
  the quality equivalent of updating a lockfile, reviewable in the PR diff.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

# offline eval (no API keys needed)
.venv/bin/ragevals run \
  --corpus data/sample/corpus.jsonl \
  --qa data/sample/qa.jsonl \
  --output runs/run.json

# regression gate (what CI runs)
.venv/bin/ragevals check --run runs/run.json
```

Output on the bundled sample dataset (12 documents, 15 annotated questions):

```
Evaluated 15 queries (k=5)
  recall@5       0.9333
  precision@5    0.1867
  hit_rate@5     0.9333
  mrr            0.8556
  ndcg@5         0.8754
```

## Evaluate your own RAG pipeline

The built-in BM25 retriever exists so the harness is demonstrable and CI
stays offline — but the harness is **stack-agnostic**. Export your
retriever's results as JSONL (one line per annotated question):

```json
{"qa_id": "q-001", "retrieved_doc_ids": ["doc-002", "doc-001"]}
```

and score them with:

```bash
ragevals run --qa qa.jsonl --retrieved retrieved.jsonl --output runs/run.json
ragevals check --run runs/run.json
```

LangChain, LlamaIndex, Elasticsearch, pgvector, OpenSearch, an internal
search API — anything works: rag-evals only needs the ranked ids. See
[examples/export_retrieved.py](examples/export_retrieved.py) for a template.
Every annotated question must have an entry (an empty list — "retriever
found nothing" — is valid; a missing entry is an error).

Try it right now against the sample QA set with a bundled example file
(an idealized retriever that beats the BM25 baseline):

```bash
ragevals run --qa data/sample/qa.jsonl \
  --retrieved examples/retrieved_jsonl/retrieved.jsonl --output runs/external.json
```

When a metric drops and you need to know *which* queries got worse:

```bash
ragevals diff --baseline runs/old.json --run runs/new.json --metric mrr
```

```
Worst regressions (mrr):
  q-015  1.0000 -> 0.0000  (-1.0000)
      Q: What does it cost to store data and how can I reduce storage spend over time?
      relevant:  doc-004, doc-005
      baseline retrieved: doc-004, doc-010, doc-001, doc-006, doc-002
      current retrieved:  doc-010, doc-001, doc-006, doc-002, doc-011
```

### With the LLM judge (AWS Bedrock)

```bash
export RAGEVALS_JUDGE_PROVIDER=bedrock
export RAGEVALS_JUDGE_MODEL=<bedrock-model-id>        # explicit on purpose
export RAGEVALS_GENERATION_MODEL=<bedrock-model-id>
# credentials/region resolve through the standard AWS chain (AWS_PROFILE, etc.)

.venv/bin/ragevals run --corpus data/sample/corpus.jsonl \
  --qa data/sample/qa.jsonl --output runs/run-judged.json --with-generation
```

This adds a `faithfulness` score per query: the fraction of the answer's
claims supported by the retrieved documents, judged by an LLM against a
strict JSON verdict contract.

## A real failure worth reading

On the sample set, BM25 misses the right documents for one query
(`q-015`, *"What does it cost to store data and how can I reduce storage
spend over time?"*) — classic vocabulary mismatch: the question says
"cost"/"spend", the relevant docs say "pricing"/"billing".

The grounded generator's answer for that query:

> "The provided documents do not answer this."

Faithfulness: **1.0**. Retrieval failed, but generation refused to invent —
and the judge confirmed the refusal was faithful. Retrieval quality and
generation discipline are different failure modes, and this harness
measures them separately.

## Design decisions

- **Offline by default.** BM25 retrieval and all retrieval metrics need no
  network and no keys, so the CI gate is free, fast and deterministic.
  Judge metrics live in a separate `generation_aggregates` block and do not
  gate CI.
- **Everything configurable via environment, nothing hardcoded.** There is
  deliberately no default judge or generation model: model choice affects
  cost and results, so it must be an explicit decision. See
  [.env.example](.env.example).
- **Absolute regression tolerance, not relative.** Metrics live in [0, 1];
  a relative rule gets stricter exactly when scores are low and looser when
  they are high — backwards for a quality gate.
- **Improvements never fail the gate, but are reported** so you remember to
  lock them in with `update-baseline`.
- **Strict dataset validation.** Duplicate ids, unannotated questions, or
  QA entries pointing at unknown documents fail at load time. A malformed
  dataset should explode, not silently skew the average.
- **Strict judge parsing.** A verdict that isn't valid JSON with a score in
  [0, 1] is an error, not a shrug. A judge you can't parse is a judge you
  can't trust. The judge runs at temperature 0.
- **Separate generator and judge models** (configurable) to avoid
  self-grading bias.

## File schemas

All files are JSONL (one JSON object per line) except the run/baseline JSON:

| File | Shape | Notes |
|---|---|---|
| `corpus.jsonl` | `{"id", "title", "text"}` | ids unique, text non-empty |
| `qa.jsonl` | `{"id", "question", "relevant_doc_ids": [..]}` | the human annotation; ids must exist in the corpus when one is provided |
| `retrieved.jsonl` | `{"qa_id", "retrieved_doc_ids": [..]}` | rank order, best first; one entry per QA id |
| `run.json` | `{"config", "aggregates", "per_query": [..]}` | written by `run`; `generation_aggregates` added by `--with-generation` |
| `baseline.json` | `{"config", "aggregates"}` | written by `update-baseline`; deliberately aggregates-only |

## Repository layout

```
src/ragevals/
  config.py             # env-driven settings (fail-fast validation)
  datasets.py           # JSONL corpus + QA + retrieved-results loaders
  retrieval.py          # Retriever protocol + BM25 implementation
  runner.py             # eval loop, aggregation, generation eval
  regression.py         # baseline comparison (the gate)
  diff.py               # per-query comparison between two runs
  generation.py         # grounded answer generation
  metrics/
    retrieval.py        # recall@k, precision@k, hit-rate@k, MRR, nDCG@k
    judge.py            # LLMClient protocol, BedrockClient, faithfulness
  cli.py                # run / check / diff / update-baseline
data/sample/            # fictional cloud-platform docs + annotated QA
examples/               # template for exporting your stack's results
baselines/baseline.json # committed quality baseline (the "lockfile")
.github/workflows/ci.yml
```

The sample corpus describes a **fictional** cloud provider ("Nimbus"), so
the dataset is self-contained and license-clean.

## Roadmap

- Dense retriever (embeddings) behind the same `Retriever` protocol — the
  vocabulary-mismatch failure above is the motivating case. (Until then,
  dense stacks can already be evaluated through `--retrieved`.)
- Judge calibration: agreement measurement against a hand-labeled subset.
- Additional judge providers (Anthropic API, OpenAI) behind `LLMClient`.
- Statistical gate (confidence intervals) for larger QA sets.

## License

MIT
