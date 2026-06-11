# rag-evals

A CI-friendly evaluation harness for RAG pipelines: retrieval metrics, an
optional LLM-as-judge for generation quality, and a regression gate that
fails your CI when quality drops.

> Most RAG demos never answer one question: **how do you know it didn't get
> worse after your last change?** This project treats evals like software
> tests — run them on every commit, compare against a baseline, fail the
> build on regression.

## Status

Under active development. Current phase: retrieval metrics (done, tested).

## Design principles

- **Offline by default.** The default retriever (BM25) and the retrieval
  metrics need no API keys and no network — CI runs free and deterministic.
- **Everything configurable via environment.** No hardcoded models, no
  hardcoded thresholds. See [.env.example](.env.example).
- **Pure, hand-checkable metrics.** Every metric is a small pure function
  with unit tests you can verify on paper.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

## License

MIT
