"""Configuration, loaded entirely from environment variables.

Nothing that selects a model or a threshold is hardcoded: CI, local runs
and future experiments are configured the same way. See ``.env.example``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class ConfigError(ValueError):
    """Raised when the environment configuration is invalid."""


@dataclass(frozen=True)
class Settings:
    top_k: int
    retriever: str
    judge_provider: str | None
    judge_model: str | None
    generation_model: str | None
    regression_tolerance: float

    @classmethod
    def from_env(cls) -> Settings:
        top_k = int(os.environ.get("RAGEVALS_TOP_K", "5"))
        if top_k <= 0:
            raise ConfigError(f"RAGEVALS_TOP_K must be positive, got {top_k}")

        retriever = os.environ.get("RAGEVALS_RETRIEVER", "bm25").lower()
        if retriever not in ("bm25",):
            raise ConfigError(f"Unsupported RAGEVALS_RETRIEVER: {retriever!r}")

        judge_provider = os.environ.get("RAGEVALS_JUDGE_PROVIDER", "").lower() or None
        judge_model = os.environ.get("RAGEVALS_JUDGE_MODEL", "") or None
        generation_model = os.environ.get("RAGEVALS_GENERATION_MODEL", "") or None
        if judge_provider and judge_provider not in ("bedrock",):
            raise ConfigError(f"Unsupported RAGEVALS_JUDGE_PROVIDER: {judge_provider!r}")
        if judge_provider and not judge_model:
            # Deliberate: no default judge model. Model choice affects results
            # and cost, so it must be an explicit decision.
            raise ConfigError("RAGEVALS_JUDGE_MODEL is required when the judge is enabled")

        tolerance = float(os.environ.get("RAGEVALS_REGRESSION_TOLERANCE", "0.02"))
        if tolerance < 0:
            raise ConfigError(f"RAGEVALS_REGRESSION_TOLERANCE must be >= 0, got {tolerance}")

        return cls(
            top_k=top_k,
            retriever=retriever,
            judge_provider=judge_provider,
            judge_model=judge_model,
            generation_model=generation_model,
            regression_tolerance=tolerance,
        )
