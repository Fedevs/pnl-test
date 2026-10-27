"""Common logic for API-based LLM classifiers."""

import time

import config
import prompts
from llm_utils import RateLimiter, api_retry, parse_json


class LLMClassifier:
    provider = None  # key into config.RPM / config.PRICES_USD_PER_M
    model = None

    def __init__(self, variant):
        examples = prompts.load_examples() if variant == "fs_rule" else None
        self.variant = variant
        self.system = prompts.system_prompt(variant, examples)
        self.limiter = RateLimiter(config.RPM[self.provider])
        self.few_shot_ids = list(examples["id"]) if examples is not None else []

    def _call(self, system, user):
        """Return (text, tokens_in, tokens_out). Implemented by each provider."""
        raise NotImplementedError

    @api_retry
    def _call_with_retry(self, user):
        self.limiter.wait()
        return self._call(self.system, user)

    def predict(self, text):
        out = {"label": None, "proba": None, "raw": None, "error": None, "tokens_in": None, "tokens_out": None}
        t0 = time.perf_counter()
        try:
            raw, out["tokens_in"], out["tokens_out"] = self._call_with_retry(prompts.user_prompt(text))
            out["raw"] = raw
            label = str(parse_json(raw).get("categoria", "")).strip().lower()
            if label in config.CATEGORIES:
                out["label"] = label
            else:
                out["error"] = f"invalid_label: {label!r}"
        except Exception as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"
        # Latency of the successful attempt plus any retries/waits: what a user would feel.
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out
