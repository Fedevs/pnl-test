"""Helpers shared by API-based scripts: rate limiting, retries and JSON parsing."""

import json
import re
import time

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


class RateLimiter:
    """Spaces calls so that at most `rpm` requests are sent per minute."""

    def __init__(self, rpm):
        self.interval = 60.0 / max(rpm, 1)
        self._last = 0.0

    def wait(self):
        elapsed = time.monotonic() - self._last
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last = time.monotonic()


def _is_retryable(exc):
    # Retry on rate limits (429), server errors (5xx) and network issues; not on 4xx
    # like invalid key or bad request, which won't fix themselves.
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    text = str(exc).lower()
    return any(k in text for k in ("429", "rate", "quota", "timeout", "temporarily", "unavailable", "connection"))


api_retry = retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=2, min=4, max=120),
    stop=stop_after_attempt(8),
    reraise=True,
)


def parse_json(text):
    """Parse a JSON object from a model response, tolerating code fences or extra text."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise
