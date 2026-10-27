"""Open-weights model served by Groq (free tier), JSON mode + validation against the enum."""

from groq import Groq

import config
from classifiers.llm_base import LLMClassifier


class GroqClassifier(LLMClassifier):
    provider = "groq"
    model = config.GROQ_MODEL

    def __init__(self, variant):
        super().__init__(variant)
        self.client = Groq(api_key=config.require_env("GROQ_API_KEY"))

    def _call(self, system, user):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content, resp.usage.prompt_tokens, resp.usage.completion_tokens
