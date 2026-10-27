"""Gemini (Google AI Studio free tier) with structured output constrained to the label enum."""

from google import genai
from google.genai import types

import config
from classifiers.llm_base import LLMClassifier


class GeminiClassifier(LLMClassifier):
    provider = "gemini"
    model = config.GEMINI_MODEL

    def __init__(self, variant):
        super().__init__(variant)
        self.client = genai.Client(api_key=config.require_env("GEMINI_API_KEY"))
        self.gen_config = types.GenerateContentConfig(
            system_instruction=self.system,
            temperature=0,
            response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {"categoria": {"type": "STRING", "enum": config.CATEGORIES}},
                "required": ["categoria"],
            },
        )

    def _call(self, system, user):
        resp = self.client.models.generate_content(model=self.model, contents=user, config=self.gen_config)
        usage = resp.usage_metadata
        # Thinking tokens (if the model uses them) are billed as output.
        tokens_out = (usage.candidates_token_count or 0) + (getattr(usage, "thoughts_token_count", None) or 0)
        return resp.text, usage.prompt_token_count, tokens_out
