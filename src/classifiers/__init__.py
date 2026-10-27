"""Classifiers with a common interface: `predict(text) -> dict`.

Every prediction dict has: label (str | None), latency_ms, tokens_in, tokens_out,
proba (dict | None), raw (str), error (str | None).
"""


def get_classifier(name, variant=None):
    if name == "tfidf":
        from classifiers.tfidf import TfidfClassifier

        return TfidfClassifier()
    if name == "gemini":
        from classifiers.gemini import GeminiClassifier

        return GeminiClassifier(variant)
    if name == "groq":
        from classifiers.groq_llm import GroqClassifier

        return GroqClassifier(variant)
    raise ValueError(f"Unknown classifier {name!r}")
