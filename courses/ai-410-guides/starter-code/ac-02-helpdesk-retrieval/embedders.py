"""
Two ways to turn text into a vector. You don't need to change this file.

LocalEmbedder  — counts words. Runs offline, needs no key, and gives
                 everyone the same numbers, so the tests use it.
GeminiEmbedder — Gemini's real embedding model (the one from HOS 2).
                 Optional: try it once your code passes, and compare.

Both return vectors you compare with `similarity()`.
"""

import math
import os
import re

STOPWORDS = set("""a an and are as at be but by can do does for from get got has have how i if in into is it
its me my no not of on or our so than that the their then there these they this to up was we were what when
where which who why will with you your""".split())


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in STOPWORDS]


class LocalEmbedder:
    """A bag-of-words "embedding": a word -> count dictionary."""

    name = "local"

    def embed_document(self, text: str) -> dict:
        return self._count(text)

    def embed_query(self, text: str) -> dict:
        return self._count(text)

    @staticmethod
    def _count(text: str) -> dict:
        counts: dict = {}
        for w in _tokens(text):
            counts[w] = counts.get(w, 0) + 1
        return counts


class GeminiEmbedder:
    """Gemini embeddings. Needs GEMINI_API_KEY in .env (free tier)."""

    name = "gemini"
    MODEL = "gemini-embedding-001"

    def __init__(self, document_task_type: str = "RETRIEVAL_DOCUMENT"):
        from dotenv import load_dotenv
        from google import genai
        from google.genai import types

        load_dotenv()
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self._types = types
        self.document_task_type = document_task_type

    def _embed(self, text: str, task_type: str) -> list[float]:
        result = self._client.models.embed_content(
            model=self.MODEL,
            contents=text,
            config=self._types.EmbedContentConfig(task_type=task_type, output_dimensionality=768),
        )
        return result.embeddings[0].values

    def embed_document(self, text: str) -> list[float]:
        return self._embed(text, self.document_task_type)

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text, "RETRIEVAL_QUERY")


def similarity(a, b) -> float:
    """Cosine similarity: 1.0 means pointing the same way, 0.0 means unrelated."""
    if isinstance(a, dict):
        dot = sum(v * b.get(k, 0) for k, v in a.items())
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
    else:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
