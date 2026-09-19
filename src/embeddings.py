"""
Local embedding model wrapper (sentence-transformers). Runs on CPU, no API
key, no network call at inference time beyond the one-time model download.
Used both for chunk embedding (QA retrieval) and topic-vs-abstract ranking,
so query and document vectors always live in the same space.
"""

from __future__ import annotations

import os

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


class EmbeddingModel:
    _instance: "EmbeddingModel | None" = None

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    @classmethod
    def get(cls, model_name: str = DEFAULT_MODEL_NAME) -> "EmbeddingModel":
        # simple singleton so the CLI and QA loop don't reload the model
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name)
        return cls._instance

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        vecs = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vecs.astype("float32")

    @property
    def dim(self) -> int:
        # newer sentence-transformers renamed this method; support both
        if hasattr(self._model, "get_embedding_dimension"):
            return self._model.get_embedding_dimension()
        return self._model.get_sentence_embedding_dimension()
