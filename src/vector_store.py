"""
Local vector store using FAISS (in-process, no server). One "collection"
per paper (keyed by arxiv_id) so QA sessions on different papers never
cross-contaminate retrieval.

Persistence: kept in-memory for the CLI session by default. `save`/`load`
are provided so a longer-lived service (e.g. a future web backend) could
persist collections to disk between the summarize and QA stages instead of
holding everything in one process — see README "state persistence" note.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np


class PaperVectorStore:
    """One instance == one collection, i.e. one paper's chunks."""

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine sim via normalized inner product
        self.chunks: list[str] = []

    def add(self, chunks: list[str], vectors: np.ndarray) -> None:
        assert len(chunks) == vectors.shape[0]
        normed = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
        self.index.add(normed)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[tuple[int, str, float]]:
        if self.index.ntotal == 0:
            return []
        q = query_vector.reshape(1, -1)
        q = q / (np.linalg.norm(q) + 1e-8)
        scores, idxs = self.index.search(q, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((int(idx), self.chunks[idx], float(score)))
        return results

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, path: str | Path, dim: int) -> "PaperVectorStore":
        path = Path(path)
        store = cls(dim)
        store.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        return store


class VectorStoreRegistry:
    """Keeps one PaperVectorStore per arxiv_id alive for the process lifetime."""

    def __init__(self, dim: int):
        self.dim = dim
        self._stores: dict[str, PaperVectorStore] = {}

    def get_or_create(self, collection_key: str) -> PaperVectorStore:
        if collection_key not in self._stores:
            self._stores[collection_key] = PaperVectorStore(self.dim)
        return self._stores[collection_key]

    def get(self, collection_key: str) -> PaperVectorStore | None:
        return self._stores.get(collection_key)
