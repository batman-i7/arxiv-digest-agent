"""
Selection/Ranking node.

arXiv's own "relevance" sort is a keyword/BM25-ish match over title+abstract
and is coarse for natural-language topic queries (e.g. "recent work on
KV-cache compression" vs. "KV cache compression techniques" can return
different orderings). We re-rank the candidate set by embedding cosine
similarity between the query and each abstract, using the same embedding
model as the paper chunks so everything lives in one comparable space.

This keeps ranking simple and dependency-light (no second LLM call just to
pick a paper) at the cost of missing some semantic nuance an LLM re-ranker
would catch. Documented as a tradeoff in the README.
"""

from __future__ import annotations

import numpy as np

from .embeddings import EmbeddingModel
from .state import PaperMetadata


def rank_candidates(
    query: str,
    candidates: list[PaperMetadata],
    embedder: EmbeddingModel,
) -> list[PaperMetadata]:
    if not candidates:
        return []

    query_vec = embedder.encode([query])[0]
    abstract_vecs = embedder.encode([c.abstract for c in candidates])

    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    for cand, vec in zip(candidates, abstract_vecs):
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
        cand.relevance_score = float(np.dot(query_norm, vec_norm))

    return sorted(candidates, key=lambda c: c.relevance_score, reverse=True)
