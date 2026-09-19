"""
Chunk & Embed node (chunking half).

Strategy: fixed-size word-count chunks with overlap, splitting on paragraph
boundaries where possible. Simple and predictable rather than "smart"
(e.g. no semantic chunking / no layout-aware section chunking beyond what
pdf_parser's section splitter already gives us) — chosen for reliability
across papers with inconsistent formatting. See README tradeoffs section.
"""

from __future__ import annotations

CHUNK_SIZE_WORDS = 220
CHUNK_OVERLAP_WORDS = 40


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    words: list[str] = []
    for p in paragraphs:
        words.extend(p.split(" "))

    if not words:
        return []

    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(words):
        chunk_words = words[start : start + chunk_size]
        chunk = " ".join(chunk_words).strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks
