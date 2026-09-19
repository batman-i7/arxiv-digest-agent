import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunker import chunk_text


def test_empty_text():
    assert chunk_text("") == []


def test_short_text_single_chunk():
    text = "just a few words here"
    chunks = chunk_text(text, chunk_size=220, overlap=40)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_overlap_produces_shared_words():
    words = [f"word{i}" for i in range(500)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    # last 20 words of chunk[0] should reappear at the start of chunk[1]
    tail_of_first = chunks[0].split()[-20:]
    head_of_second = chunks[1].split()[:20]
    assert tail_of_first == head_of_second


def test_no_word_loss_across_chunks_with_zero_overlap():
    words = [f"w{i}" for i in range(50)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=10, overlap=0)
    rejoined = " ".join(chunks)
    assert rejoined == text


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
