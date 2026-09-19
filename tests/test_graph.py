"""
Tests for the graph routing logic, using fakes for arXiv/LLM/embeddings so
they run offline and fast. These exercise the *branching logic* (zero
results, parse failure, ranking selection) which is the part most worth
unit-testing — the actual network calls are thin wrappers already covered
by manual runs (see README "Example run").
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src import graph as graph_mod
from src.llm import LLMClient
from src.state import NodeStatus, PaperMetadata, QueryMode
from src.vector_store import VectorStoreRegistry


class FakeEmbedder:
    dim = 4

    def encode(self, texts):
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        out = []
        for t in texts:
            seed = abs(hash(t)) % 1000
            rng = np.random.default_rng(seed)
            out.append(rng.random(self.dim).astype("float32"))
        return np.array(out, dtype="float32")


class FakeLLM(LLMClient):
    def summarize(self, paper_title, text, degraded):
        return {
            "summary": f"Fake summary of {paper_title}",
            "problem_statement": "fake problem",
            "method": ["fake method step"],
            "key_results": ["fake result"],
            "limitations": ["fake limitation"],
            "suggested_questions": ["fake question?"],
        }

    def answer(self, question, context_chunks, paper_title):
        return "fake answer"


def make_paper(arxiv_id="1234.5678"):
    return PaperMetadata(
        arxiv_id=arxiv_id,
        title=f"Fake Paper {arxiv_id}",
        authors=["A. Uthor"],
        abstract="This is a fake abstract about testing.",
        pdf_url="https://example.invalid/does-not-exist.pdf",  # forces parse fallback -> DEGRADED path
    )


def test_topic_search_zero_results(monkeypatch):
    monkeypatch.setattr(graph_mod.arxiv_client, "search_by_topic", lambda q, max_results=8: [])
    state = graph_mod.run_graph(
        "an extremely obscure nonexistent topic xyzzy",
        FakeLLM(),
        FakeEmbedder(),
        VectorStoreRegistry(dim=4),
    )
    assert state.status == NodeStatus.NEEDS_INPUT
    assert state.briefing is None


def test_paper_id_not_found(monkeypatch):
    monkeypatch.setattr(graph_mod.arxiv_client, "fetch_by_id", lambda aid: None)
    state = graph_mod.run_graph(
        "9999.99999",
        FakeLLM(),
        FakeEmbedder(),
        VectorStoreRegistry(dim=4),
    )
    assert state.mode == QueryMode.PAPER_ID
    assert state.status == NodeStatus.ERROR


def test_topic_search_happy_path_with_parse_fallback(monkeypatch):
    papers = [make_paper("1111.1111"), make_paper("2222.2222")]
    monkeypatch.setattr(graph_mod.arxiv_client, "search_by_topic", lambda q, max_results=8: papers)

    state = graph_mod.run_graph(
        "fake topic query",
        FakeLLM(),
        FakeEmbedder(),
        VectorStoreRegistry(dim=4),
    )

    # PDF url is unreachable -> pdf_parser should degrade to abstract-only, not crash
    assert state.status == NodeStatus.DEGRADED
    assert state.parsed is not None and state.parsed.parse_degraded is True
    assert state.briefing is not None
    assert state.briefing["arxiv_id"] == state.selected_paper.arxiv_id
    assert len(state.chunks) >= 1
    assert state.vector_collection == state.selected_paper.arxiv_id


def test_select_callback_is_used(monkeypatch):
    papers = [make_paper("1111.1111"), make_paper("2222.2222"), make_paper("3333.3333")]
    monkeypatch.setattr(graph_mod.arxiv_client, "search_by_topic", lambda q, max_results=8: papers)

    picked = {}

    def pick_last(candidates):
        picked["chosen"] = candidates[-1]
        return candidates[-1]

    state = graph_mod.run_graph(
        "fake topic query",
        FakeLLM(),
        FakeEmbedder(),
        VectorStoreRegistry(dim=4),
        select_callback=pick_last,
    )
    assert state.selected_paper.arxiv_id == picked["chosen"].arxiv_id


def test_qa_grounding_marker(monkeypatch):
    from src.llm import NOT_FOUND_MARKER
    from src.qa import ask
    from src.state import AgentState

    class RefusingLLM(FakeLLM):
        def answer(self, question, context_chunks, paper_title):
            return NOT_FOUND_MARKER

    embedder = FakeEmbedder()
    registry = VectorStoreRegistry(dim=4)
    store = registry.get_or_create("paper-x")
    store.add(["some chunk"], embedder.encode(["some chunk"]))

    state = AgentState(raw_query="irrelevant")
    state.selected_paper = make_paper("paper-x")
    state.vector_collection = "paper-x"

    turn = ask(state, "unanswerable question", RefusingLLM(), embedder, registry)
    assert turn.grounded is False
    assert "couldn't find" in turn.answer.lower()
    assert state.qa_history[-1] is turn


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
