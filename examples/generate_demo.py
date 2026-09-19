"""
Generates examples/sample_run.md.

This sandbox has no route to arxiv.org or the Gemini API (its egress is
locked to package registries only), so this script uses fake LLM/embedder
implementations - identical in shape to the real ones - to produce a
representative, fully-working end-to-end transcript through the real graph
and QA code (src/graph.py, src/qa.py, src/cli.py:format_briefing). Only the
network-touching leaves (arxiv_client.search_by_topic, GeminiClient,
sentence-transformers download) are swapped out; every line of pipeline
logic - routing, chunking, vector search, fallback handling - is real.

Run with a real GEMINI_API_KEY and internet access and this same script
structure works unmodified against the live graph; swap DemoLLM ->
GeminiClient() and DemoEmbedder -> EmbeddingModel.get().
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import src.arxiv_client as arxiv_client
from src import graph as graph_mod
from src.cli import format_briefing
from src.llm import NOT_FOUND_MARKER, LLMClient
from src.qa import ask
from src.state import PaperMetadata
from src.vector_store import VectorStoreRegistry


class DemoEmbedder:
    dim = 8

    def encode(self, texts):
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        out = []
        for t in texts:
            seed = abs(hash(t)) % (2**32)
            rng = np.random.default_rng(seed)
            out.append(rng.random(self.dim).astype("float32"))
        return np.array(out, dtype="float32")


class DemoLLM(LLMClient):
    """Stands in for GeminiClient with fixed, representative outputs."""

    def summarize(self, paper_title, text, degraded):
        return {
            "summary": (
                "This paper proposes a KV-cache compression scheme for transformer-based LLM "
                "inference that reduces memory footprint by selectively evicting low-attention-score "
                "cache entries. It matters because KV-cache size is a major bottleneck for "
                "long-context serving, and the approach reports substantial memory savings with "
                "minimal quality loss."
            ),
            "problem_statement": (
                "As context lengths grow, the key-value cache in transformer decoders grows "
                "linearly with sequence length, becoming the dominant memory cost during inference "
                "and limiting batch size / context length on fixed hardware."
            ),
            "method": [
                "Track per-token attention scores across recent decoding steps",
                "Periodically evict cache entries below a dynamic importance threshold",
                "Use a small recovery buffer to reduce quality loss from mis-evicted tokens",
            ],
            "key_results": [
                "Up to 4.2x reduction in KV-cache memory on 7B and 13B parameter models",
                "Less than 0.5 perplexity degradation on long-context benchmarks",
                "Throughput improvement of ~1.8x at fixed batch size due to freed memory",
            ],
            "limitations": [
                "Evaluated primarily on English text; multilingual/code behavior untested",
                "Eviction threshold is tuned per-model and not shown to transfer automatically",
                "No evaluation against adversarial or retrieval-heavy long-context tasks",
            ],
            "suggested_questions": [
                "How does the eviction threshold get chosen for a new model?",
                "What happens to accuracy at extreme compression ratios (>8x)?",
                "Was this compared against quantization-based KV-cache methods?",
            ],
        }

    def answer(self, question, context_chunks, paper_title):
        q = question.lower()
        if "threshold" in q:
            return (
                "The eviction threshold is set dynamically per layer based on a moving average of "
                "attention scores over the last N decoding steps, calibrated on a small held-out "
                "validation set for each model (Section 3.2)."
            )
        if "quant" in q:
            return NOT_FOUND_MARKER
        return (
            "The method reduces KV-cache memory by evicting low-importance entries while keeping a "
            "small recovery buffer to limit quality loss (Section 3)."
        )


def main() -> None:
    paper = PaperMetadata(
        arxiv_id="2401.12345",
        title="Selective KV-Cache Eviction for Memory-Efficient LLM Inference",
        authors=["J. Researcher", "A. Coauthor"],
        abstract=(
            "We present a method for compressing the key-value cache in transformer decoders via "
            "selective eviction of low-attention-score entries, achieving significant memory "
            "savings with minimal quality degradation."
        ),
        # deliberately unreachable -> exercises the real parse-fallback path, not just the happy path
        pdf_url="https://example.invalid/paper.pdf",
        published=None,
    )

    orig_search = arxiv_client.search_by_topic
    arxiv_client.search_by_topic = lambda q, max_results=8: [paper]
    try:
        embedder = DemoEmbedder()
        registry = VectorStoreRegistry(dim=embedder.dim)
        llm = DemoLLM()
        state = graph_mod.run_graph(
            "recent work on KV-cache compression for LLMs", llm, embedder, registry
        )
    finally:
        arxiv_client.search_by_topic = orig_search

    out = []
    out.append("# Example Run\n")
    out.append(
        "This transcript was produced by running the real pipeline "
        "(`src/graph.py`, `src/qa.py`, `src/cli.py:format_briefing`) end-to-end, with the "
        "network-touching leaves swapped for deterministic fakes (see `generate_demo.py`) because "
        "this sandbox cannot reach arxiv.org or the Gemini API. The routing, chunking, vector "
        "search, and fallback-on-parse-failure logic below is real, not mocked.\n"
    )
    out.append("## Input\n```\nrecent work on KV-cache compression for LLMs\n```\n")
    out.append("## Pipeline log (note the PDF fetch fails and the graph degrades gracefully "
                "to abstract-only rather than crashing)\n```")
    out.extend(f"[log] {m}" for m in state.messages)
    out.append("```\n")
    out.append("## Executive Briefing Output\n")
    out.append(format_briefing(state.briefing))
    out.append("\n## Sample QA Exchanges\n")

    questions = [
        "How is the eviction threshold chosen?",
        "Was this compared against quantization-based KV-cache methods?",
        "What is the main method used?",
    ]
    for q in questions:
        turn = ask(state, q, llm, embedder, registry)
        out.append(f"**Q: {q}**\n\n{turn.answer}\n\n*(grounded={turn.grounded})*\n")

    text = "\n".join(out)
    out_path = Path(__file__).parent / "sample_run.md"
    out_path.write_text(text)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
