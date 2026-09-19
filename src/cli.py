"""
CLI entry point.

Usage:
    python -m src.cli "recent work on KV-cache compression for LLMs"
    python -m src.cli 2401.12345
    python -m src.cli https://arxiv.org/abs/2401.12345
    python -m src.cli "..." --json out/briefing.json --no-interactive
"""

from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv

from .embeddings import EmbeddingModel
from .graph import run_graph
from .llm import get_llm_client
from .qa import ask
from .state import NodeStatus
from .vector_store import VectorStoreRegistry


class _FriendlyArgParser(argparse.ArgumentParser):
    """Show usage examples on error too, not just on --help."""

    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}\n", file=sys.stderr)
        print(self.epilog, file=sys.stderr)
        sys.exit(2)


def format_briefing(briefing: dict) -> str:
    lines = [
        f"# {briefing.get('title', 'Untitled')}",
        "",
        f"**Authors:** {', '.join(briefing.get('authors', []))}",
        f"**arXiv ID:** {briefing.get('arxiv_id')}  |  **Published:** {briefing.get('published')}",
        f"**Link:** {briefing.get('link')}",
        "",
        "## Summary",
        briefing.get("summary", ""),
        "",
        "## Problem Statement",
        briefing.get("problem_statement", ""),
        "",
        "## Method",
    ]
    lines += [f"- {m}" for m in briefing.get("method", [])]
    lines += ["", "## Key Results"]
    lines += [f"- {r}" for r in briefing.get("key_results", [])]
    lines += ["", "## Limitations"]
    lines += [f"- {l}" for l in briefing.get("limitations", [])]
    lines += ["", "## Suggested Follow-up Questions"]
    lines += [f"- {q}" for q in briefing.get("suggested_questions", [])]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    parser = _FriendlyArgParser(
        description="Autonomous arXiv Paper Digest & QA Agent",
        epilog=(
            "examples:\n"
            '  python -m src.cli "recent work on KV-cache compression for LLMs"\n'
            "  python -m src.cli 2401.12345\n"
            "  python -m src.cli https://arxiv.org/abs/2401.12345\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", help="A topic, an arXiv ID, or an arxiv.org/abs URL")
    parser.add_argument("--json", metavar="PATH", help="Also write the briefing JSON to this path")
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip the QA loop (useful for scripting/CI)",
    )
    args = parser.parse_args(argv)

    print("Loading embedding model (first run downloads weights, ~90MB)...", file=sys.stderr)
    embedder = EmbeddingModel.get()
    vector_registry = VectorStoreRegistry(dim=embedder.dim)

    try:
        llm_client = get_llm_client()
    except (RuntimeError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    state = run_graph(args.query, llm_client, embedder, vector_registry)

    for msg in state.messages:
        print(f"[log] {msg}", file=sys.stderr)

    if state.status == NodeStatus.NEEDS_INPUT:
        print("\nCouldn't proceed:", state.messages[-1] if state.messages else "needs more input.")
        return 2
    if state.status == NodeStatus.ERROR:
        print("\nError:", state.messages[-1] if state.messages else "unknown error.")
        return 1

    assert state.briefing is not None
    print("\n" + format_briefing(state.briefing) + "\n")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(state.briefing, f, indent=2)
        print(f"(briefing JSON written to {args.json})", file=sys.stderr)

    if args.no_interactive:
        return 0

    print("---\nAsk questions about this paper (blank line or Ctrl+D to quit).\n")
    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break
        turn = ask(state, question, llm_client, embedder, vector_registry)
        print(f"A: {turn.answer}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
