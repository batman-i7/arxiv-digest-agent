"""
Shared state object that flows through every node in the graph.

Design note: we use a single mutable dataclass rather than passing a growing
tuple of arguments between functions. Every node reads what it needs off
`AgentState` and writes its results back onto it. This makes the state shape
explicit and inspectable at any point in the pipeline (useful for debugging,
logging, and for the QA loop, which needs to resume from a state that was
built minutes/hours earlier).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional


class QueryMode(str, Enum):
    TOPIC = "topic"
    PAPER_ID = "paper_id"
    UNKNOWN = "unknown"


class NodeStatus(str, Enum):
    OK = "ok"
    NEEDS_INPUT = "needs_input"   # e.g. ambiguous topic, multiple candidates
    ERROR = "error"               # unrecoverable for this run
    DEGRADED = "degraded"         # recovered, but with reduced quality (e.g. abstract-only fallback)


@dataclass
class PaperMetadata:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    pdf_url: str
    published: Optional[date] = None
    categories: list[str] = field(default_factory=list)
    # similarity score against the user's query, filled in by the ranking node
    relevance_score: float = 0.0


@dataclass
class ParsedPaper:
    full_text: str
    sections: dict[str, str] = field(default_factory=dict)  # best-effort section splitting
    parse_degraded: bool = False       # True if we fell back to abstract-only or partial text
    parse_notes: str = ""


@dataclass
class QATurn:
    question: str
    answer: str
    grounded: bool                     # False if the model reported "not found in the paper"
    source_chunk_ids: list[int] = field(default_factory=list)


@dataclass
class AgentState:
    # --- input ---
    raw_query: str

    # --- query understanding ---
    mode: QueryMode = QueryMode.UNKNOWN

    # --- retrieval ---
    candidates: list[PaperMetadata] = field(default_factory=list)

    # --- selection ---
    selected_paper: Optional[PaperMetadata] = None

    # --- fetch & parse ---
    parsed: Optional[ParsedPaper] = None

    # --- chunk & embed ---
    chunks: list[str] = field(default_factory=list)
    # index into a VectorStore keyed by paper arxiv_id; the store itself lives
    # outside the state (see vector_store.py) since it holds numpy arrays /
    # a FAISS index that we don't want to deep-copy around.
    vector_collection: Optional[str] = None

    # --- summarize ---
    briefing: Optional[dict] = None

    # --- QA loop ---
    qa_history: list[QATurn] = field(default_factory=list)

    # --- bookkeeping ---
    status: NodeStatus = NodeStatus.OK
    messages: list[str] = field(default_factory=list)  # human-readable trail of what happened

    def log(self, msg: str) -> None:
        self.messages.append(msg)
