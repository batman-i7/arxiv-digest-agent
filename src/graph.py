"""
The agentic state graph.

Implemented as a plain Python state machine rather than pulling in
LangGraph/CrewAI/etc: the pipeline here is mostly linear with a couple of
conditional branches, so a framework's overhead (learning its state-passing
conventions, its own retry/checkpoint semantics) wasn't worth it for a
2-3 day scope. Each node is a pure-ish function: (AgentState, **deps) ->
(AgentState, next_node_name | None). Deps (llm client, embedder, vector
registry) are passed in rather than imported as globals so the graph is
testable with fakes/mocks.

Edges (see NODES / run_graph below):

    query_understanding
        --topic-->      retrieve (topic search)
        --paper_id-->   retrieve (direct fetch)

    retrieve
        --zero results-->      END (status=NEEDS_INPUT)
        --topic, >=1 result--> rank_select
        --paper_id, found-->   fetch_parse
        --paper_id, not found--> END (status=ERROR)

    rank_select
        --always-->     fetch_parse   (after picking/confirming selected_paper)

    fetch_parse
        --always-->     chunk_embed   (parse failure degrades state but does not stop the graph)

    chunk_embed
        --always-->     summarize

    summarize
        --always-->     END (status=OK, or DEGRADED if parsing fell back to abstract-only)

After the graph ends with status OK/DEGRADED, the CLI drives the QA loop
separately (qa.py) since that's interactive/repeated, not a one-shot node.
"""

from __future__ import annotations

from typing import Callable, Optional

from . import arxiv_client, pdf_parser
from .chunker import chunk_text
from .embeddings import EmbeddingModel
from .llm import LLMClient
from .ranking import rank_candidates
from .state import AgentState, NodeStatus, PaperMetadata, QueryMode
from .vector_store import VectorStoreRegistry

NodeFn = Callable[..., tuple[AgentState, Optional[str]]]
SelectCallback = Callable[[list[PaperMetadata]], PaperMetadata]


def _default_select(candidates: list[PaperMetadata]) -> PaperMetadata:
    """Auto-pick the top-ranked candidate. See README for the interactive alternative."""
    return candidates[0]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def node_query_understanding(state: AgentState, **_) -> tuple[AgentState, Optional[str]]:
    arxiv_id = arxiv_client.looks_like_arxiv_id(state.raw_query)
    if arxiv_id:
        state.mode = QueryMode.PAPER_ID
        state.log(f"Query understood as a specific paper ID: {arxiv_id}")
    else:
        state.mode = QueryMode.TOPIC
        state.log(f"Query understood as a topic search: '{state.raw_query}'")
    return state, "retrieve"


def node_retrieve(state: AgentState, **_) -> tuple[AgentState, Optional[str]]:
    if state.mode == QueryMode.PAPER_ID:
        arxiv_id = arxiv_client.looks_like_arxiv_id(state.raw_query)
        paper = arxiv_client.fetch_by_id(arxiv_id)
        if paper is None:
            state.status = NodeStatus.ERROR
            state.log(f"No paper found for ID '{arxiv_id}'. Check the ID and try again.")
            return state, None
        state.candidates = [paper]
        state.selected_paper = paper
        state.log(f"Fetched paper directly: {paper.title}")
        return state, "fetch_parse"

    # topic search
    candidates = arxiv_client.search_by_topic(state.raw_query, max_results=8)
    if not candidates:
        state.status = NodeStatus.NEEDS_INPUT
        state.log(
            "arXiv returned zero candidates for this topic. Try a broader or "
            "differently-worded query (e.g. drop very specific qualifiers)."
        )
        return state, None

    state.candidates = candidates
    state.log(f"Retrieved {len(candidates)} candidate paper(s) for topic search.")
    return state, "rank_select"


def node_rank_select(
    state: AgentState,
    embedder: EmbeddingModel,
    select_callback: Optional[SelectCallback] = None,
    **_,
) -> tuple[AgentState, Optional[str]]:
    ranked = rank_candidates(state.raw_query, state.candidates, embedder)
    state.candidates = ranked

    top = ranked[0]
    others = ", ".join(f"{c.arxiv_id} ({c.relevance_score:.2f})" for c in ranked[1:4])
    state.log(
        f"Ranked {len(ranked)} candidates by embedding similarity. "
        f"Top pick: {top.arxiv_id} ({top.relevance_score:.2f})."
        + (f" Runner-up(s): {others}." if others else "")
    )

    selector = select_callback or _default_select
    state.selected_paper = selector(ranked)
    state.log(f"Selected paper: {state.selected_paper.title} ({state.selected_paper.arxiv_id})")
    return state, "fetch_parse"


def node_fetch_parse(state: AgentState, **_) -> tuple[AgentState, Optional[str]]:
    paper = state.selected_paper
    assert paper is not None
    parsed = pdf_parser.fetch_and_parse(paper.pdf_url, fallback_abstract=paper.abstract)
    state.parsed = parsed
    if parsed.parse_degraded:
        state.status = NodeStatus.DEGRADED
        state.log(f"Parsing degraded: {parsed.parse_notes}")
    else:
        note = f" ({parsed.parse_notes})" if parsed.parse_notes else ""
        state.log(f"Parsed full text successfully.{note}")
    return state, "chunk_embed"


def node_chunk_embed(
    state: AgentState,
    embedder: EmbeddingModel,
    vector_registry: VectorStoreRegistry,
    **_,
) -> tuple[AgentState, Optional[str]]:
    assert state.parsed is not None and state.selected_paper is not None
    chunks = chunk_text(state.parsed.full_text)
    if not chunks:
        # extremely short text (e.g. abstract-only fallback on a very short abstract)
        chunks = [state.parsed.full_text]
    state.chunks = chunks

    vectors = embedder.encode(chunks)
    store = vector_registry.get_or_create(state.selected_paper.arxiv_id)
    store.add(chunks, vectors)
    state.vector_collection = state.selected_paper.arxiv_id
    state.log(f"Chunked into {len(chunks)} piece(s) and embedded into the vector store.")
    return state, "summarize"


def node_summarize(state: AgentState, llm_client: LLMClient, **_) -> tuple[AgentState, Optional[str]]:
    assert state.parsed is not None and state.selected_paper is not None
    briefing = llm_client.summarize(
        paper_title=state.selected_paper.title,
        text=state.parsed.full_text,
        degraded=state.parsed.parse_degraded,
    )
    # attach metadata the model wasn't asked to invent
    briefing = {
        "title": state.selected_paper.title,
        "authors": state.selected_paper.authors,
        "arxiv_id": state.selected_paper.arxiv_id,
        "published": str(state.selected_paper.published) if state.selected_paper.published else None,
        "link": f"https://arxiv.org/abs/{state.selected_paper.arxiv_id}",
        **briefing,
    }
    state.briefing = briefing
    state.log("Generated executive briefing.")
    if state.status == NodeStatus.OK:
        pass  # stays OK
    return state, None


NODES: dict[str, NodeFn] = {
    "query_understanding": node_query_understanding,
    "retrieve": node_retrieve,
    "rank_select": node_rank_select,
    "fetch_parse": node_fetch_parse,
    "chunk_embed": node_chunk_embed,
    "summarize": node_summarize,
}


def run_graph(
    raw_query: str,
    llm_client: LLMClient,
    embedder: EmbeddingModel,
    vector_registry: VectorStoreRegistry,
    select_callback: Optional[SelectCallback] = None,
) -> AgentState:
    state = AgentState(raw_query=raw_query)
    current: Optional[str] = "query_understanding"
    visited_guard = 0
    while current is not None:
        visited_guard += 1
        if visited_guard > len(NODES) + 2:
            raise RuntimeError("Graph exceeded expected step count - possible routing bug.")
        node_fn = NODES[current]
        state, current = node_fn(
            state,
            llm_client=llm_client,
            embedder=embedder,
            vector_registry=vector_registry,
            select_callback=select_callback,
        )
    return state
