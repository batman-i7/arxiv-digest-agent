"""
QA loop, run after the graph produces a briefing.

Kept separate from graph.py because it's interactive/repeated rather than a
one-shot node - the state (vector store, paper metadata, conversation
history) is already sitting in AgentState from the summarize stage; this
module just reads from it on each question and appends a QATurn.

Grounding: we never let the model answer from parametric knowledge. The
prompt (see llm.py's _QA_SYSTEM_PROMPT) instructs the model to say
NOT_FOUND_IN_PAPER if the retrieved excerpts don't contain the answer, and
we surface that to the user verbatim rather than silently retrying or
letting the model guess.
"""

from __future__ import annotations

from .embeddings import EmbeddingModel
from .llm import NOT_FOUND_MARKER, LLMClient
from .state import AgentState, QATurn
from .vector_store import VectorStoreRegistry

TOP_K = 4


def ask(
    state: AgentState,
    question: str,
    llm_client: LLMClient,
    embedder: EmbeddingModel,
    vector_registry: VectorStoreRegistry,
) -> QATurn:
    assert state.selected_paper is not None and state.vector_collection is not None
    store = vector_registry.get(state.vector_collection)
    if store is None:
        turn = QATurn(question=question, answer="No vector store available for this paper.", grounded=False)
        state.qa_history.append(turn)
        return turn

    query_vec = embedder.encode([question])[0]
    hits = store.search(query_vec, top_k=TOP_K)
    context_chunks = [chunk for _, chunk, _ in hits]
    chunk_ids = [idx for idx, _, _ in hits]

    raw_answer = llm_client.answer(
        question=question,
        context_chunks=context_chunks,
        paper_title=state.selected_paper.title,
    )

    grounded = NOT_FOUND_MARKER not in raw_answer
    answer = (
        "I couldn't find this in the paper's text — it may not be covered, "
        "or it may be in a figure/table that wasn't extracted."
        if not grounded
        else raw_answer
    )

    turn = QATurn(question=question, answer=answer, grounded=grounded, source_chunk_ids=chunk_ids)
    state.qa_history.append(turn)
    return turn
