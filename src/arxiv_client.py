"""
Thin wrapper around the official arXiv API (Atom feed), via the `arxiv`
package. No scraping — this hits export.arxiv.org/api/query under the hood,
which is the sanctioned entry point.

Two entry points, mirroring the two input modes in the assessment:
    - search_by_topic(query, max_results)
    - fetch_by_id(arxiv_id)
"""

from __future__ import annotations

import re
from typing import Optional

import arxiv

from .state import PaperMetadata

_ID_URL_RE = re.compile(r"arxiv\.org/abs/([\w.\-/]+)", re.IGNORECASE)
_BARE_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")


def looks_like_arxiv_id(text: str) -> Optional[str]:
    """
    Returns a normalized arXiv ID if `text` looks like an ID or an
    arxiv.org URL, else None. This is the core of "query understanding":
    distinguishing a topic string from a specific-paper lookup.
    """
    text = text.strip()
    url_match = _ID_URL_RE.search(text)
    if url_match:
        return url_match.group(1)
    if _BARE_ID_RE.match(text):
        return text
    return None


def _result_to_metadata(result: arxiv.Result) -> PaperMetadata:
    return PaperMetadata(
        arxiv_id=result.get_short_id(),
        title=result.title.strip(),
        authors=[a.name for a in result.authors],
        abstract=result.summary.strip().replace("\n", " "),
        pdf_url=result.pdf_url,
        published=result.published.date() if result.published else None,
        categories=list(result.categories),
    )


def search_by_topic(query: str, max_results: int = 8) -> list[PaperMetadata]:
    """
    Topic search. Returns up to `max_results` candidates ranked by arXiv's
    own relevance sort as a first pass; our own ranking node re-scores these
    against the query using embeddings (see ranking.py) since arXiv's
    relevance sort is keyword-based and often coarse.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    return [_result_to_metadata(r) for r in client.results(search)]


def fetch_by_id(arxiv_id: str) -> Optional[PaperMetadata]:
    """Direct lookup for a specific paper ID."""
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    results = list(client.results(search))
    if not results:
        return None
    return _result_to_metadata(results[0])
