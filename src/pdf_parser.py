"""
Fetch & Parse node.

Downloads the PDF and extracts text with PyMuPDF. This is the most failure-
prone node in the graph (scanned/image-only PDFs, malformed layouts, huge
papers), so it's built to degrade rather than crash the whole run:

    - Scanned/image-only PDF (little to no extractable text) -> we fall back
      to the abstract already in hand from arXiv metadata, flag the paper as
      `parse_degraded`, and let downstream nodes (summarize, chunk) work off
      the abstract alone. The briefing will note reduced depth.
    - Huge paper -> we cap extraction at MAX_PAGES to keep latency and token
      usage bounded, and note the truncation.
    - Any other parse exception -> same abstract-only fallback, with the
      exception message recorded in `parse_notes` for debuggability.
"""

from __future__ import annotations

import io

import pymupdf as fitz  # PyMuPDF (new import name; `fitz` alias kept for readability below)
import requests

from .state import ParsedPaper

MAX_PAGES = 60
MIN_CHARS_PER_PAGE_TO_TRUST = 40  # below this, we assume scanned/garbled text
REQUEST_TIMEOUT_S = 30

_SECTION_HEADER_RE = None  # built lazily to avoid import cost if unused


def _looks_scanned(pages_text: list[str]) -> bool:
    if not pages_text:
        return True
    avg_chars = sum(len(p) for p in pages_text) / len(pages_text)
    return avg_chars < MIN_CHARS_PER_PAGE_TO_TRUST


def _split_sections(full_text: str) -> dict[str, str]:
    """
    Best-effort section splitting on common heading patterns. This is
    intentionally simple (regex over common headings) rather than a layout
    model — good enough to give the summarizer structure to work with,
    not good enough to trust for citation-grade extraction.
    """
    import re

    global _SECTION_HEADER_RE
    if _SECTION_HEADER_RE is None:
        headings = [
            "abstract", "introduction", "related work", "background",
            "method", "methods", "methodology", "approach",
            "experiments", "experimental setup", "results",
            "discussion", "limitations", "conclusion", "conclusions",
            "references", "acknowledgments", "acknowledgements",
        ]
        pattern = r"(?im)^\s*(?:\d+[\.\)]?\s*)?(" + "|".join(headings) + r")\s*$"
        _SECTION_HEADER_RE = re.compile(pattern)

    matches = list(_SECTION_HEADER_RE.finditer(full_text))
    if not matches:
        return {}

    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group(1).strip().lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body = full_text[start:end].strip()
        if body:
            # keep the longest occurrence if a heading repeats (e.g. TOC + body)
            if name not in sections or len(body) > len(sections[name]):
                sections[name] = body
    return sections


def fetch_and_parse(pdf_url: str, fallback_abstract: str) -> ParsedPaper:
    try:
        resp = requests.get(pdf_url, timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001 - deliberately broad, this is a fallback boundary
        return ParsedPaper(
            full_text=fallback_abstract,
            parse_degraded=True,
            parse_notes=f"PDF download failed ({e!r}); using abstract only.",
        )

    try:
        doc = fitz.open(stream=io.BytesIO(resp.content), filetype="pdf")
    except Exception as e:  # noqa: BLE001
        return ParsedPaper(
            full_text=fallback_abstract,
            parse_degraded=True,
            parse_notes=f"PDF could not be opened ({e!r}); using abstract only.",
        )

    total_pages = len(doc)
    truncated = total_pages > MAX_PAGES
    n_pages = min(total_pages, MAX_PAGES)
    pages_text = []
    try:
        for i in range(n_pages):
            pages_text.append(doc[i].get_text("text"))
    except Exception as e:  # noqa: BLE001
        return ParsedPaper(
            full_text=fallback_abstract,
            parse_degraded=True,
            parse_notes=f"Text extraction failed mid-document ({e!r}); using abstract only.",
        )
    finally:
        doc.close()

    if _looks_scanned(pages_text):
        return ParsedPaper(
            full_text=fallback_abstract,
            parse_degraded=True,
            parse_notes="Extracted text density too low (likely scanned/image PDF); using abstract only.",
        )

    full_text = "\n".join(pages_text)
    sections = _split_sections(full_text)

    notes = ""
    if truncated:
        notes = f"Paper has {total_pages} pages; truncated extraction to first {MAX_PAGES}."

    return ParsedPaper(
        full_text=full_text,
        sections=sections,
        parse_degraded=False,
        parse_notes=notes,
    )
