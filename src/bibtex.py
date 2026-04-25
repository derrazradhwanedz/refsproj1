"""Fetch BibTeX from multiple sources: arXiv, Crossref, Semantic Scholar."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import requests

from src.config import (
    CROSSREF_API,
    SEMANTIC_SCHOLAR_API,
    default_headers,
    request_delay,
)
from src.ref_logging import get_logger

if TYPE_CHECKING:
    from src.search import PaperEntry

_log = get_logger("bibtex")


def _arxiv_id_from_entry(entry: Any) -> str:
    eid = getattr(entry, "id", "") or ""
    if "/abs/" in eid:
        return eid.split("/abs/")[-1]
    return eid.rsplit("/", 1)[-1]


def _doi_from_entry(entry: Any) -> str | None:
    """Extract DOI from entry if available."""
    # Try direct DOI attribute
    doi = getattr(entry, "doi", None)
    if doi:
        return str(doi)
    # Try links
    for link in getattr(entry, "links", []) or []:
        href = getattr(link, "href", "") or ""
        if "doi.org" in href:
            return href.split("doi.org/")[-1].split("?")[0]
    return None


def fetch_arxiv_bibtex(entry: Any, *, session: requests.Session | None = None) -> str:
    """Return BibTeX text for one feedparser entry."""
    sess = session or requests
    headers = default_headers()
    delay = request_delay()

    bibtex_link = None
    for link in getattr(entry, "links", []) or []:
        rel = getattr(link, "rel", "") or ""
        href = getattr(link, "href", "") or ""
        title = link.get("title") if hasattr(link, "get") else getattr(link, "title", None)
        if title == "bibtex" or (rel == "related" and "bibtex" in href):
            bibtex_link = href
            break

    if not bibtex_link:
        arxiv_id = _arxiv_id_from_entry(entry)
        bibtex_link = f"https://arxiv.org/bibtex/{arxiv_id}"

    _log.debug("GET arXiv bibtex %s", bibtex_link)
    r = sess.get(bibtex_link, headers=headers, timeout=60)
    r.raise_for_status()
    time.sleep(delay)
    text = r.text.strip()
    _log.info("fetched arXiv BibTeX (%s chars) for %s", len(text), bibtex_link)
    return text


def fetch_crossref_bibtex(entry: Any, *, session: requests.Session | None = None) -> str:
    """Fetch BibTeX for a Crossref DOI entry."""
    sess = session or requests
    headers = {**default_headers(), "Accept": "application/x-bibtex"}
    delay = request_delay()

    doi = _doi_from_entry(entry)
    if not doi:
        raise ValueError("No DOI found in entry for Crossref lookup")

    url = f"{CROSSREF_API}/{doi}/transform/application/x-bibtex"
    _log.debug("GET Crossref bibtex %s", url)
    r = sess.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    time.sleep(delay)
    text = r.text.strip()
    _log.info("fetched Crossref BibTeX (%s chars) for DOI %s", len(text), doi)
    return text


def fetch_semantic_scholar_bibtex(entry: Any, *, session: requests.Session | None = None) -> str:
    """Fetch BibTeX for a Semantic Scholar paper ID entry."""
    sess = session or requests
    headers = default_headers()
    delay = request_delay()

    # Extract paper ID from entry
    paper_id = getattr(entry, "paper_id", None) or getattr(entry, "paperId", None)
    if not paper_id:
        # Try to get from arXiv ID if available
        arxiv_id = _arxiv_id_from_entry(entry)
        if arxiv_id and arxiv_id != "unknown":
            paper_id = f"arxiv:{arxiv_id}"
        else:
            raise ValueError("No paper_id or arxiv_id found in entry for Semantic Scholar lookup")

    # Fetch paper details including external IDs
    url = f"{SEMANTIC_SCHOLAR_API}/paper/{paper_id}?fields=externalIds,citationStyles"
    _log.debug("GET Semantic Scholar paper %s", url)
    r = sess.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()

    # Try to get BibTeX from citation styles if available
    citation_styles = data.get("citationStyles", {})
    bibtex = citation_styles.get("bibtex")
    if bibtex:
        time.sleep(delay)
        _log.info("fetched Semantic Scholar BibTeX (%s chars) for %s", len(bibtex), paper_id)
        return bibtex

    # Fallback: construct BibTeX from available metadata
    external_ids = data.get("externalIds", {})
    doi = external_ids.get("DOI")
    if doi:
        # Use Crossref for DOI-based lookup
        time.sleep(delay)
        return fetch_crossref_bibtex(type("Entry", (), {"doi": doi, "links": []})(), session=sess)

    raise ValueError(f"No BibTeX available for Semantic Scholar paper {paper_id}")


def _detect_source(entry: Any) -> str:
    """Detect the source of an entry based on its attributes."""
    # Check for explicit source tag
    source = getattr(entry, "source", None)
    if source:
        return str(source).lower()

    # Check for arXiv ID pattern in entry ID
    eid = getattr(entry, "id", "") or ""
    if "arxiv.org" in eid or "/abs/" in eid:
        return "arxiv"

    # Check for DOI (likely Crossref or other DOI-based source)
    if _doi_from_entry(entry):
        return "crossref"

    # Check for Semantic Scholar specific fields
    if hasattr(entry, "paper_id") or hasattr(entry, "paperId"):
        return "semantic_scholar"

    # Default to arXiv for backward compatibility
    return "arxiv"


def fetch_bibtex(entry: Any, *, session: requests.Session | None = None, source: str | None = None) -> str:
    """Smart dispatcher: fetch BibTeX from appropriate source.

    Args:
        entry: The paper entry object (raw API response or PaperEntry)
        session: Optional requests session
        source: Explicit source override ('arxiv', 'crossref', 'semantic_scholar')
                If None, auto-detects from entry metadata

    Returns:
        BibTeX formatted string
    """
    # Handle PaperEntry dataclass - extract raw entry
    raw_entry = entry
    if hasattr(entry, "raw") and hasattr(entry, "source"):
        # It's a PaperEntry - use the raw underlying entry
        raw_entry = entry.raw
        # If source not specified, use PaperEntry's source
        if source is None:
            source = entry.source

    src = (source or _detect_source(raw_entry)).lower()

    if src == "arxiv":
        return fetch_arxiv_bibtex(raw_entry, session=session)
    elif src in ("crossref", "doi"):
        return fetch_crossref_bibtex(raw_entry, session=session)
    elif src in ("semantic_scholar", "semanticscholar", "s2"):
        return fetch_semantic_scholar_bibtex(raw_entry, session=session)
    else:
        _log.warning("Unknown source '%s', defaulting to arXiv", src)
        return fetch_arxiv_bibtex(raw_entry, session=session)
