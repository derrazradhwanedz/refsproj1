"""Multi-source API search: arXiv, Crossref, and Semantic Scholar."""

from __future__ import annotations

import re
import time
from functools import lru_cache
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, NamedTuple, Tuple

import feedparser
import requests
import yaml

from src.config import ARXIV_API, default_headers, request_delay
from src.ref_logging import get_logger

_log = get_logger("search")


@dataclass
class PaperEntry:
    """Unified paper entry across arXiv, Crossref, and Semantic Scholar."""

    id: str
    title: str
    authors: str
    year: str
    url: str
    source: str
    raw: Any


class _SearchFilters(NamedTuple):
    field_prefix: str
    join_and: str
    join_or: str
    stopwords: frozenset[str]


@lru_cache(maxsize=1)
def _search_filters() -> _SearchFilters:
    path = Path(__file__).resolve().parent / "filters.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    arxiv = raw.get("arxiv") or {}
    field_prefix = str(arxiv.get("field_prefix") or "all:")
    join_and = str(arxiv.get("join_and") or " AND ")
    join_or = str(arxiv.get("join_or") or " OR ")
    sw_block = raw.get("stopwords")
    if isinstance(sw_block, list):
        stopwords = frozenset(str(w).lower() for w in sw_block if str(w).strip())
    elif isinstance(sw_block, str):
        stopwords = frozenset(w.lower() for w in sw_block.split() if w.strip())
    else:
        stopwords = frozenset()
    if not stopwords:
        _log.warning("filters.yaml: no stopwords; prose queries will not filter stopwords")
    return _SearchFilters(field_prefix, join_and, join_or, stopwords)


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _tokenize_for_query(snippet: str) -> list[str]:
    """
    Split prose into keyword candidates. Dots stay inside a token (arXiv-style ids,
    ``cs.LG``, ``v1.2``) so they are not broken into separate numbers; phrase breaks
    use whitespace, commas, hyphens, etc. (``[^\w.]``).
    """
    parts = re.split(r"[^\w.]+", snippet.lower())
    out: list[str] = []
    for p in parts:
        w = p.strip(".")
        if w:
            out.append(w)
    return out


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(default_headers())
    return s


def search_bulk(
    search_query: str,
    *,
    max_results: int = 50,
    sort_by: str = "submittedDate",
    sort_order: str = "descending",
) -> List[Any]:
    """
    Paginated search. `search_query` is an arXiv query string, e.g.
    `all:"small language model"` or `cat:cs.CL AND all:survey`.
    """
    sess = _session()
    delay = request_delay()
    entries: list[Any] = []
    start = 0

    while len(entries) < max_results:
        page = min(100, max_results - len(entries))
        params = {
            "search_query": search_query,
            "start": start,
            "max_results": page,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
        _log.debug("GET %s params=%s", ARXIV_API, params)
        r = sess.get(ARXIV_API, params=params, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"arXiv API HTTP {r.status_code}: {r.text[:500]}")
        feed = feedparser.parse(r.content)
        batch = list(feed.entries)
        _log.info("bulk page start=%s got %s entries (total so far %s)", start, len(batch), len(entries) + len(batch))
        entries.extend(batch)
        if len(batch) < page:
            break
        start += len(batch)
        time.sleep(delay)

    return entries[:max_results]


def _fuzzy_search_query(partial_title: str, max_words: int = 8) -> str:
    """Build a broad `all:` AND query from title words (arXiv-friendly)."""
    fp, ja, _, _ = _search_filters()
    words = [w for w in _tokenize_for_query(partial_title) if len(w) > 2]
    if not words:
        words = [partial_title.strip()]
    words = words[:max_words]
    q = ja.join(f"{fp}{w}" for w in words)
    _log.debug("fuzzy title query built: %s", q)
    return q


def text_to_arxiv_query(
    text: str,
    *,
    max_words: int = 6,
    max_chars: int = 1500,
    join: str = "or",
) -> str:
    """
    Build an arXiv ``all:`` query from prose. Default **OR** (any keyword) works better for
    long chunks than AND (all keywords). Title-style ``match`` still uses AND via
    ``_fuzzy_search_query``.
    """
    fp, ja, jo, stopwords = _search_filters()
    snippet = text.strip()[:max_chars]
    words = [w for w in _tokenize_for_query(snippet) if len(w) > 2 and w not in stopwords]
    seen: set[str] = set()
    uniq: list[str] = []
    for w in words:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    words = uniq[:max_words]
    if not words:
        fallback = _fuzzy_search_query(snippet, max_words=min(max_words, 4))
        _log.debug("text_to_arxiv_query fallback AND: %s", fallback)
        return fallback
    sep = jo if str(join).lower() == "or" else ja
    q = sep.join(f"{fp}{w}" for w in words)
    _log.debug("text_to_arxiv_query %s: %s", join, q)
    return q


def search_fuzzy_title(
    partial_title: str,
    *,
    max_results: int = 100,
) -> List[Tuple[float, Any]]:
    """Search candidates, then rank by string similarity to `partial_title`."""
    sess = _session()
    delay = request_delay()
    q = _fuzzy_search_query(partial_title)
    params = {
        "search_query": q,
        "max_results": max_results,
        "sortBy": "relevance",
    }
    _log.debug("GET %s params=%s", ARXIV_API, params)
    r = sess.get(ARXIV_API, params=params, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"arXiv API HTTP {r.status_code}: {r.text[:500]}")
    time.sleep(delay)
    feed = feedparser.parse(r.content)
    if not feed.entries:
        _log.info("fuzzy search returned 0 entries")
        return []

    scored: list[tuple[float, Any]] = []
    for entry in feed.entries:
        title = entry.title.strip().replace("\n", " ")
        score = title_similarity(partial_title, title)
        scored.append((score, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    if scored:
        _log.info("fuzzy ranked %s candidates; top score=%.3f", len(scored), scored[0][0])
    return scored


# =============================================================================
# Crossref Search Functions
# =============================================================================

from src.config import CROSSREF_API  # noqa: E402


def search_crossref(
    query: str,
    *,
    max_results: int = 20,
    sort_by: str = "relevance",
) -> List[Any]:
    """Search Crossref API for papers by query string."""
    sess = _session()
    delay = request_delay()

    rows = min(100, max_results)
    params: dict[str, Any] = {
        "query": query,
        "rows": rows,
        "sort": sort_by,
        "order": "desc",
    }

    _log.debug("GET Crossref %s params=%s", CROSSREF_API, params)
    r = sess.get(CROSSREF_API, params=params, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Crossref API HTTP {r.status_code}: {r.text[:500]}")

    data = r.json()
    items = data.get("message", {}).get("items", [])

    # Convert Crossref items to entry-like objects with source tag
    entries: list[Any] = []
    for item in items[:max_results]:
        # Create a simple namespace-like object
        entry = type("Entry", (), {})()
        entry.source = "crossref"
        entry.id = item.get("DOI", "")
        entry.doi = item.get("DOI", "")
        entry.title = item.get("title", [""])[0] if item.get("title") else ""
        entry.author = _crossref_authors(item.get("author", []))
        entry.published = item.get("published-print", {}).get("date-parts", [[""]])[0][0]
        if not entry.published:
            entry.published = item.get("published-online", {}).get("date-parts", [[""]])[0][0]
        if not entry.published:
            entry.published = item.get("created", {}).get("date-parts", [[""]])[0][0]
        entry.links = [{"href": f"https://doi.org/{entry.doi}"}]
        entries.append(entry)

    time.sleep(delay)
    _log.info("Crossref search returned %s entries", len(entries))
    return entries


def _crossref_authors(authors: list[dict]) -> str:
    """Format Crossref author list to string."""
    if not authors:
        return "N/A"
    names = []
    for a in authors[:5]:  # Limit to first 5 authors
        given = a.get("given", "")
        family = a.get("family", "")
        if given and family:
            names.append(f"{given} {family}")
        elif family:
            names.append(family)
    if len(authors) > 5:
        names.append("et al.")
    return ", ".join(names) if names else "N/A"


def search_crossref_fuzzy_title(
    partial_title: str,
    *,
    max_results: int = 20,
) -> List[Tuple[float, Any]]:
    """Search Crossref and rank by title similarity."""
    entries = search_crossref(partial_title, max_results=max_results)

    scored: list[tuple[float, Any]] = []
    for entry in entries:
        title = entry.title.strip().replace("\n", " ")
        score = title_similarity(partial_title, title)
        scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    if scored:
        _log.info("Crossref fuzzy ranked %s candidates; top score=%.3f", len(scored), scored[0][0])
    return scored


# =============================================================================
# Semantic Scholar Search Functions
# =============================================================================

from src.config import SEMANTIC_SCHOLAR_API  # noqa: E402


def search_semantic_scholar(
    query: str,
    *,
    max_results: int = 20,
    fields: str | None = None,
) -> List[Any]:
    """Search Semantic Scholar API for papers by query string."""
    sess = _session()
    delay = request_delay()

    # Default fields to fetch
    if fields is None:
        fields = "paperId,title,authors,year,externalIds,url,abstract"

    params: dict[str, Any] = {
        "query": query,
        "limit": min(100, max_results),
        "fields": fields,
    }

    url = f"{SEMANTIC_SCHOLAR_API}/paper/search"
    _log.debug("GET Semantic Scholar %s params=%s", url, params)
    r = sess.get(url, params=params, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Semantic Scholar API HTTP {r.status_code}: {r.text[:500]}")

    data = r.json()
    items = data.get("data", [])

    # Convert Semantic Scholar items to entry-like objects
    entries: list[Any] = []
    for item in items[:max_results]:
        entry = type("Entry", (), {})()
        entry.source = "semantic_scholar"
        entry.paper_id = item.get("paperId", "")
        entry.paperId = entry.paper_id  # Alias for compatibility
        entry.id = f"https://www.semanticscholar.org/paper/{entry.paper_id}"
        entry.title = item.get("title", "")
        entry.author = _semantic_scholar_authors(item.get("authors", []))
        entry.published = str(item.get("year", ""))
        entry.year = item.get("year")
        entry.abstract = item.get("abstract", "")
        entry.externalIds = item.get("externalIds", {})
        entry.links = [{"href": item.get("url", "")}]
        entries.append(entry)

    time.sleep(delay)
    _log.info("Semantic Scholar search returned %s entries", len(entries))
    return entries


def _semantic_scholar_authors(authors: list[dict]) -> str:
    """Format Semantic Scholar author list to string."""
    if not authors:
        return "N/A"
    names = []
    for a in authors[:5]:  # Limit to first 5 authors
        name = a.get("name", "")
        if name:
            names.append(name)
    if len(authors) > 5:
        names.append("et al.")
    return ", ".join(names) if names else "N/A"


def search_semantic_scholar_fuzzy_title(
    partial_title: str,
    *,
    max_results: int = 20,
) -> List[Tuple[float, Any]]:
    """Search Semantic Scholar and rank by title similarity."""
    entries = search_semantic_scholar(partial_title, max_results=max_results)

    scored: list[tuple[float, Any]] = []
    for entry in entries:
        title = entry.title.strip().replace("\n", " ")
        score = title_similarity(partial_title, title)
        scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    if scored:
        _log.info("Semantic Scholar fuzzy ranked %s candidates; top score=%.3f", len(scored), scored[0][0])
    return scored


# =============================================================================
# Unified API Interface
# =============================================================================

def _arxiv_to_entry(entry: Any) -> PaperEntry:
    """Convert arXiv feedparser entry to PaperEntry."""
    eid = getattr(entry, "id", "") or ""
    if "/abs/" in eid:
        aid = eid.split("/abs/")[-1]
    else:
        aid = eid.rsplit("/", 1)[-1]
    title = getattr(entry, "title", "").replace("\n", " ").strip()
    authors = getattr(entry, "author", "N/A")
    published = getattr(entry, "published", None) or ""
    year = published[:4] if len(published) >= 4 else "?"
    url = f"https://arxiv.org/abs/{aid}"
    return PaperEntry(
        id=aid,
        title=title,
        authors=str(authors),
        year=year,
        url=url,
        source="arxiv",
        raw=entry,
    )


def _crossref_to_entry(entry: Any) -> PaperEntry:
    """Convert Crossref entry object to PaperEntry."""
    doi = entry.doi
    title = entry.title.strip().replace("\n", " ")
    authors = entry.author
    year = str(entry.published) if entry.published else "?"
    url = f"https://doi.org/{doi}"
    return PaperEntry(
        id=doi,
        title=title,
        authors=authors,
        year=year,
        url=url,
        source="crossref",
        raw=entry,
    )


def _semantic_scholar_to_entry(entry: Any) -> PaperEntry:
    """Convert Semantic Scholar entry object to PaperEntry."""
    paper_id = entry.paper_id
    title = entry.title.strip().replace("\n", " ")
    authors = entry.author
    year = str(entry.published) if entry.published else "?"
    url = entry.links[0]["href"] if entry.links else f"https://www.semanticscholar.org/paper/{paper_id}"
    return PaperEntry(
        id=paper_id,
        title=title,
        authors=authors,
        year=year,
        url=url,
        source="semantic_scholar",
        raw=entry,
    )


def search_bulk_unified(
    query: str,
    api: str = "arxiv",
    *,
    max_results: int = 50,
    sort_by: str = "relevance",
    sort_order: str = "descending",
) -> List[PaperEntry]:
    """Unified bulk search across APIs."""
    if api == "arxiv":
        entries = search_bulk(query, max_results=max_results, sort_by=sort_by, sort_order=sort_order)
        return [_arxiv_to_entry(e) for e in entries]
    elif api == "crossref":
        entries = search_crossref(query, max_results=max_results, sort_by=sort_by)
        return [_crossref_to_entry(e) for e in entries]
    elif api == "semantic_scholar":
        entries = search_semantic_scholar(query, max_results=max_results)
        return [_semantic_scholar_to_entry(e) for e in entries]
    else:
        raise ValueError(f"Unknown API: {api}")


def search_fuzzy_title_unified(
    partial_title: str,
    api: str = "arxiv",
    *,
    max_results: int = 100,
) -> List[Tuple[float, PaperEntry]]:
    """Unified fuzzy title search across APIs."""
    if api == "arxiv":
        scored = search_fuzzy_title(partial_title, max_results=max_results)
        return [(score, _arxiv_to_entry(e)) for score, e in scored]
    elif api == "crossref":
        scored = search_crossref_fuzzy_title(partial_title, max_results=max_results)
        return [(score, _crossref_to_entry(e)) for score, e in scored]
    elif api == "semantic_scholar":
        scored = search_semantic_scholar_fuzzy_title(partial_title, max_results=max_results)
        return [(score, _semantic_scholar_to_entry(e)) for score, e in scored]
    else:
        raise ValueError(f"Unknown API: {api}")


def search_all_apis(
    query: str,
    *,
    max_results: int = 50,
) -> List[PaperEntry]:
    """Search all APIs and merge results, deduplicating by title similarity."""
    all_entries: List[PaperEntry] = []

    # Search arXiv
    try:
        arxiv_entries = search_bulk_unified(query, "arxiv", max_results=max_results)
        all_entries.extend(arxiv_entries)
        _log.info("arXiv returned %s entries", len(arxiv_entries))
    except Exception as e:
        _log.warning("arXiv search failed: %s", e)

    # Search Crossref
    try:
        crossref_entries = search_bulk_unified(query, "crossref", max_results=max_results)
        all_entries.extend(crossref_entries)
        _log.info("Crossref returned %s entries", len(crossref_entries))
    except Exception as e:
        _log.warning("Crossref search failed: %s", e)

    # Search Semantic Scholar
    try:
        s2_entries = search_bulk_unified(query, "semantic_scholar", max_results=max_results)
        all_entries.extend(s2_entries)
        _log.info("Semantic Scholar returned %s entries", len(s2_entries))
    except Exception as e:
        _log.warning("Semantic Scholar search failed: %s", e)

    # Deduplicate by title similarity
    deduped: List[PaperEntry] = []
    for entry in all_entries:
        is_duplicate = False
        for existing in deduped:
            if title_similarity(entry.title, existing.title) > 0.9:
                is_duplicate = True
                break
        if not is_duplicate:
            deduped.append(entry)

    _log.info("After deduplication: %s entries", len(deduped))
    return deduped


def text_to_query(
    text: str,
    api: str = "arxiv",
    *,
    max_words: int = 6,
) -> str:
    """Build API-appropriate query from text."""
    if api == "arxiv":
        return text_to_arxiv_query(text, max_words=max_words, join="or")
    elif api in ("crossref", "semantic_scholar", "all"):
        # For Crossref, S2, and 'all' mode, use simple keyword extraction
        words = [w for w in _tokenize_for_query(text) if len(w) > 2][:max_words]
        return " ".join(words)
    else:
        raise ValueError(f"Unknown API: {api}")
