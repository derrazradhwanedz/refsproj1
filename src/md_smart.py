"""Chunk markdown, search arXiv per chunk, insert [n] citations and a reference list."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from src.bibtex import fetch_bibtex
from src.ref_logging import get_logger
from src.search import (
    PaperEntry,
    search_all_apis,
    search_bulk_unified,
    text_to_arxiv_query,
    text_to_query,
)

_log = get_logger("md_smart")

ChunkBy = Literal["paragraphs", "sentences"]
RefFormat = Literal["simple", "numbered", "links", "compact"]
BibMode = Literal["none", "combined", "split"]


def _entry_url(entry: PaperEntry) -> str:
    """Get URL from PaperEntry."""
    return entry.url


def _entry_id(entry: PaperEntry) -> str:
    """Get ID from PaperEntry."""
    return entry.id


def _entry_title(entry: PaperEntry) -> str:
    """Get title from PaperEntry."""
    return entry.title.replace("\n", " ").strip()


def _entry_year(entry: PaperEntry) -> str:
    """Get year from PaperEntry."""
    return entry.year if entry.year else "?"


def _entry_authors(entry: PaperEntry) -> str:
    """Get authors from PaperEntry."""
    return entry.authors if entry.authors else "Unknown"


def extract_front_matter(text: str) -> tuple[str, str]:
    """If front matter is delimited by --- lines, return (prefix, body) for chunking."""
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return "", text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            fm = "\n".join(lines[: i + 1])
            body = "\n".join(lines[i + 1 :]).lstrip("\n")
            return fm + "\n\n", body
    return "", text


def split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


# Window ends at the candidate sentence-ending dot; require full abbrev (not the dot in "e." of "e.g.").
_ABBREV_BEFORE_DOT = re.compile(
    r"(?:\be\.g\.|\bi\.e\.|\betc\.|\bvs\.|\bcf\.|\bet\s+al\.|\bfig\.|\bdr\.|\bmr\.|\bmrs\.|\bms\.|\bprof\.|"
    r"\boct\.|\bsep\.|\bdec\.|\bjan\.|\bmar\.|\bapr\.|\bjun\.|\bjul\.|\baug\.|\bnov\.|\bfeb\.)\s*$",
    re.I,
)


def split_sentences(text: str) -> list[str]:
    """
    Sentence boundaries: ``.?!`` then optional ``)`` / ``]`` / quotes, then
    whitespace, **or** markdown ``.** **Next`` (bold block after bold block).
    Skips common abbreviations (``e.g.``, ``i.e.``, months, etc.).
    """
    s = text.strip()
    if not s:
        return []
    n = len(s)
    breaks: list[int] = []
    i = 0
    while i < n:
        if s[i] in ".!?":
            # Dot between letters in "e.g" / "i.e" (not a sentence boundary).
            if s[i] == "." and i > 0 and i + 1 < n and s[i - 1 : i + 2].lower() in ("e.g", "i.e"):
                i += 1
                continue
            win = s[max(0, i - 24) : i + 1]
            if _ABBREV_BEFORE_DOT.search(win):
                i += 1
                continue
            j = i + 1
            while j < n and s[j] in ")]}\"'":
                j += 1
            if j + 1 < n and s[j : j + 2] == "**":
                j += 2
            while j < n and s[j].isspace():
                j += 1
            if j + 1 < n and s[j : j + 2] == "**":
                breaks.append(j)
                i = j
                continue
            while j < n and s[j].isspace():
                j += 1
            if j < n:
                breaks.append(j)
                i = j
                continue
        i += 1
    if not breaks:
        return [s] if s else []
    out: list[str] = []
    start = 0
    for b in breaks:
        piece = s[start:b].strip()
        if piece:
            out.append(piece)
        start = b
    tail = s[start:].strip()
    if tail:
        out.append(tail)
    return out


def group_every(items: list[str], n: int, chunk_by: ChunkBy) -> list[str]:
    if n < 1:
        n = 1
    sep = "\n\n" if chunk_by == "paragraphs" else " "
    out: list[str] = []
    for i in range(0, len(items), n):
        out.append(sep.join(items[i : i + n]))
    return out


def build_chunks(text: str, chunk_by: ChunkBy, chunk_n: int) -> list[str]:
    """Flat chunk list for **paragraph** mode only. Sentence mode uses line-wise processing."""
    if chunk_by == "paragraphs":
        units = split_paragraphs(text)
        return group_every(units, chunk_n, chunk_by)
    units = split_sentences(text)
    return group_every(units, chunk_n, chunk_by)


def format_reference_line(num: int, entry: PaperEntry, style: RefFormat) -> str:
    title = _entry_title(entry)
    url = _entry_url(entry)
    eid = _entry_id(entry)
    authors = _entry_authors(entry)
    year = _entry_year(entry)
    source_label = entry.source.upper() if entry.source else "PAPER"
    if style == "links":
        return f"{num}. [{title}]({url}) — `{eid}`"
    if style == "numbered":
        return f"{num}. {title} ({authors}, {year}) — {source_label}:{eid} — {url}"
    if style == "compact":
        return f"[{num}] {title} [{eid}]({url})"
    # simple
    return f"[{num}] {title} — {url}"


def process_markdown_smart(
    input_path: Path,
    output_md: Path,
    *,
    chunk_by: ChunkBy = "sentences",
    chunk_n: int = 1,
    refs_per_chunk: int = 1,
    query_max_words: int = 8,
    ref_format: RefFormat = "simple",
    bib_mode: BibMode = "none",
    bibtex_combined_path: Path | None = None,
    bibtex_split_dir: Path | None = None,
    api: str = "arxiv",
) -> tuple[Path, list[tuple[int, PaperEntry]]]:
    """
    Read ``input_path``, chunk, search per chunk, write ``output_md``.
    Returns (output_md, ordered (num, entry) for bibliography).

    Args:
        api: API to use ('arxiv', 'crossref', 'semantic_scholar', or 'all')
    """
    text = input_path.read_text(encoding="utf-8")
    prefix, body = extract_front_matter(text)

    id_to_num: dict[str, int] = {}
    ordered: list[tuple[int, PaperEntry]] = []
    next_n = 1
    chunk_counter = 0

    def cite_chunk(chunk: str) -> str:
        nonlocal next_n, chunk_counter
        chunk_counter += 1
        # Use appropriate query builder based on API
        if api == "arxiv":
            q = text_to_arxiv_query(chunk, max_words=query_max_words)
        else:
            q = text_to_query(chunk, api=api, max_words=query_max_words)
        _log.info("chunk %s query=%r", chunk_counter, q[:120])
        markers = ""
        try:
            # Use appropriate search function based on API
            if api == "all":
                hits = search_all_apis(q, max_results=refs_per_chunk)
            else:
                hits = search_bulk_unified(
                    q,
                    api=api,
                    max_results=max(refs_per_chunk, 1),
                    sort_by="relevance",
                )[:refs_per_chunk]
        except Exception as e:
            _log.warning("search failed chunk %s: %s", chunk_counter, e)
            hits = []

        cite_nums: list[int] = []
        for ent in hits:
            eid = _entry_id(ent)
            if eid not in id_to_num:
                id_to_num[eid] = next_n
                ordered.append((next_n, ent))
                next_n += 1
            cite_nums.append(id_to_num[eid])
        if cite_nums:
            markers = " " + "".join(f"[{n}]" for n in cite_nums)
        return chunk.rstrip() + markers

    if chunk_by == "sentences":
        # Line-wise: citations after each sentence, join with spaces inside the line so
        # markdown tables and single-newline structure stay intact (no blank line per sentence).
        line_parts: list[str] = []
        for raw_line in body.splitlines(keepends=True):
            if raw_line.endswith("\r\n"):
                nl, core = "\r\n", raw_line[:-2]
            elif raw_line.endswith("\n"):
                nl, core = "\n", raw_line[:-1]
            else:
                nl, core = "", raw_line
            if not core.strip():
                line_parts.append(raw_line)
                continue
            sents = split_sentences(core)
            grouped = group_every(sents, chunk_n, "sentences") if sents else [core]
            cited = [cite_chunk(g) for g in grouped]
            line_parts.append(" ".join(cited) + nl)
        body_out = "".join(line_parts)
    else:
        chunks = build_chunks(body, chunk_by, chunk_n)
        if not chunks:
            _log.warning("no chunks from %s", input_path)
            out_text = prefix + body + "\n\n## References\n\n*(no chunks)*\n"
            output_md.parent.mkdir(parents=True, exist_ok=True)
            output_md.write_text(out_text, encoding="utf-8")
            return output_md, []
        body_out = "\n\n".join(cite_chunk(c) for c in chunks)
    ref_lines = [format_reference_line(n, e, ref_format) for n, e in ordered]
    ref_section = "\n".join(ref_lines) if ref_lines else "*(no citations retrieved)*"
    full = f"{prefix}{body_out}\n\n## References\n\n{ref_section}\n"
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(full, encoding="utf-8")
    _log.info("wrote %s (%s citations)", output_md, len(ordered))

    if bib_mode == "combined" and ordered and bibtex_combined_path:
        bibtex_combined_path.parent.mkdir(parents=True, exist_ok=True)
        parts: list[str] = []
        for _, ent in ordered:
            try:
                parts.append(fetch_bibtex(ent))
            except Exception as e:
                _log.warning("bibtex skip %s: %s", _entry_id(ent), e)
        header = "% Thank you to arXiv for use of its open access interoperability.\n\n"
        bibtex_combined_path.write_text(header + "\n\n".join(parts), encoding="utf-8")
        _log.info("wrote combined bib %s", bibtex_combined_path)

    if bib_mode == "split" and ordered and bibtex_split_dir:
        bibtex_split_dir.mkdir(parents=True, exist_ok=True)
        for _, ent in ordered:
            eid = _entry_id(ent)
            try:
                bib = fetch_bibtex(ent)
            except Exception as e:
                _log.warning("bibtex split skip %s: %s", eid, e)
                continue
            p = bibtex_split_dir / f"{eid.replace('/', '_')}.bib"
            p.write_text(
                "% Thank you to arXiv for use of its open access interoperability.\n\n" + bib,
                encoding="utf-8",
            )
        _log.info("wrote split bib under %s", bibtex_split_dir)

    return output_md, ordered
