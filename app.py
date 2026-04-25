"""
arXiv reference tools: bulk BibTeX harvest or fuzzy single-paper match.

Run from repo root (e.g. D:\\SAMs\\refsproj1):
  python app.py bulk --query 'all:"small language model"' --max-results 20
  python app.py bulk -q 'cat:cs.CL' -n 5 --split
  python app.py bulk -q 'cat:cs.CL' -n 5 --split --session
  python app.py match --title "Survey of Small Language Models" -o paper.bib   # -> refs/paper.bib
  python app.py smart notes.md -o cited.md --chunk-n 2 --refs-per-chunk 2 --bibtex combined

Optional env: ARXIV_CONTACT_EMAIL, ARXIV_REQUEST_DELAY (default 3.0).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from src.bibtex import fetch_bibtex
from src.md_smart import process_markdown_smart
from src.ref_logging import get_logger, setup_logging
from src.search import (
    PaperEntry,
    search_all_apis,
    search_bulk_unified,
    search_fuzzy_title_unified,
)

_log = get_logger("app")

_PROJECT_ROOT = Path(__file__).resolve().parent
_REFS_DIR = _PROJECT_ROOT / "refs"


def resolve_bib_output(path_str: str) -> Path:
    """
    Every **relative** output path is placed under ``refs/`` (next to ``app.py``).
    Only an **absolute** path is written exactly where given.
    """
    s = path_str.strip().strip("\"'")
    p = Path(s)
    if p.is_absolute():
        return p
    rel = Path(os.path.normpath(s))
    if rel.is_absolute():
        return rel
    if ".." in rel.parts:
        raise ValueError("Output path must not contain '..'")
    return _REFS_DIR / rel


def _write_bib(path: Path, chunks: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "% Thank you to arXiv for use of its open access interoperability.\n\n"
    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n\n".join(chunks))


def _arxiv_id_for_filename(entry: object) -> str:
    """Stable filesystem-safe id from a feedparser entry (e.g. 2401.12345v2)."""
    eid = getattr(entry, "id", "") or ""
    if "/abs/" in eid:
        aid = eid.split("/abs/")[-1]
    else:
        aid = eid.rsplit("/", 1)[-1]
    for c in '\\/:*?"<>|':
        aid = aid.replace(c, "_")
    return aid or "unknown"


def _title_slug(entry: object, max_len: int = 120) -> str:
    """Filesystem-safe slug from paper title; falls back to arXiv id."""
    raw = getattr(entry, "title", "") or ""
    raw = " ".join(raw.split())
    s = raw.replace("–", "-").replace("—", "-")
    s = re.sub(r"[^\w\s\-]+", "", s, flags=re.UNICODE)
    s = re.sub(r"[\s\-_]+", "_", s)
    s = s.strip("_").lower()
    if len(s) > max_len:
        s = s[:max_len].strip("_")
    return s if s else _arxiv_id_for_filename(entry)


def _split_output_dir(resolved_output: Path, session: bool) -> Path:
    """
    Directory for split files: parent of the combined ``-o`` path (usually ``refs/``).
    With ``session``, append ``YYYYMMDD_HHMMSS``.
    """
    if resolved_output.suffix.lower() == ".bib":
        base = resolved_output.parent
    else:
        base = resolved_output
    if session:
        base = base / datetime.now().strftime("%Y%m%d_%H%M%S")
    return base


def _entry_id_for_filename(entry: PaperEntry) -> str:
    """Extract safe ID for filename from PaperEntry."""
    eid = entry.id
    for c in '\\/:*?"<>|':
        eid = eid.replace(c, "_")
    return eid or "unknown"


def cmd_bulk(args: argparse.Namespace) -> int:
    q = args.query.strip()
    if not q:
        print("Error: --query must not be empty.", file=sys.stderr)
        _log.error("bulk: empty query")
        return 2
    try:
        out = resolve_bib_output(args.output)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    api = getattr(args, "api", "arxiv")
    _log.info("bulk search api=%s query=%r max_results=%s", api, q, args.max_results)
    print(f"Searching {api}: {q!r} (max {args.max_results})")

    try:
        if api == "all":
            papers = search_all_apis(q, max_results=args.max_results)
        else:
            papers = search_bulk_unified(
                q,
                api=api,
                max_results=args.max_results,
                sort_by=args.sort_by,
                sort_order=args.sort_order,
            )
    except Exception as e:
        _log.exception("bulk search failed: %s", e)
        print(f"Search failed: {e}", file=sys.stderr)
        return 1
    _log.info("bulk found %s entries", len(papers))
    print(f"Found {len(papers)} entries. Fetching BibTeX...")
    split = getattr(args, "split", False)
    session = getattr(args, "session", False)

    if split:
        out_dir = _split_output_dir(out, session)
        out_dir.mkdir(parents=True, exist_ok=True)
        _log.info("bulk split mode -> %s", out_dir.resolve())
        print(f"Split mode (title filenames): {out_dir.resolve()}")
        written = 0
        used_slugs: dict[str, int] = {}
        for i, paper in enumerate(papers, 1):
            title = paper.title.replace("\n", " ")
            print(f"  [{i}/{len(papers)}] {title[:72]}...")
            try:
                bib = fetch_bibtex(paper)
            except Exception as e:
                _log.warning("bulk BibTeX failed [%s/%s]: %s", i, len(papers), e)
                print(f"    Failed: {e}", file=sys.stderr)
                continue
            slug = _title_slug(paper.raw)
            n = used_slugs.get(slug, 0)
            used_slugs[slug] = n + 1
            short_id = _entry_id_for_filename(paper)
            if n == 0:
                fname = f"{slug}.bib"
            else:
                fname = f"{slug}_{short_id}.bib" if n == 1 else f"{slug}_{short_id}_{n + 1}.bib"
            fpath = out_dir / fname
            _write_bib(fpath, [bib])
            written += 1
        _log.info("bulk split wrote %s files under %s", written, out_dir.resolve())
        print(f"Done: {written} files in {out_dir.resolve()}")
        return 0

    bib_chunks: list[str] = []
    for i, paper in enumerate(papers, 1):
        title = paper.title.replace("\n", " ")
        print(f"  [{i}/{len(papers)}] {title[:72]}...")
        try:
            bib_chunks.append(fetch_bibtex(paper))
        except Exception as e:
            _log.warning("bulk BibTeX failed [%s/%s]: %s", i, len(papers), e)
            print(f"    Failed: {e}", file=sys.stderr)
    _write_bib(out, bib_chunks)
    _log.info("bulk wrote %s entries -> %s", len(bib_chunks), out.resolve())
    print(f"Done: {len(bib_chunks)} entries -> {out.resolve()}")
    return 0


def cmd_match(args: argparse.Namespace) -> int:
    title = (args.title or "").strip()
    if not title and args.interactive:
        title = input("Enter partial / full paper title:\n> ").strip()
    if not title:
        print("Error: provide --title or use --interactive.", file=sys.stderr)
        _log.error("match: missing title")
        return 2

    try:
        out = resolve_bib_output(args.output)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    api = getattr(args, "api", "arxiv")
    _log.info("match api=%s title=%r max_results=%s pick=%s", api, title, args.max_results, args.pick)
    print(f"Searching {api} for titles similar to: {title!r}\n")

    try:
        ranked = search_fuzzy_title_unified(title, api=api, max_results=args.max_results)
    except Exception as e:
        _log.exception("match search failed: %s", e)
        print(f"Search failed: {e}", file=sys.stderr)
        return 1
    if not ranked:
        _log.warning("match: no results for title=%r", title)
        print("No results. Try shorter or different keywords.")
        return 1

    top_k = min(5, len(ranked))
    print("Top matches:\n" + "=" * 60)
    for i in range(top_k):
        score, entry = ranked[i]
        t = entry.title.strip().replace("\n", " ")
        authors = entry.authors
        year = entry.year if entry.year else "?"
        print(f"{i + 1}. [{score:.3f}] {t}")
        print(f"   Authors: {authors} | Year: {year} | {entry.id}\n")

    pick = args.pick
    if args.interactive:
        default_pick = max(1, min(pick, len(ranked)))
        raw = input(f"Download BibTeX for which rank 1–{min(5, len(ranked))} [{default_pick}]: ").strip()
        if raw:
            try:
                pick = int(raw)
            except ValueError:
                print("Invalid rank.", file=sys.stderr)
                return 1

    if pick < 1 or pick > len(ranked):
        print(f"Error: pick must be 1..{len(ranked)}", file=sys.stderr)
        return 2
    pick_idx = pick - 1
    score, chosen = ranked[pick_idx]
    _log.info("match selected rank=%s similarity=%.3f id=%s", pick, score, chosen.id)

    if not args.yes and score < args.min_similarity:
        if args.interactive:
            raw = input(
                f"Similarity {score:.2f} < {args.min_similarity}. Continue anyway? [y/N]: "
            ).strip().lower()
            if raw not in ("y", "yes"):
                return 1
        else:
            print(
                f"Error: similarity {score:.3f} < {args.min_similarity}. "
                f"Use --yes, --pick N, --min-similarity, or --interactive.",
                file=sys.stderr,
            )
            return 1

    print(f"\nFetching BibTeX (rank {pick_idx + 1}, similarity {score:.3f})...")
    try:
        bib = fetch_bibtex(chosen)
    except Exception as e:
        _log.exception("match BibTeX failed: %s", e)
        print(f"Failed: {e}", file=sys.stderr)
        return 1
    _write_bib(out, [bib])
    _log.info("match saved -> %s", out.resolve())
    print(f"Saved -> {out.resolve()}")
    print(f"Title: {chosen.title.strip().replace(chr(10), ' ')}")
    return 0


def cmd_smart(args: argparse.Namespace) -> int:
    inp = Path(args.input_md).expanduser()
    if not inp.is_file() and not inp.is_absolute():
        alt = _PROJECT_ROOT / args.input_md
        if alt.is_file():
            inp = alt
    if not inp.is_file():
        print(f"Error: input file not found: {inp}", file=sys.stderr)
        return 2
    try:
        out_md = resolve_bib_output(args.output)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    chunk_by = args.chunk_by
    if chunk_by not in ("paragraphs", "sentences"):
        print("Error: --chunk-by must be paragraphs or sentences.", file=sys.stderr)
        return 2
    ref_fmt = args.ref_format
    if ref_fmt not in ("simple", "numbered", "links", "compact"):
        print("Error: invalid --ref-format.", file=sys.stderr)
        return 2
    bib_mode = args.bibtex
    if bib_mode not in ("none", "combined", "split"):
        print("Error: invalid --bibtex.", file=sys.stderr)
        return 2

    bib_combined: Path | None = None
    bib_split: Path | None = None
    if bib_mode == "combined":
        if args.bibtex_output:
            try:
                bib_combined = resolve_bib_output(args.bibtex_output)
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 2
        else:
            bib_combined = out_md.with_suffix(".bib")
    elif bib_mode == "split":
        bib_split = out_md.parent / f"{out_md.stem}_bib"

    api = getattr(args, "api", "arxiv")
    print(f"Smart cite: {inp.resolve()} -> {out_md.resolve()}")
    print(f"  api={api} chunk-by={chunk_by} n={args.chunk_n} refs/chunk={args.refs_per_chunk} format={ref_fmt} bibtex={bib_mode}")
    try:
        _, ordered = process_markdown_smart(
            inp,
            out_md,
            chunk_by=chunk_by,
            chunk_n=args.chunk_n,
            refs_per_chunk=args.refs_per_chunk,
            query_max_words=args.query_words,
            ref_format=ref_fmt,
            bib_mode=bib_mode,
            bibtex_combined_path=bib_combined,
            bibtex_split_dir=bib_split,
            api=api,
        )
    except Exception as e:
        _log.exception("smart mode failed: %s", e)
        print(f"Failed: {e}", file=sys.stderr)
        return 1
    print(f"Done: {len(ordered)} unique citations -> {out_md.resolve()}")
    if bib_combined:
        print(f"  BibTeX (combined): {bib_combined.resolve()}")
    if bib_split:
        print(f"  BibTeX (split): {bib_split.resolve()}/")
    return 0


def main(argv: list[str] | None = None) -> int:
    log_parent = argparse.ArgumentParser(add_help=False)
    log_parent.add_argument(
        "--log-level",
        default=os.environ.get("REFSPROJ_LOG_LEVEL", "INFO"),
        help="Logging level (default: INFO or REFSPROJ_LOG_LEVEL).",
    )
    log_parent.add_argument(
        "--no-log-file",
        action="store_true",
        help="Log only to stderr (ignored if logging.conf INI is active).",
    )
    log_parent.add_argument("-v", "--verbose", action="store_true", help="Shorthand for --log-level DEBUG.")

    parser = argparse.ArgumentParser(
        description="arXiv BibTeX downloader, fuzzy match, and smart markdown citations.",
        parents=[log_parent],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_bulk = sub.add_parser("bulk", help="Search arXiv and save many BibTeX entries.")
    p_bulk.add_argument(
        "-q",
        "--query",
        required=True,
        help='arXiv search_query, e.g. all:"neural network" OR cat:cs.CL AND all:survey',
    )
    p_bulk.add_argument("-n", "--max-results", type=int, default=50)
    p_bulk.add_argument(
        "-o",
        "--output",
        default="arxiv_references.bib",
        help="Combined: single .bib under refs/. Split: sets parent folder (default refs/ via this path's parent).",
    )
    p_bulk.add_argument(
        "--split",
        action="store_true",
        help="One .bib per paper, filename from sanitized title (collisions get arXiv id suffix).",
    )
    p_bulk.add_argument(
        "--session",
        action="store_true",
        help="With --split, write under refs/YYYYMMDD_HHMMSS/ (or under -o's parent + that subfolder).",
    )
    p_bulk.add_argument("--sort-by", default="submittedDate", help="e.g. submittedDate, relevance, lastUpdatedDate")
    p_bulk.add_argument("--sort-order", default="descending", choices=("ascending", "descending"))
    p_bulk.add_argument(
        "--api",
        choices=("arxiv", "crossref", "semantic_scholar", "all"),
        default="arxiv",
        help="API to search: arxiv (default), crossref, semantic_scholar, or all (merges results).",
    )
    p_bulk.set_defaults(func=cmd_bulk)

    p_match = sub.add_parser("match", help="Fuzzy-match one paper by title and save one .bib.")
    p_match.add_argument("-t", "--title", default="", help="Partial or full title (optional if --interactive).")
    p_match.add_argument("-n", "--max-results", type=int, default=100, help="Candidate pool size from arXiv.")
    p_match.add_argument(
        "-o",
        "--output",
        default="paper_citation.bib",
        help="Output .bib under refs/ unless an absolute path (default: paper_citation.bib).",
    )
    p_match.add_argument(
        "--pick",
        type=int,
        default=1,
        help="1-based rank after fuzzy sort (default: best match).",
    )
    p_match.add_argument(
        "--min-similarity",
        type=float,
        default=0.6,
        help="Abort unless similarity >= this (unless --yes).",
    )
    p_match.add_argument("-y", "--yes", action="store_true", help="Skip low-similarity guard.")
    p_match.add_argument("-i", "--interactive", action="store_true", help="Prompt for title / confirmations.")
    p_match.add_argument(
        "--api",
        choices=("arxiv", "crossref", "semantic_scholar"),
        default="arxiv",
        help="API to search: arxiv (default), crossref, or semantic_scholar.",
    )
    p_match.set_defaults(func=cmd_match)

    p_smart = sub.add_parser(
        "smart",
        help="Chunk a .md file, search arXiv per chunk, insert [n] cites + reference list (new file).",
    )
    p_smart.add_argument("input_md", help="Path to input Markdown file.")
    p_smart.add_argument(
        "-o",
        "--output",
        default="cited.md",
        help="Output .md under refs/ unless absolute (default: cited.md).",
    )
    p_smart.add_argument(
        "--chunk-by",
        choices=("paragraphs", "sentences"),
        default="sentences",
        help="Split body into paragraphs or sentences before grouping (default: sentences).",
    )
    p_smart.add_argument(
        "--chunk-n",
        type=int,
        default=1,
        metavar="N",
        help="Group N paragraphs (or N sentences) into one chunk (default: 1).",
    )
    p_smart.add_argument(
        "--refs-per-chunk",
        type=int,
        default=1,
        metavar="N",
        help="Top N arXiv hits per chunk (relevance-sorted, default: 1).",
    )
    p_smart.add_argument(
        "--query-words",
        type=int,
        default=8,
        metavar="N",
        help="Use up to N keywords from each chunk for the arXiv query.",
    )
    p_smart.add_argument(
        "--ref-format",
        choices=("simple", "numbered", "links", "compact"),
        default="simple",
        help="Style of the ## References section.",
    )
    p_smart.add_argument(
        "--bibtex",
        choices=("none", "combined", "split"),
        default="none",
        help="none | combined (one .bib) | split (one .bib per paper in <output_stem>_bib/).",
    )
    p_smart.add_argument(
        "--bibtex-output",
        default="",
        help="Combined .bib path (relative -> refs/). Default: same stem as -o with .bib.",
    )
    p_smart.add_argument(
        "--api",
        choices=("arxiv", "crossref", "semantic_scholar", "all"),
        default="arxiv",
        help="API to search per chunk: arxiv (default), crossref, semantic_scholar, or all (merges results).",
    )
    p_smart.set_defaults(func=cmd_smart)

    args = parser.parse_args(argv)
    level = "DEBUG" if args.verbose else args.log_level
    setup_logging(level=level, enable_file=not args.no_log_file)
    _log.debug("sys.argv=%s", sys.argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
