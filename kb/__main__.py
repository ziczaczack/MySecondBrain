"""Command-line entrypoint: ``python -m kb ingest|query``."""

from __future__ import annotations

import argparse
import json
import re
import sys

from .ingest import DEFAULT_INDEX_DIR, ingest
from .query import IncompatibleIndexError, query
from .status import status as status_fn


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kb",
        description="A pure-local personal knowledge base.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser(
        "ingest", help="Embed and index a directory of .md/.txt/.py/.js (code + notes)."
    )
    ingest_p.add_argument("dir", help="Directory to walk recursively.")
    ingest_p.add_argument(
        "--index-dir",
        default=DEFAULT_INDEX_DIR,
        help=f"Where to write the index (default: {DEFAULT_INDEX_DIR}).",
    )
    ingest_p.add_argument(
        "--rebuild",
        action="store_true",
        default=False,
        help="Ignore any existing index and re-embed everything from scratch.",
    )

    query_p = sub.add_parser("query", help="Search the index.")
    query_p.add_argument("question", help="Natural-language query string.")
    query_p.add_argument(
        "--index-dir",
        default=DEFAULT_INDEX_DIR,
        help=f"Index directory to search (default: {DEFAULT_INDEX_DIR}).",
    )
    query_p.add_argument(
        "-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5).",
    )
    query_p.add_argument(
        "--since",
        default=None,
        help="Only results modified within a window, e.g. '7d', '30d', "
        "or an absolute 'YYYY-MM-DD' date.",
    )
    query_p.add_argument(
        "--kind",
        choices=["code", "note"],
        default=None,
        help="Filter results by kind: 'code' or 'note'.",
    )
    query_p.add_argument(
        "--hybrid",
        action="store_true",
        help="Fuse semantic + keyword (BM25) ranking via RRF.",
    )
    query_p.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON.",
    )

    status_p = sub.add_parser("status", help="Show statistics about an existing index.")
    status_p.add_argument(
        "--index-dir",
        default=DEFAULT_INDEX_DIR,
        help=f"Index directory (default: {DEFAULT_INDEX_DIR}).",
    )

    return parser


def _run_ingest(args: argparse.Namespace) -> int:
    ingest(args.dir, index_dir=args.index_dir, rebuild=args.rebuild)
    return 0


def _highlight(excerpt: str, terms: list[str]) -> str:
    """Wrap each matched term occurrence in ``excerpt`` with markdown bold.

    Matching is case-insensitive; the original casing in the excerpt is kept.
    """
    highlighted = excerpt
    for term in terms:
        if not term:
            continue
        highlighted = re.sub(
            re.escape(term),
            lambda m: f"**{m.group(0)}**",
            highlighted,
            flags=re.IGNORECASE,
        )
    return highlighted


def _run_query(args: argparse.Namespace) -> int:
    try:
        results = query(
            args.question,
            index_dir=args.index_dir,
            k=args.k,
            since=args.since,
            kind=args.kind,
            hybrid=args.hybrid,
        )
    except FileNotFoundError:
        print(
            f"No index found in '{args.index_dir}'. "
            "Run `python -m kb ingest <dir>` first to build the index.",
            file=sys.stderr,
        )
        return 1
    except IncompatibleIndexError as err:
        print(str(err), file=sys.stderr)
        return 1

    if args.json:
        # Machine-readable view: results are already JSON-safe (floats, strs,
        # lists). No human header is printed in this mode.
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    if not results:
        print("No results found.")
        return 0

    print(f"Top-{len(results)} results for: {args.question}")
    for rank, r in enumerate(results, 1):
        print(
            f"{rank}. [{r.get('kind', 'note')}] {r['filename']}:{r['start_line']}   "
            f"[{r.get('date', '')}]   ({r['score']:.4f})"
        )
        if r["excerpt"]:
            print(f"   {_highlight(r['excerpt'], r.get('matched_terms', []))}")
    return 0


def _humanize_bytes(n: int) -> str:
    """Render a byte count as a compact human-readable size (B/KB/MB/GB)."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def _run_status(args: argparse.Namespace) -> int:
    st = status_fn(args.index_dir)
    if not st.get("exists"):
        print(
            f"No index found in '{args.index_dir}'. "
            "Run `python -m kb ingest <dir>` first to build the index.",
            file=sys.stderr,
        )
        return 1

    kinds = st.get("kinds", {})
    kinds_str = (
        ", ".join(f"{kind}: {count}" for kind, count in sorted(kinds.items()))
        or "(none)"
    )
    last = st.get("last_ingest_date") or "(unknown)"

    print(f"Index directory: {st['index_dir']}")
    print(f"Files:           {st['files']}")
    print(f"Chunks:          {st['chunks']}")
    print(f"Kinds:           {kinds_str}")
    print(f"Index size:      {_humanize_bytes(st.get('index_bytes', 0))}")
    print(f"Last ingest:     {last}")
    return 0


def _force_utf8_output() -> None:
    """Ensure non-ASCII queries/notes print on consoles that default to a
    legacy code page (e.g. Windows cp1252). No-op where reconfigure is absent."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass


def main() -> None:
    _force_utf8_output()
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        sys.exit(_run_ingest(args))
    elif args.command == "query":
        sys.exit(_run_query(args))
    elif args.command == "status":
        sys.exit(_run_status(args))
    else:  # pragma: no cover - argparse enforces a valid subcommand.
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
