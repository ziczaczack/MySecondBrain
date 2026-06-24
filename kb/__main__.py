"""Command-line entrypoint: ``python -m kb ingest|query``."""

from __future__ import annotations

import argparse
import json
import re
import sys

from . import config
from .ingest import DEFAULT_INDEX_DIR, ingest
from .llm import KbLLMError
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
        default=None,
        help="Where to write the index (default: the managed knowledge base).",
    )
    ingest_p.add_argument(
        "--rebuild",
        action="store_true",
        default=False,
        help="Ignore any existing index and re-embed everything from scratch.",
    )

    bm_p = sub.add_parser(
        "ingest-bookmarks",
        help="Embed and index bookmarks from a Chrome/Edge 'Bookmarks' JSON file.",
    )
    bm_p.add_argument("path", help="Path to the Chrome/Edge 'Bookmarks' JSON file.")
    bm_p.add_argument(
        "--index-dir",
        default=None,
        help="Where to write the index (default: the managed knowledge base).",
    )
    bm_p.add_argument(
        "--rebuild",
        action="store_true",
        default=False,
        help="Ignore any existing index and re-embed everything from scratch.",
    )

    add_p = sub.add_parser(
        "add", help="Register a folder (or --bookmarks file) as a source and index it."
    )
    add_p.add_argument(
        "path",
        help="Folder of notes/code, or a Chrome/Edge Bookmarks file with --bookmarks.",
    )
    add_p.add_argument(
        "--bookmarks",
        action="store_true",
        help="Treat <path> as a Chrome/Edge Bookmarks JSON file.",
    )
    add_p.add_argument(
        "--index-dir",
        default=None,
        help="Override the managed index location (advanced).",
    )

    query_p = sub.add_parser("query", help="Search the index.")
    _add_query_args(query_p)

    ask_p = sub.add_parser(
        "ask", help="Ask a question; answer is synthesized by an LLM with citations."
    )
    _add_query_args(ask_p)
    ask_p.add_argument(
        "--no-synthesis",
        action="store_true",
        help="Skip LLM synthesis and return raw retrieval results (same as query).",
    )

    sub.add_parser("sources", help="List registered sources.")

    watch_p = sub.add_parser(
        "watch", help="Watch registered folders and auto-reindex on change."
    )
    watch_p.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="Polling interval in seconds (default: 3).",
    )
    watch_p.add_argument(
        "--index-dir",
        default=None,
        help="Override the managed index location (advanced).",
    )

    status_p = sub.add_parser("status", help="Show statistics about an existing index.")
    status_p.add_argument(
        "--index-dir",
        default=None,
        help="Index directory (default: the managed knowledge base).",
    )

    return parser


def _add_query_args(p: argparse.ArgumentParser) -> None:
    """Attach the shared query/ask options to *p* (same flags for both)."""
    p.add_argument("question", help="Natural-language query string.")
    p.add_argument(
        "--index-dir",
        default=None,
        help="Index directory to search (default: the managed knowledge base).",
    )
    p.add_argument(
        "-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5).",
    )
    p.add_argument(
        "--since",
        default=None,
        help="Only results modified within a window, e.g. '7d', '30d', "
        "or an absolute 'YYYY-MM-DD' date.",
    )
    p.add_argument(
        "--kind",
        choices=["code", "note"],
        default=None,
        help="Filter results by kind: 'code' or 'note'.",
    )
    p.add_argument(
        "--hybrid",
        action="store_true",
        help="Fuse semantic + keyword (BM25) ranking via RRF.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON.",
    )


def _resolve_index_dir(value):
    """Return *value* when given, else the managed default index location."""
    return value if value else config.default_index_dir()


def _run_ingest(args: argparse.Namespace) -> int:
    index_dir = _resolve_index_dir(args.index_dir)
    ingest(args.dir, index_dir=index_dir, rebuild=args.rebuild)
    return 0


def _run_ingest_bookmarks(args: argparse.Namespace) -> int:
    # Source selection lives in the CLI layer; ingest.py stays source-agnostic.
    from .source import BookmarkSource
    from .ingest import _ingest_from_source

    index_dir = _resolve_index_dir(args.index_dir)
    _ingest_from_source(
        BookmarkSource(args.path),
        index_dir=index_dir,
        rebuild=args.rebuild,
        label=args.path,
    )
    return 0


def _run_add(args: argparse.Namespace) -> int:
    index_dir = _resolve_index_dir(args.index_dir)
    if args.bookmarks:
        config.add_source("bookmarks", args.path)
        from .ingest import _ingest_from_source
        from .source import BookmarkSource

        _ingest_from_source(
            BookmarkSource(args.path), index_dir=index_dir, label=args.path
        )
        print(f"Added bookmarks source: {args.path}")
    else:
        config.add_source("files", args.path)
        ingest(args.path, index_dir=index_dir)
        print(f"Added files source: {args.path}")
    return 0


def _run_sources(args: argparse.Namespace) -> int:
    srcs = config.load_sources()
    if not srcs:
        print("No sources registered yet. Add one with `python -m kb add <folder>`.")
    else:
        for s in srcs:
            print(f"- [{s.get('kind','?')}] {s.get('path','')}")
    return 0


def _run_watch(args: argparse.Namespace) -> int:
    from . import watch as watch_mod

    watch_mod.watch(
        index_dir=_resolve_index_dir(args.index_dir), interval=args.interval
    )
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


def _run_ask(args: argparse.Namespace) -> int:
    if getattr(args, "no_synthesis", False):
        return _run_query(args)

    from .answer import answer

    index_dir = _resolve_index_dir(args.index_dir)
    try:
        result = answer(
            args.question,
            index_dir=index_dir,
            k=args.k,
            since=args.since,
            kind=args.kind,
            hybrid=args.hybrid,
        )
    except FileNotFoundError:
        print(
            f"No index found in '{index_dir}'. "
            "Run `python -m kb ingest <dir>` first to build the index.",
            file=sys.stderr,
        )
        return 1
    except IncompatibleIndexError as err:
        print(str(err), file=sys.stderr)
        return 1
    except KbLLMError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "answer": result["answer"],
                    "citations": result["citations"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print(result["answer"])
    if result["citations"]:
        print("\nSources:")
        for c in result["citations"]:
            print(f"  [{c['n']}] {c['filename']}:{c['start_line']}")
    return 0


def _run_query(args: argparse.Namespace) -> int:
    index_dir = _resolve_index_dir(args.index_dir)
    try:
        results = query(
            args.question,
            index_dir=index_dir,
            k=args.k,
            since=args.since,
            kind=args.kind,
            hybrid=args.hybrid,
        )
    except FileNotFoundError:
        print(
            f"No index found in '{index_dir}'. "
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
    index_dir = _resolve_index_dir(args.index_dir)
    st = status_fn(index_dir)
    if not st.get("exists"):
        print(
            f"No index found in '{index_dir}'. "
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
    elif args.command == "ingest-bookmarks":
        sys.exit(_run_ingest_bookmarks(args))
    elif args.command == "add":
        sys.exit(_run_add(args))
    elif args.command == "query":
        sys.exit(_run_query(args))
    elif args.command == "ask":
        sys.exit(_run_ask(args))
    elif args.command == "sources":
        sys.exit(_run_sources(args))
    elif args.command == "watch":
        sys.exit(_run_watch(args))
    elif args.command == "status":
        sys.exit(_run_status(args))
    else:  # pragma: no cover - argparse enforces a valid subcommand.
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
