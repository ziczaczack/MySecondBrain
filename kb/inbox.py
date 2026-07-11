"""Inbox processing: expand bare-URL notes into full clip notes, in place.

The phone-capture path: the user shares a URL into an ``Inbox/`` folder of
their (synced) Obsidian vault; on the PC, ``kb watch`` calls
:func:`process_inbox`, which fetches the content and rewrites the note —
frontmatter identical to a ``kb clip`` note (via :func:`kb.clip.render_note`),
so the ``kb-clipped: true`` marker suppresses reprocessing and the expanded
note syncs back to the phone as a read-later copy.

Safety rule: only a note whose entire body is a single bare URL (or a
markdown link containing only a URL) is ever rewritten. Anything with prose
is untouchable.

Fetch failures are retried across watch cycles; after ``_MAX_FAILS``
failures the note is stamped ``kb-clip-failed: true`` (delete that line to
re-trigger). Fail counts persist in ``<kb_home>/inbox_state.json``.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from . import config, fetch
from .clip import render_note

_MAX_FAILS = 3

_URL_LINE = re.compile(r"^https?://\S+$")
_MD_LINK_LINE = re.compile(r"^\[[^\]]*\]\((https?://\S+)\)$")


def _split_frontmatter(text: str) -> tuple:
    """Return ``(body, frontmatter)``; frontmatter is ``""`` when absent.

    A frontmatter block is a leading ``---`` line closed by another ``---``
    line. An unterminated opener is treated as body, not frontmatter.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text, ""
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[i + 1:]), "\n".join(lines[1:i])
    return text, ""


def parse_bare_url_note(text: str) -> Optional[str]:
    """Return the URL when *text* is a bare-URL note, else ``None``.

    A bare-URL note contains, after stripping frontmatter and blank lines,
    exactly one line that is a plain http(s) URL or a markdown link wrapping
    one. Notes already carrying ``kb-clipped`` / ``kb-clip-failed`` markers
    are never candidates.
    """
    body, frontmatter = _split_frontmatter(text)
    if "kb-clipped:" in frontmatter or "kb-clip-failed:" in frontmatter:
        return None
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if len(lines) != 1:
        return None
    line = lines[0]
    if _URL_LINE.match(line):
        return line
    m = _MD_LINK_LINE.match(line)
    return m.group(1) if m else None


def _state_path() -> Path:
    return config.kb_home() / "inbox_state.json"


def _load_state() -> dict:
    try:
        data = json.loads(_state_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    home = config.kb_home()
    try:
        home.mkdir(parents=True, exist_ok=True)
        _state_path().write_text(
            json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass  # state is an optimisation; losing it only means extra retries


def _mark_failed(path: Path, url: str) -> None:
    """Stamp *path* as given-up: keep the URL, add the kb-clip-failed marker."""
    content = "\n".join(
        ["---", f"url: {url}", "kb-clip-failed: true", "---", "", url, ""]
    )
    try:
        path.write_text(content, encoding="utf-8")
    except Exception:
        pass


def process_inbox(
    folder: str,
    timeout: float = 15.0,
    fetcher: Optional[Callable] = None,
) -> int:
    """Expand every bare-URL note under *folder* in place; return the count.

    *fetcher* defaults to :func:`kb.fetch.fetch_url` (injectable for tests).
    All errors are per-note and non-fatal: an unreadable note is skipped, a
    failed fetch increments its retry count, and after ``_MAX_FAILS``
    failures the note is stamped and abandoned.
    """
    if fetcher is None:
        fetcher = fetch.fetch_url
    root = Path(folder)
    if not root.is_dir():
        return 0

    state = _load_state()
    expanded = 0
    for path in sorted(root.rglob("*.md")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        url = parse_bare_url_note(text)
        if url is None:
            continue

        key = str(path)
        doc = fetcher(url, timeout=timeout)
        if doc is None:
            fails = int(state.get(key, 0)) + 1
            if fails >= _MAX_FAILS:
                _mark_failed(path, url)
                state.pop(key, None)
                print(
                    f"kb inbox: giving up on {path.name} "
                    f"after {_MAX_FAILS} failed fetches of {url}",
                    file=sys.stderr,
                )
            else:
                state[key] = fails
            continue

        date = time.strftime("%Y-%m-%d", time.localtime(doc.fetched_at))
        try:
            path.write_text(render_note(doc, date), encoding="utf-8")
        except Exception:
            continue
        state.pop(key, None)
        expanded += 1

    _save_state(state)
    return expanded
