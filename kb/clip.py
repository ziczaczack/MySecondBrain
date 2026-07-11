"""Save fetched content as markdown clip notes.

One note per clip, YAML frontmatter carrying identity (``url``) and the
``kb-clipped: true`` marker that the inbox processor uses to skip
already-expanded notes. :func:`render_note` is the single source of truth
for that format — the inbox expansion (kb.inbox) reuses it verbatim so
clipped-on-PC and expanded-from-phone notes look identical.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

from .fetch import FetchedDoc

# Filename slugs: keep word characters (\w matches CJK under re.UNICODE),
# collapse every other run into a single dash.
_SLUG_BAD = re.compile(r"[^\w]+", re.UNICODE)
_SLUG_MAX = 60


def slug(title: str) -> str:
    """Filesystem-safe filename stem from *title*; ``"clip"`` when empty."""
    s = _SLUG_BAD.sub("-", title).strip("-_")
    s = s[:_SLUG_MAX].strip("-_")
    return s or "clip"


def render_note(doc: FetchedDoc, clipped_date: str) -> str:
    """Render *doc* as a markdown note with the kb clip frontmatter."""
    title = (doc.title or doc.url or "Untitled").replace('"', "'")
    lines = ["---", f'title: "{title}"']
    if doc.url:
        lines.append(f"url: {doc.url}")
    lines += [
        f"clipped: {clipped_date}",
        "kb-clipped: true",
        "---",
        "",
        doc.text.rstrip(),
        "",
    ]
    return "\n".join(lines)


def find_existing(clips_dir: Path, url: str) -> Optional[Path]:
    """Path of a note in *clips_dir* whose frontmatter claims *url*, else None.

    Linear scan over the folder's ``.md`` files, reading only the head of
    each — fine at personal-knowledge-base scale. Unreadable files are
    skipped, never fatal.
    """
    needle = f"url: {url}"
    for p in sorted(clips_dir.glob("*.md")):
        try:
            head = p.read_text(encoding="utf-8", errors="replace")[:2000]
        except Exception:
            continue
        if any(line.strip() == needle for line in head.splitlines()[:10]):
            return p
    return None


def save_clip(doc: FetchedDoc, clips_dir: str) -> Optional[Path]:
    """Write *doc* into *clips_dir*; return the path, or None if already saved.

    Dedup is by frontmatter URL (text clips with ``url=""`` never dedupe).
    Filename collisions between different URLs append ``-2``, ``-3``, …
    The folder is created on demand.
    """
    folder = Path(clips_dir)
    folder.mkdir(parents=True, exist_ok=True)
    if doc.url and find_existing(folder, doc.url) is not None:
        return None

    date = time.strftime("%Y-%m-%d", time.localtime(doc.fetched_at))
    base = slug(doc.title or doc.url)
    path = folder / f"{base}.md"
    counter = 2
    while path.exists():
        path = folder / f"{base}-{counter}.md"
        counter += 1
    path.write_text(render_note(doc, date), encoding="utf-8")
    return path
