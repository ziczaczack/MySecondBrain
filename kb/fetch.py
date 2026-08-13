"""Content fetcher for the capture layer (``kb clip`` and inbox expansion).

Turns a URL into a :class:`FetchedDoc` (title + main text). Two paths:

* YouTube watch URLs -> transcript via ``youtube-transcript-api``.
* Any other http(s) URL -> HTML via stdlib urllib, main text via
  ``trafilatura``.

Both third-party libraries belong to the ``[clip]`` extra and are imported
lazily inside functions (same pattern as ``_extract_pdf`` in kb.source), so
the core package stays pure-Python. Every failure mode — missing library,
network error, timeout, unparseable page — returns ``None`` rather than
raising, mirroring the defensive posture of :mod:`kb.source`.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional

_USER_AGENT = "Mozilla/5.0 (kb-clip)"

# youtube.com/watch?v=<id> (v= anywhere in the query) or youtu.be/<id>.
_YT_RE = re.compile(
    r"^https?://(?:www\.|m\.)?"
    r"(?:youtube\.com/watch\?(?:[^#]*&)?v=|youtu\.be/)"
    r"([\w-]{6,})"
)

# Preferred transcript languages, most-specific first.
_YT_LANGS = ["zh-Hans", "zh-Hant", "zh", "en"]


@dataclass
class FetchedDoc:
    """A fetched piece of content ready to be saved as a note."""

    title: str        # extracted title; may be ""
    text: str         # main text (plain/markdown)
    url: str          # original URL ("" for pasted-text clips)
    fetched_at: float # unix timestamp of the fetch


def youtube_video_id(url: str) -> Optional[str]:
    """Return the YouTube video id in *url*, or ``None`` if not a watch URL."""
    m = _YT_RE.match(url)
    return m.group(1) if m else None


def _http_get(url: str, timeout: float) -> Optional[bytes]:
    """GET *url* and return the raw body; ``None`` on any error."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        return None


def extract_article(html: str, url: str = "") -> Optional[tuple]:
    """Extract ``(title, text)`` from raw *html* via trafilatura.

    Returns ``None`` when trafilatura is not installed (the ``[clip]`` extra)
    or no main content could be identified. The title falls back to ``""``
    when metadata extraction fails — the text is the part that matters.
    """
    try:
        import trafilatura
    except ImportError:
        return None
    try:
        # Markdown, not plain text: the default output drops heading markers
        # and renumbers <ol> items as bullets, which costs a clip its outline
        # in Obsidian and, in a how-to, the order of the steps.
        text = trafilatura.extract(
            html,
            url=url or None,
            include_comments=False,
            output_format="markdown",
        )
    except Exception:
        return None
    if not text or not text.strip():
        return None
    title = ""
    try:
        meta = trafilatura.extract_metadata(html)
        title = (getattr(meta, "title", "") or "") if meta else ""
    except Exception:
        pass
    return title, text


def _fetch_transcript(video_id: str) -> Optional[str]:
    """Fetch a YouTube transcript as one string; ``None`` on any failure.

    Handles both youtube-transcript-api generations: the >=1.0 instance API
    (``YouTubeTranscriptApi().fetch``) and the older class-method API
    (``YouTubeTranscriptApi.get_transcript``).
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return None
    try:
        api = YouTubeTranscriptApi()
        if hasattr(api, "fetch"):  # >= 1.0
            fetched = api.fetch(video_id, languages=_YT_LANGS)
            return " ".join(s.text for s in fetched)
        data = YouTubeTranscriptApi.get_transcript(video_id, languages=_YT_LANGS)
        return " ".join(d["text"] for d in data)
    except Exception:
        return None


def _youtube_title(url: str, timeout: float) -> str:
    """Video title via the keyless oEmbed endpoint; ``""`` on any failure."""
    oembed = (
        "https://www.youtube.com/oembed?format=json&url="
        + urllib.parse.quote(url, safe="")
    )
    raw = _http_get(oembed, timeout)
    if raw is None:
        return ""
    try:
        return json.loads(raw.decode("utf-8", errors="replace")).get("title") or ""
    except Exception:
        return ""


def _fetch_youtube(url: str, video_id: str, timeout: float) -> Optional[FetchedDoc]:
    text = _fetch_transcript(video_id)
    if not text or not text.strip():
        return None
    return FetchedDoc(
        title=_youtube_title(url, timeout),
        text=text,
        url=url,
        fetched_at=time.time(),
    )


def _fetch_article(url: str, timeout: float) -> Optional[FetchedDoc]:
    raw = _http_get(url, timeout)
    if raw is None:
        return None
    html = raw.decode("utf-8", errors="replace")
    extracted = extract_article(html, url)
    if extracted is None:
        return None
    title, text = extracted
    return FetchedDoc(title=title, text=text, url=url, fetched_at=time.time())


def fetch_url(url: str, timeout: float = 15.0) -> Optional[FetchedDoc]:
    """Fetch *url* into a :class:`FetchedDoc`, or ``None`` on any failure."""
    if not url.lower().startswith(("http://", "https://")):
        return None
    video_id = youtube_video_id(url)
    if video_id:
        return _fetch_youtube(url, video_id, timeout)
    return _fetch_article(url, timeout)
