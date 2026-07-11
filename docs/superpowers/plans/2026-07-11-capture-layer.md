# kb 采集层(clip + inbox)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让用户能一条命令(PC)或一次分享(手机 → Obsidian Inbox)把网页/YouTube/纯文字收进 kb 知识库。

**Architecture:** 新模块 `kb/fetch.py`(URL → 正文,防御式,全部错误返回 None)、`kb/clip.py`(FetchedDoc → markdown 笔记落盘)、`kb/inbox.py`(裸 URL 笔记原地展开)。CLI 加 `kb clip` 子命令和 `kb add --inbox`;`kb watch` 每轮先跑 inbox 处理步。索引复用现有 `ingest`,不改 `ingest.py`。

**Tech Stack:** Python(stdlib urllib 抓取)+ optional extra `[clip]`:`trafilatura`(正文提取)、`youtube-transcript-api`(字幕)。

**Spec:** `docs/superpowers/specs/2026-07-11-capture-layer-design.md`

## Global Constraints

- `requires-python = ">=3.9"`;每个新文件顶部 `from __future__ import annotations`。
- 核心包纯 Python 红线:`trafilatura` / `youtube-transcript-api` 只放 `[clip]` extra,且**只在函数体内 lazy import**(模式同 `kb/source.py` 的 `_extract_pdf`)。
- 防御式姿态:fetch / inbox 的任何网络、解析、IO 错误一律返回 `None` / 跳过,绝不抛异常打断调用方。
- 测试绝不打真网络:HTML 用内联 fixture,HTTP 层与 YouTube API 一律 monkeypatch。
- 所有文件读写显式 `encoding="utf-8"`(Windows 主场)。
- 测试里凡触碰 config 的,必须 `monkeypatch.setenv("KB_HOME", str(tmp_path))` 隔离真实用户配置。
- 每个 Task 结束提交一次,commit message 结尾带 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`。

---

### Task 1: `kb/fetch.py` — 内容抓取器 + `[clip]` extra

**Files:**
- Create: `kb/fetch.py`
- Create: `tests/test_fetch.py`
- Modify: `pyproject.toml`(`[project.optional-dependencies]` 段)

**Interfaces:**
- Consumes: 仅 stdlib。
- Produces(后续 Task 依赖,签名必须一致):
  - `@dataclass FetchedDoc(title: str, text: str, url: str, fetched_at: float)`
  - `fetch_url(url: str, timeout: float = 15.0) -> Optional[FetchedDoc]`
  - `youtube_video_id(url: str) -> Optional[str]`
  - `extract_article(html: str, url: str = "") -> Optional[tuple]`(返回 `(title, text)` 或 `None`)
  - 可 monkeypatch 的内部缝:`_http_get(url, timeout)`、`_fetch_transcript(video_id)`、`_youtube_title(url, timeout)`

- [ ] **Step 1: 在 pyproject.toml 加 extras 并安装**

`pyproject.toml` 的 `[project.optional-dependencies]` 改为:

```toml
[project.optional-dependencies]
# Cited-answer synthesis (`kb ask`) talks to the Anthropic API.
synthesis = ["anthropic>=0.39"]
# Index PDF and .docx files in addition to plain text.
documents = ["pypdf>=4", "python-docx>=1"]
# Web/YouTube capture (`kb clip` and inbox expansion).
clip = ["trafilatura>=1.8", "youtube-transcript-api>=0.6"]
# Everything needed to run the test suite.
dev = ["pytest>=7.0", "fpdf2>=2.7", "pypdf>=4", "python-docx>=1", "trafilatura>=1.8"]
```

Run: `pip install -e ".[dev]"`
Expected: 安装成功,`python -c "import trafilatura"` 无报错。

- [ ] **Step 2: 写失败测试**

创建 `tests/test_fetch.py`:

```python
"""Tests for kb.fetch — no real network access anywhere in this file."""

from __future__ import annotations

import kb.fetch as fetch

# Realistic-enough article HTML: trafilatura needs a few substantial
# paragraphs before it recognises a main-content block.
ARTICLE_HTML = """<!DOCTYPE html>
<html><head><title>Postgres Partial Indexes Explained</title></head>
<body>
<nav><a href="/">Home</a> <a href="/about">About</a></nav>
<article>
<h1>Postgres Partial Indexes Explained</h1>
<p>A partial index covers only the rows that satisfy a predicate, which makes
it dramatically smaller than a full index when the predicate is selective.
This paragraph exists to give the extractor a substantial block of prose to
recognise as main content rather than boilerplate navigation.</p>
<p>The classic example is indexing only unshipped orders: the index stays tiny
because shipped orders fall out of it automatically. Queries that repeat the
predicate in their WHERE clause can use the index; queries that do not repeat
it cannot, which is the most common surprise for newcomers.</p>
<p>Partial indexes also combine well with UNIQUE, letting you enforce
uniqueness over a subset of rows, such as at most one active subscription per
customer. That trick is hard to express any other way and is worth remembering
when modelling soft-deleted records.</p>
</article>
<footer>Copyright 2026</footer>
</body></html>"""


def test_youtube_video_id_matches_watch_and_short_urls():
    assert fetch.youtube_video_id(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    ) == "dQw4w9WgXcQ"
    assert fetch.youtube_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert fetch.youtube_video_id(
        "https://m.youtube.com/watch?feature=share&v=dQw4w9WgXcQ"
    ) == "dQw4w9WgXcQ"


def test_youtube_video_id_rejects_non_youtube():
    assert fetch.youtube_video_id("https://example.com/watch?v=abcdef1") is None
    assert fetch.youtube_video_id("https://vimeo.com/12345") is None


def test_extract_article_returns_title_and_text():
    result = fetch.extract_article(ARTICLE_HTML)
    assert result is not None
    title, text = result
    assert "Partial Indexes" in title
    assert "unshipped orders" in text
    # Boilerplate must not leak into the extracted text.
    assert "Copyright" not in text


def test_extract_article_rejects_empty_html():
    assert fetch.extract_article("<html><body></body></html>") is None


def test_fetch_url_rejects_non_http_schemes():
    assert fetch.fetch_url("file:///etc/passwd") is None
    assert fetch.fetch_url("notaurl") is None


def test_fetch_url_article_path(monkeypatch):
    monkeypatch.setattr(
        fetch, "_http_get", lambda url, timeout: ARTICLE_HTML.encode("utf-8")
    )
    doc = fetch.fetch_url("https://example.com/post")
    assert doc is not None
    assert doc.url == "https://example.com/post"
    assert "unshipped orders" in doc.text
    assert doc.fetched_at > 0


def test_fetch_url_returns_none_on_network_failure(monkeypatch):
    monkeypatch.setattr(fetch, "_http_get", lambda url, timeout: None)
    assert fetch.fetch_url("https://example.com/post") is None


def test_fetch_url_youtube_path(monkeypatch):
    monkeypatch.setattr(fetch, "_fetch_transcript", lambda vid: "hello transcript")
    monkeypatch.setattr(fetch, "_youtube_title", lambda url, timeout: "Video Title")
    doc = fetch.fetch_url("https://youtu.be/dQw4w9WgXcQ")
    assert doc is not None
    assert doc.title == "Video Title"
    assert doc.text == "hello transcript"


def test_fetch_url_youtube_no_transcript(monkeypatch):
    monkeypatch.setattr(fetch, "_fetch_transcript", lambda vid: None)
    assert fetch.fetch_url("https://youtu.be/dQw4w9WgXcQ") is None
```

- [ ] **Step 3: 跑测试确认失败**

Run: `python -m pytest tests/test_fetch.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'kb.fetch'`(collection error)。

- [ ] **Step 4: 实现 `kb/fetch.py`**

```python
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
        text = trafilatura.extract(html, url=url or None, include_comments=False)
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
```

- [ ] **Step 5: 跑测试确认通过**

Run: `python -m pytest tests/test_fetch.py -v`
Expected: 全部 PASS。若 `test_extract_article_returns_title_and_text` 失败(trafilatura 认为 fixture 不够长),把 fixture 的三段 `<p>` 各再加一两句英文散文后重跑——不许改断言。

- [ ] **Step 6: 跑全量测试防回归,然后提交**

Run: `python -m pytest tests/ -q`
Expected: 全部 PASS(已有测试不受影响)。

```bash
git add kb/fetch.py tests/test_fetch.py pyproject.toml
git commit -m "feat: content fetcher (web article + YouTube transcript) behind [clip] extra"
```

---

### Task 2: config — `clips_dir` 解析 + `"inbox"` source kind

**Files:**
- Modify: `kb/config.py`
- Create: `tests/test_capture_config.py`

**Interfaces:**
- Consumes: 无新依赖。
- Produces:
  - `config.clips_dir() -> Optional[str]`(解析顺序 `$KB_CLIPS_DIR` → `config.json` 的 `clips_dir` 键 → `None`;纯查询,无副作用)
  - `config.set_clips_dir(path: str) -> None`(写 `<kb_home>/config.json`,保留其他键)
  - `config.add_source("inbox", path)` 合法(`_VALID_KINDS` 增加 `"inbox"`)

- [ ] **Step 1: 写失败测试**

创建 `tests/test_capture_config.py`:

```python
"""Tests for capture-layer config: clips_dir resolution and the inbox kind."""

from __future__ import annotations

import json

from kb import config


def test_clips_dir_unconfigured_is_none(monkeypatch, tmp_path):
    monkeypatch.setenv("KB_HOME", str(tmp_path))
    monkeypatch.delenv("KB_CLIPS_DIR", raising=False)
    assert config.clips_dir() is None


def test_clips_dir_env_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("KB_HOME", str(tmp_path))
    monkeypatch.setenv("KB_CLIPS_DIR", "D:/vault/Clips")
    assert config.clips_dir() == "D:/vault/Clips"


def test_set_clips_dir_persists_and_resolves(monkeypatch, tmp_path):
    monkeypatch.setenv("KB_HOME", str(tmp_path))
    monkeypatch.delenv("KB_CLIPS_DIR", raising=False)
    config.set_clips_dir(str(tmp_path / "Clips"))
    assert config.clips_dir() == str(tmp_path / "Clips")


def test_set_clips_dir_preserves_other_config_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("KB_HOME", str(tmp_path))
    monkeypatch.delenv("KB_MODEL", raising=False)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"model": "claude-sonnet-5"}), encoding="utf-8")
    config.set_clips_dir(str(tmp_path / "Clips"))
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data["model"] == "claude-sonnet-5"
    assert data["clips_dir"] == str(tmp_path / "Clips")
    # synthesis_model still resolves from the same file.
    assert config.synthesis_model() == "claude-sonnet-5"


def test_set_clips_dir_recovers_from_corrupt_config(monkeypatch, tmp_path):
    monkeypatch.setenv("KB_HOME", str(tmp_path))
    (tmp_path / "config.json").write_text("not json{", encoding="utf-8")
    config.set_clips_dir(str(tmp_path / "Clips"))
    assert config.clips_dir() == str(tmp_path / "Clips")


def test_add_inbox_source(monkeypatch, tmp_path):
    monkeypatch.setenv("KB_HOME", str(tmp_path))
    sources = config.add_source("inbox", str(tmp_path / "Inbox"))
    assert sources == [{"kind": "inbox", "path": str(tmp_path / "Inbox")}]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_capture_config.py -v`
Expected: FAIL,`AttributeError: module 'kb.config' has no attribute 'clips_dir'`;inbox 用例 `ValueError: invalid source kind 'inbox'`。

- [ ] **Step 3: 实现**

`kb/config.py` 三处修改。

(a) `_VALID_KINDS` 一行替换:

```python
# Allowed values for a registered source's "kind" field.
_VALID_KINDS = {"files", "bookmarks", "inbox"}
```

(b) 常量区(`_API_KEY_ENV` 下方)加:

```python
# Environment variable overriding the configured clips folder.
_CLIPS_DIR_ENV = "KB_CLIPS_DIR"
```

(c) 在 `api_key_source()` 之后加两个函数:

```python
def clips_dir() -> "Optional[str]":
    """Return the configured clips folder, or ``None`` when unconfigured.

    Resolution order (pure, never side-effecting — same shape as
    :func:`synthesis_model`):

    1. ``$KB_CLIPS_DIR`` if set and non-empty.
    2. ``"clips_dir"`` key in ``<kb_home>/config.json``.
    3. ``None`` — callers surface a "run kb clip --set-dir" hint.
    """
    env_val = os.environ.get(_CLIPS_DIR_ENV)
    if env_val:
        return env_val

    cfg_path = kb_home() / "config.json"
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        val = data.get("clips_dir") if isinstance(data, dict) else None
        if isinstance(val, str) and val:
            return val
    except Exception:
        pass
    return None


def set_clips_dir(path: str) -> None:
    """Persist *path* as ``clips_dir`` in ``<kb_home>/config.json``.

    Merges into the existing config file so other keys (e.g. ``model``)
    survive; an unreadable or corrupt file is replaced rather than fatal.
    ``kb_home`` is created on write, mirroring :func:`_write_sources`.
    """
    home = kb_home()
    home.mkdir(parents=True, exist_ok=True)
    cfg_path = home / "config.json"
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    data["clips_dir"] = str(Path(path))
    cfg_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
```

并在文件顶部 import 区加 `from typing import Optional`。

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_capture_config.py tests/test_zeroconfig.py -v`
Expected: 全部 PASS(zeroconfig 是 config 的既有测试,必须不回归)。

- [ ] **Step 5: 提交**

```bash
git add kb/config.py tests/test_capture_config.py
git commit -m "feat: clips_dir config resolution and inbox source kind"
```

---

### Task 3: `kb/clip.py` — 笔记渲染与落盘

**Files:**
- Create: `kb/clip.py`
- Create: `tests/test_clip.py`

**Interfaces:**
- Consumes: `kb.fetch.FetchedDoc`(Task 1)。
- Produces(Task 4/5 依赖):
  - `slug(title: str) -> str`
  - `render_note(doc: FetchedDoc, clipped_date: str) -> str`(Task 5 的 inbox 展开复用同一渲染)
  - `find_existing(clips_dir: Path, url: str) -> Optional[Path]`
  - `save_clip(doc: FetchedDoc, clips_dir: str) -> Optional[Path]`(`None` 表示该 URL 已存在,跳过)

- [ ] **Step 1: 写失败测试**

创建 `tests/test_clip.py`:

```python
"""Tests for kb.clip — note rendering and on-disk saving."""

from __future__ import annotations

from pathlib import Path

from kb.clip import find_existing, render_note, save_clip, slug
from kb.fetch import FetchedDoc

# 2026-07-11 12:00:00 UTC — any fixed timestamp works; the date assertion
# below derives the expected string the same way the implementation does.
_TS = 1783166400.0


def _doc(title="Partial Indexes", url="https://example.com/post", text="Body text."):
    return FetchedDoc(title=title, text=text, url=url, fetched_at=_TS)


def test_slug_ascii():
    assert slug("Postgres Partial Indexes!") == "Postgres-Partial-Indexes"


def test_slug_keeps_cjk():
    assert slug("局部索引详解") == "局部索引详解"


def test_slug_truncates_to_60_chars():
    assert len(slug("x" * 200)) <= 60


def test_slug_empty_falls_back():
    assert slug("///???") == "clip"


def test_render_note_frontmatter():
    text = render_note(_doc(), "2026-07-11")
    lines = text.splitlines()
    assert lines[0] == "---"
    assert 'title: "Partial Indexes"' in lines
    assert "url: https://example.com/post" in lines
    assert "clipped: 2026-07-11" in lines
    assert "kb-clipped: true" in lines
    assert "Body text." in text


def test_render_note_escapes_double_quotes_in_title():
    text = render_note(_doc(title='He said "hi"'), "2026-07-11")
    assert "title: \"He said 'hi'\"" in text


def test_save_clip_writes_file(tmp_path):
    path = save_clip(_doc(), str(tmp_path))
    assert path is not None and path.exists()
    content = path.read_text(encoding="utf-8")
    assert "kb-clipped: true" in content
    assert path.name == "Partial-Indexes.md"


def test_save_clip_dedupes_by_url(tmp_path):
    assert save_clip(_doc(), str(tmp_path)) is not None
    assert save_clip(_doc(title="Different Title"), str(tmp_path)) is None
    assert len(list(tmp_path.glob("*.md"))) == 1


def test_save_clip_text_mode_never_dedupes(tmp_path):
    # Pasted-text clips have url="" and must not collide with each other.
    assert save_clip(_doc(title="Note A", url=""), str(tmp_path)) is not None
    assert save_clip(_doc(title="Note B", url=""), str(tmp_path)) is not None
    assert len(list(tmp_path.glob("*.md"))) == 2


def test_save_clip_filename_collision_appends_counter(tmp_path):
    assert save_clip(_doc(url="https://example.com/a"), str(tmp_path)) is not None
    p2 = save_clip(_doc(url="https://example.com/b"), str(tmp_path))
    assert p2 is not None and p2.name == "Partial-Indexes-2.md"


def test_find_existing_matches_frontmatter_url(tmp_path):
    save_clip(_doc(), str(tmp_path))
    assert find_existing(Path(tmp_path), "https://example.com/post") is not None
    assert find_existing(Path(tmp_path), "https://example.com/other") is None


def test_save_clip_creates_missing_folder(tmp_path):
    target = tmp_path / "nested" / "Clips"
    assert save_clip(_doc(), str(target)) is not None
    assert target.is_dir()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_clip.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'kb.clip'`。

- [ ] **Step 3: 实现 `kb/clip.py`**

```python
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_clip.py -v`
Expected: 全部 PASS。

- [ ] **Step 5: 提交**

```bash
git add kb/clip.py tests/test_clip.py
git commit -m "feat: clip note rendering and saving with URL dedup"
```

---

### Task 4: CLI `kb clip` 子命令

**Files:**
- Modify: `kb/__main__.py`(`_build_parser` 加子命令;新增 `_run_clip`;`main()` 加分派)
- Create: `tests/test_clip_cli.py`

**Interfaces:**
- Consumes: `config.clips_dir()` / `config.set_clips_dir()`(Task 2)、`clip.save_clip`(Task 3)、`fetch.fetch_url` / `fetch.FetchedDoc`(Task 1)、既有 `config.add_source`、`ingest`、`_resolve_index_dir`。
- Produces: 子命令 `kb clip [url] [--text TITLE] [--set-dir PATH] [--index-dir DIR]`。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_clip_cli.py`:

```python
"""CLI-level tests for `kb clip` — fetch and ingest are always monkeypatched."""

from __future__ import annotations

import io
import sys

import pytest

import kb.__main__ as cli
from kb import config
from kb.fetch import FetchedDoc

_TS = 1783166400.0


def _run(argv, monkeypatch):
    """Invoke the real CLI entrypoint; return its SystemExit code."""
    monkeypatch.setattr(sys, "argv", ["kb"] + argv)
    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    return exc_info.value.code


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("KB_HOME", str(tmp_path))
    monkeypatch.delenv("KB_CLIPS_DIR", raising=False)
    return tmp_path


def test_clip_set_dir(home, monkeypatch):
    target = home / "Clips"
    assert _run(["clip", "--set-dir", str(target)], monkeypatch) == 0
    assert config.clips_dir() == str(target)


def test_clip_without_configured_dir_errors(home, monkeypatch):
    assert _run(["clip", "https://example.com/post"], monkeypatch) == 1


def test_clip_url_saves_registers_and_ingests(home, monkeypatch):
    clips = home / "Clips"
    config.set_clips_dir(str(clips))
    doc = FetchedDoc(
        title="Post", text="Body.", url="https://example.com/post", fetched_at=_TS
    )
    monkeypatch.setattr(cli, "fetch_url", lambda url, timeout=15.0: doc)
    ingested = []
    monkeypatch.setattr(cli, "ingest", lambda d, **kw: ingested.append(d))

    assert _run(["clip", "https://example.com/post"], monkeypatch) == 0
    assert list(clips.glob("*.md")), "clip note not written"
    assert ingested == [str(clips)]
    assert {"kind": "files", "path": str(clips)} in config.load_sources()


def test_clip_fetch_failure_exits_1(home, monkeypatch):
    config.set_clips_dir(str(home / "Clips"))
    monkeypatch.setattr(cli, "fetch_url", lambda url, timeout=15.0: None)
    assert _run(["clip", "https://example.com/post"], monkeypatch) == 1


def test_clip_duplicate_url_is_ok_but_skips(home, monkeypatch):
    clips = home / "Clips"
    config.set_clips_dir(str(clips))
    doc = FetchedDoc(
        title="Post", text="Body.", url="https://example.com/post", fetched_at=_TS
    )
    monkeypatch.setattr(cli, "fetch_url", lambda url, timeout=15.0: doc)
    monkeypatch.setattr(cli, "ingest", lambda d, **kw: None)
    assert _run(["clip", "https://example.com/post"], monkeypatch) == 0
    assert _run(["clip", "https://example.com/post"], monkeypatch) == 0
    assert len(list(clips.glob("*.md"))) == 1


def test_clip_text_mode_reads_stdin(home, monkeypatch):
    clips = home / "Clips"
    config.set_clips_dir(str(clips))
    monkeypatch.setattr(cli, "ingest", lambda d, **kw: None)
    monkeypatch.setattr(sys, "stdin", io.StringIO("A conclusion from claude.ai"))
    assert _run(["clip", "--text", "Agent memory notes"], monkeypatch) == 0
    files = list(clips.glob("*.md"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert "A conclusion from claude.ai" in content
    assert 'title: "Agent memory notes"' in content


def test_clip_no_url_no_text_errors(home, monkeypatch):
    config.set_clips_dir(str(home / "Clips"))
    assert _run(["clip"], monkeypatch) == 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_clip_cli.py -v`
Expected: FAIL,argparse 报 `invalid choice: 'clip'`(SystemExit 2)。

- [ ] **Step 3: 实现 CLI**

`kb/__main__.py` 四处修改。

(a) 顶部 import 区加(与既有顶层 import 并列;`fetch` 只含 stdlib,顶层 import 不违反红线):

```python
from .fetch import FetchedDoc, fetch_url
```

(b) `_build_parser` 中,`sub.add_parser("sources", ...)` 之前插入:

```python
    clip_p = sub.add_parser(
        "clip",
        help="Capture a web page / YouTube URL (or stdin text) into the clips folder.",
    )
    clip_p.add_argument(
        "url",
        nargs="?",
        default=None,
        help="URL to fetch. Requires the [clip] extra: pip install -e '.[clip]'.",
    )
    clip_p.add_argument(
        "--text",
        default=None,
        metavar="TITLE",
        help="Read a text clip from stdin and save it under TITLE.",
    )
    clip_p.add_argument(
        "--set-dir",
        default=None,
        metavar="PATH",
        help="Set and persist the clips folder, then exit.",
    )
    clip_p.add_argument(
        "--index-dir",
        default=None,
        help="Override the managed index location (advanced).",
    )
```

(c) `_run_sources` 之前加:

```python
def _run_clip(args: argparse.Namespace) -> int:
    if args.set_dir:
        config.set_clips_dir(args.set_dir)
        print(f"Clips folder set to: {args.set_dir}")
        return 0

    clips = config.clips_dir()
    if not clips:
        print(
            "No clips folder configured. "
            "Run `kb clip --set-dir <folder>` first (e.g. a Clips/ folder "
            "inside your Obsidian vault).",
            file=sys.stderr,
        )
        return 1

    import time as _time

    if args.text is not None:
        body = sys.stdin.read()
        if not body.strip():
            print("No text on stdin to clip.", file=sys.stderr)
            return 1
        doc = FetchedDoc(title=args.text, text=body, url="", fetched_at=_time.time())
    elif args.url:
        doc = fetch_url(args.url)
        if doc is None:
            print(
                "Could not fetch that URL. Check the address and your network, "
                "and that the capture extra is installed: "
                "pip install -e \".[clip]\"",
                file=sys.stderr,
            )
            return 1
    else:
        print("Provide a URL, or --text TITLE with content on stdin.", file=sys.stderr)
        return 1

    from .clip import save_clip

    path = save_clip(doc, clips)
    if path is None:
        print(f"Already clipped: {doc.url}")
        return 0

    # First use registers the clips folder so query/watch cover it from now on.
    config.add_source("files", clips)
    ingest(clips, index_dir=_resolve_index_dir(args.index_dir))
    print(f"Clipped: {path}")
    return 0
```

(d) `main()` 的分派链中 `elif args.command == "sources":` 之前加:

```python
    elif args.command == "clip":
        sys.exit(_run_clip(args))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_clip_cli.py -v`
Expected: 全部 PASS。

- [ ] **Step 5: 全量回归 + 提交**

Run: `python -m pytest tests/ -q`
Expected: 全部 PASS。

```bash
git add kb/__main__.py tests/test_clip_cli.py
git commit -m "feat: kb clip command (URL and stdin-text capture)"
```

---

### Task 5: `kb/inbox.py` — 裸 URL 笔记识别与原地展开

**Files:**
- Create: `kb/inbox.py`
- Create: `tests/test_inbox.py`

**Interfaces:**
- Consumes: `fetch.fetch_url`(默认 fetcher)、`clip.render_note`、`config.kb_home()`。
- Produces(Task 6 依赖):
  - `parse_bare_url_note(text: str) -> Optional[str]`
  - `process_inbox(folder: str, timeout: float = 15.0, fetcher: Optional[Callable] = None) -> int`(返回展开条数;`fetcher(url, timeout=...)` 返回 `Optional[FetchedDoc]`)

- [ ] **Step 1: 写失败测试**

创建 `tests/test_inbox.py`:

```python
"""Tests for kb.inbox — bare-URL detection and in-place expansion."""

from __future__ import annotations

import json

import pytest

from kb.fetch import FetchedDoc
from kb.inbox import parse_bare_url_note, process_inbox

_TS = 1783166400.0
_URL = "https://example.com/post"


# --- parse_bare_url_note: table-driven -------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        (f"{_URL}\n", _URL),                                # bare URL
        (f"\n\n  {_URL}  \n\n", _URL),                      # surrounding blanks
        (f"[An article]({_URL})", _URL),                    # markdown link only
        (f"---\ntags: [reading]\n---\n\n{_URL}\n", _URL),   # harmless frontmatter
        (f"Check this out:\n{_URL}\n", None),               # prose + URL
        (f"{_URL}\nhttps://example.com/two\n", None),       # two URLs
        ("Just a normal note about postgres.", None),       # no URL at all
        ("", None),                                         # empty file
        ("---\ntags: [x]\n---\n", None),                    # frontmatter only
        (f"---\nkb-clipped: true\n---\n\n{_URL}\n", None),  # already expanded
        (f"---\nkb-clip-failed: true\n---\n\n{_URL}\n", None),  # given up
        ("ftp://example.com/file", None),                   # non-http scheme
    ],
)
def test_parse_bare_url_note(text, expected):
    assert parse_bare_url_note(text) == expected


# --- process_inbox ----------------------------------------------------------

@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("KB_HOME", str(tmp_path / "kbhome"))
    inbox = tmp_path / "Inbox"
    inbox.mkdir()
    return inbox


def _ok_fetcher(url, timeout=15.0):
    return FetchedDoc(title="Fetched Title", text="Full body.", url=url, fetched_at=_TS)


def _fail_fetcher(url, timeout=15.0):
    return None


def test_expands_bare_url_note_in_place(home):
    note = home / "share.md"
    note.write_text(_URL + "\n", encoding="utf-8")
    assert process_inbox(str(home), fetcher=_ok_fetcher) == 1
    content = note.read_text(encoding="utf-8")
    assert "kb-clipped: true" in content
    assert f"url: {_URL}" in content
    assert "Full body." in content
    # Second pass is a no-op: the marker suppresses reprocessing.
    assert process_inbox(str(home), fetcher=_ok_fetcher) == 0


def test_never_touches_notes_with_prose(home):
    note = home / "real-note.md"
    original = "My own thoughts.\n" + _URL + "\n"
    note.write_text(original, encoding="utf-8")
    assert process_inbox(str(home), fetcher=_ok_fetcher) == 0
    assert note.read_text(encoding="utf-8") == original


def test_failure_retries_then_gives_up(home, monkeypatch, tmp_path):
    note = home / "share.md"
    note.write_text(_URL + "\n", encoding="utf-8")

    # Failures 1 and 2: note untouched, fail count persisted.
    assert process_inbox(str(home), fetcher=_fail_fetcher) == 0
    assert process_inbox(str(home), fetcher=_fail_fetcher) == 0
    assert note.read_text(encoding="utf-8") == _URL + "\n"
    state_file = tmp_path / "kbhome" / "inbox_state.json"
    assert json.loads(state_file.read_text(encoding="utf-8"))[str(note)] == 2

    # Failure 3: note marked failed, state entry cleared, no more retries.
    assert process_inbox(str(home), fetcher=_fail_fetcher) == 0
    content = note.read_text(encoding="utf-8")
    assert "kb-clip-failed: true" in content
    assert _URL in content  # the URL itself must survive
    assert str(note) not in json.loads(state_file.read_text(encoding="utf-8"))
    assert parse_bare_url_note(content) is None  # marker suppresses retry


def test_success_clears_fail_count(home, tmp_path):
    note = home / "share.md"
    note.write_text(_URL + "\n", encoding="utf-8")
    process_inbox(str(home), fetcher=_fail_fetcher)
    assert process_inbox(str(home), fetcher=_ok_fetcher) == 1
    state = json.loads(
        (tmp_path / "kbhome" / "inbox_state.json").read_text(encoding="utf-8")
    )
    assert str(note) not in state


def test_missing_folder_returns_zero(home):
    assert process_inbox(str(home / "nope"), fetcher=_ok_fetcher) == 0


def test_processes_nested_subfolders(home):
    sub = home / "2026" / "07"
    sub.mkdir(parents=True)
    (sub / "share.md").write_text(_URL + "\n", encoding="utf-8")
    assert process_inbox(str(home), fetcher=_ok_fetcher) == 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_inbox.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'kb.inbox'`。

- [ ] **Step 3: 实现 `kb/inbox.py`**

```python
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_inbox.py -v`
Expected: 全部 PASS。

- [ ] **Step 5: 提交**

```bash
git add kb/inbox.py tests/test_inbox.py
git commit -m "feat: inbox processor — expand bare-URL notes in place with retry cap"
```

---

### Task 6: `kb add --inbox` + watch 接入

**Files:**
- Modify: `kb/__main__.py`(add 子命令加 `--inbox`;`_run_add` 加分支)
- Modify: `kb/watch.py`(`_files_folders` 纳入 inbox;新增 `_inbox_folders`;`run_once` 加 inbox 步)
- Create: `tests/test_watch_inbox.py`

**Interfaces:**
- Consumes: `config.add_source("inbox", path)`(Task 2)、`inbox.process_inbox`(Task 5)。
- Produces: `kb add --inbox <path>`;`watch.run_once` 在快照/ingest 前自动展开 inbox 笔记。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_watch_inbox.py`:

```python
"""Watch-cycle integration: inbox expansion feeds the normal reingest path."""

from __future__ import annotations

import sys

import pytest

import kb.__main__ as cli
import kb.watch as watch
from kb import config
from kb.fetch import FetchedDoc

_TS = 1783166400.0
_URL = "https://example.com/post"


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("KB_HOME", str(tmp_path / "kbhome"))
    return tmp_path


def test_add_inbox_registers_and_ingests(home, monkeypatch):
    inbox = home / "Inbox"
    inbox.mkdir()
    ingested = []
    monkeypatch.setattr(cli, "ingest", lambda d, **kw: ingested.append(d))
    monkeypatch.setattr(sys, "argv", ["kb", "add", "--inbox", str(inbox)])
    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    assert exc_info.value.code == 0
    assert {"kind": "inbox", "path": str(inbox)} in config.load_sources()
    assert ingested == [str(inbox)]


def test_run_once_expands_inbox_then_reingests(home, monkeypatch):
    inbox = home / "Inbox"
    inbox.mkdir()
    (inbox / "share.md").write_text(_URL + "\n", encoding="utf-8")
    config.add_source("inbox", str(inbox))

    monkeypatch.setattr(
        "kb.fetch.fetch_url",
        lambda url, timeout=15.0: FetchedDoc(
            title="T", text="Body.", url=url, fetched_at=_TS
        ),
    )
    ingested = []
    monkeypatch.setattr(watch, "ingest_fn", lambda d, **kw: ingested.append(d))

    changed, state = watch.run_once(index_dir=str(home / "idx"), state={})
    assert changed is True
    assert ingested == [str(inbox)]  # inbox folders are watched as files too
    content = (inbox / "share.md").read_text(encoding="utf-8")
    assert "kb-clipped: true" in content


def test_run_once_inbox_failure_does_not_break_cycle(home, monkeypatch):
    inbox = home / "Inbox"
    inbox.mkdir()
    (inbox / "share.md").write_text(_URL + "\n", encoding="utf-8")
    config.add_source("inbox", str(inbox))

    def _boom(folder, timeout=15.0, fetcher=None):
        raise RuntimeError("inbox exploded")

    monkeypatch.setattr(watch, "process_inbox", _boom)
    monkeypatch.setattr(watch, "ingest_fn", lambda d, **kw: None)
    # Must not raise: the inbox step is isolated from the watch cycle.
    changed, state = watch.run_once(index_dir=str(home / "idx"), state={})
    assert changed is True  # the bare-URL note itself still counts as a file
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_watch_inbox.py -v`
Expected: FAIL——`add --inbox` 报 argparse 错误(exit 2);watch 用例因 `AttributeError: ... no attribute 'process_inbox'` 失败。

- [ ] **Step 3: 实现 `kb add --inbox`**

`kb/__main__.py` 两处。

(a) `_build_parser` 中 `add_p.add_argument("--bookmarks", ...)` 之后加:

```python
    add_p.add_argument(
        "--inbox",
        action="store_true",
        help="Register <path> as an inbox folder: notes containing only a "
        "URL are auto-expanded into full articles by `kb watch`.",
    )
```

(b) `_run_add` 整体替换为:

```python
def _run_add(args: argparse.Namespace) -> int:
    index_dir = _resolve_index_dir(args.index_dir)
    if args.bookmarks and args.inbox:
        print("--bookmarks and --inbox are mutually exclusive.", file=sys.stderr)
        return 1
    if args.bookmarks:
        config.add_source("bookmarks", args.path)
        from .ingest import _ingest_from_source
        from .source import BookmarkSource

        _ingest_from_source(
            BookmarkSource(args.path), index_dir=index_dir, label=args.path
        )
        print(f"Added bookmarks source: {args.path}")
    elif args.inbox:
        config.add_source("inbox", args.path)
        ingest(args.path, index_dir=index_dir)
        print(f"Added inbox source: {args.path}")
    else:
        config.add_source("files", args.path)
        ingest(args.path, index_dir=index_dir)
        print(f"Added files source: {args.path}")
    return 0
```

- [ ] **Step 4: 实现 watch 接入**

`kb/watch.py` 三处。

(a) import 区 `from .source import FileSource` 之后加:

```python
from .inbox import process_inbox
```

(b) `_files_folders` 整体替换(inbox 文件夹同时参与文件监视/索引),并在其后新增 `_inbox_folders`:

```python
def _files_folders() -> list[str]:
    """Normalized paths of every source the file watcher should index.

    Inbox folders are watched exactly like ``"files"`` sources — expansion
    rewrites notes on disk, and those rewrites must trigger a reingest.
    Bookmark sources remain ignored by the watcher at this stage.
    """
    return [
        entry["path"]
        for entry in config.load_sources()
        if entry.get("kind") in ("files", "inbox") and entry.get("path")
    ]


def _inbox_folders() -> list[str]:
    """Normalized paths of every registered ``"inbox"`` source."""
    return [
        entry["path"]
        for entry in config.load_sources()
        if entry.get("kind") == "inbox" and entry.get("path")
    ]
```

(c) `run_once` 中,`folders = _files_folders()` 之前插入 inbox 步:

```python
    # Inbox step first, so this cycle's snapshot sees the expanded notes and
    # the normal diff-and-reingest path picks them up immediately. Isolated:
    # an inbox failure (offline, bad folder) must never break the cycle.
    for inbox_folder in _inbox_folders():
        try:
            n = process_inbox(inbox_folder)
            if n:
                print(f"kb watch: expanded {n} inbox note(s) in {inbox_folder}")
        except Exception as exc:
            print(
                f"kb watch: inbox step failed for {inbox_folder}: {exc}",
                file=sys.stderr,
            )
```

同时更新 `run_once` docstring 的 cycle 列表:在第 2 步前加一条 "1.5. Expand bare-URL notes in every registered inbox folder (isolated per folder)."(措辞可顺手融入,保持编号连贯即可)。

- [ ] **Step 5: 跑测试确认通过**

Run: `python -m pytest tests/test_watch_inbox.py -v`
Expected: 全部 PASS。

- [ ] **Step 6: 全量回归 + 提交**

Run: `python -m pytest tests/ -q`
Expected: 全部 PASS。

```bash
git add kb/__main__.py kb/watch.py tests/test_watch_inbox.py
git commit -m "feat: inbox source kind wired into add and watch"
```

---

### Task 7: README 文档

**Files:**
- Modify: `README.md`

**Interfaces:** 无代码;文档必须与 Task 1–6 的实际行为一致。

- [ ] **Step 1: 更新 README**

(a) Install 一节的 extras 列表加一行(在 `[synthesis]` 行后):

```markdown
pip install -e ".[clip]"        # enable `kb clip` and inbox capture (web/YouTube)
```

(b) Commands 表格加两行(`add <path>` 行之后):

```markdown
| `clip <url>`       | Capture a web page or YouTube transcript into your clips folder and index it. `--text TITLE` clips stdin; `--set-dir` configures the folder. |
| `add --inbox <path>` | Register an inbox folder: notes containing only a URL are auto-expanded into full articles by `kb watch`. |
```

(c) Daily workflow 一节的 "**4. Keep the index fresh.**" 之后新增一节:

```markdown
**5. Capture from anywhere.**

On the PC, clip any article or YouTube video straight into your knowledge
base (one-time setup: point `kb clip` at a folder, e.g. a `Clips/` folder
inside your Obsidian vault):

```sh
kb clip --set-dir "D:/vault/Clips"       # once
kb clip https://example.com/great-post   # article -> markdown note, indexed
kb clip --text "Agent memory notes"      # reads the note body from stdin
```

From your phone, share a URL into an `Inbox/` folder of your synced vault
and let `kb watch` do the rest — it detects notes that contain only a URL,
fetches the full article, and expands the note in place (the expanded copy
syncs back to your phone as a read-later version):

```sh
kb add --inbox "D:/vault/Inbox"   # once
kb watch                          # expands inbox notes + reindexes on change
```

Capture needs the `[clip]` extra: `pip install -e ".[clip]"`.
```

- [ ] **Step 2: 提交**

```bash
git add README.md
git commit -m "docs: document kb clip and inbox capture workflow"
```
