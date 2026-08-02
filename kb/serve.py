"""Local web UI for kb: search, ask, and clip over plain HTTP.

The counterpart to the CLI. One stdlib :class:`~http.server.ThreadingHTTPServer`
serves a single self-contained page plus a small JSON API, so the knowledge base
is reachable from a browser -- including a phone on the same wifi, which is the
one thing the file-drop inbox could never give you.

Project red line -- *pure Python, no native-compiled dependencies*. No Flask, no
FastAPI, no bundler: ``http.server`` and one HTML file with its CSS and JS
inlined. The page fetches nothing from the network, so it works with the
machine offline (right up until you press Ask).

Routes
------
``GET  /``            the page itself.
``GET  /api/search``  ``?q=`` plus the usual ``k`` / ``kind`` / ``since`` /
                      ``hybrid`` options; pure-local retrieval.
``POST /api/ask``     ``{"question": ...}``; the only route that leaves the machine.
``POST /api/clip``    ``{"url": ...}``; fetch, save into the clips folder, reindex.

Exposure
--------
Binding is ``127.0.0.1`` by default: nothing outside this machine can reach it.
Serving on a LAN address is an explicit choice (``--host 0.0.0.0``), and because
that hands anyone on the network your whole knowledge base -- and your API
credits via ``/api/ask`` -- :func:`serve` mints a token for non-loopback binds
and rejects requests that do not carry it. That is a speed bump for a home
network, not authentication: do not put this on the open internet.
"""

from __future__ import annotations

import json
import secrets
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import config
from .answer import answer
from .clip import save_clip
from .fetch import fetch_url
from .ingest import ingest
from .llm import KbLLMError
from .query import IncompatibleIndexError, query

DEFAULT_PORT = 7777

# The page lives beside this module so it can be edited as HTML rather than as
# a Python string. Re-read per request: a refresh picks up edits, and it costs
# nothing at this size.
_PAGE_PATH = Path(__file__).resolve().parent / "web" / "index.html"

# Cap on a single POST body. Nothing we accept is large, and an unbounded read
# would let one request exhaust memory.
_MAX_BODY_BYTES = 64 * 1024

_TRUTHY = {"1", "true", "yes", "on"}


def _as_bool(value: str | None) -> bool:
    return (value or "").strip().lower() in _TRUTHY


def _first(params: dict, name: str, default=None):
    """First value for *name* in a parse_qs mapping, or *default*."""
    values = params.get(name)
    return values[0] if values else default


def _build_handler(index_dir: str, token: str | None):
    """Return a request-handler class bound to *index_dir* and *token*."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "kb"
        sys_version = ""

        # -- plumbing ---------------------------------------------------

        def log_message(self, fmt, *args):  # noqa: A003 - stdlib hook name
            """One terse line per request instead of the stdlib's noisy default."""
            sys.stderr.write(f"kb serve: {self.command} {self.path} {fmt % args}\n")

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            # The page is self-contained; forbid outbound requests outright so a
            # future edit cannot quietly introduce a CDN dependency.
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; connect-src 'self'; style-src 'unsafe-inline'; "
                "script-src 'unsafe-inline'; form-action 'none'; base-uri 'none'",
            )
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"error": message})

        def _authorized(self, params: dict) -> bool:
            """True when no token is configured, or the request carries it."""
            if not token:
                return True
            supplied = _first(params, "t") or self.headers.get("X-KB-Token") or ""
            return secrets.compare_digest(supplied, token)

        def _read_json_body(self) -> dict | None:
            """Parse the request body as a JSON object; ``None`` if unusable."""
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return None
            if length <= 0 or length > _MAX_BODY_BYTES:
                return None
            try:
                data = json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception:
                return None
            return data if isinstance(data, dict) else None

        # -- retrieval options shared by search and ask -----------------

        @staticmethod
        def _retrieval_options(src: dict, getter) -> dict:
            """Pull k/kind/since/hybrid out of a query string or JSON body."""
            try:
                k = int(getter(src, "k") or 5)
            except (TypeError, ValueError):
                k = 5
            return {
                "k": max(1, min(k, 50)),
                "kind": getter(src, "kind") or None,
                "since": getter(src, "since") or None,
                "hybrid": _as_bool(str(getter(src, "hybrid") or "")),
            }

        # -- routes -----------------------------------------------------

        def do_GET(self):  # noqa: N802 - stdlib hook name
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)

            if not self._authorized(params):
                return self._error(401, "Missing or invalid token.")

            if parsed.path in ("/", "/index.html"):
                return self._serve_page()
            if parsed.path == "/api/search":
                return self._search(params)
            return self._error(404, f"No such path: {parsed.path}")

        def do_POST(self):  # noqa: N802 - stdlib hook name
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)

            if not self._authorized(params):
                return self._error(401, "Missing or invalid token.")

            body = self._read_json_body()
            if body is None:
                return self._error(400, "Expected a JSON object body.")

            if parsed.path == "/api/ask":
                return self._ask(body)
            if parsed.path == "/api/clip":
                return self._clip(body)
            return self._error(404, f"No such path: {parsed.path}")

        def _serve_page(self) -> None:
            try:
                html = _PAGE_PATH.read_bytes()
            except OSError:
                return self._error(500, "The web page asset is missing.")
            self._send(200, html, "text/html; charset=utf-8")

        def _search(self, params: dict) -> None:
            question = (_first(params, "q") or "").strip()
            if not question:
                return self._json(200, {"results": []})
            options = self._retrieval_options(params, _first)
            try:
                results = query(question, index_dir=index_dir, **options)
            except FileNotFoundError:
                return self._error(
                    400,
                    "No index yet. Run `kb add <folder>` to build one.",
                )
            except (IncompatibleIndexError, ValueError) as err:
                return self._error(400, str(err))
            self._json(200, {"results": results})

        def _ask(self, body: dict) -> None:
            question = str(body.get("question") or "").strip()
            if not question:
                return self._error(400, "A question is required.")
            options = self._retrieval_options(body, lambda d, key: d.get(key))
            try:
                result = answer(question, index_dir=index_dir, **options)
            except FileNotFoundError:
                return self._error(
                    400,
                    "No index yet. Run `kb add <folder>` to build one.",
                )
            except (IncompatibleIndexError, ValueError, KbLLMError) as err:
                return self._error(400, str(err))
            self._json(
                200,
                {"answer": result["answer"], "citations": result["citations"]},
            )

        def _clip(self, body: dict) -> None:
            url = str(body.get("url") or "").strip()
            if not url.lower().startswith(("http://", "https://")):
                return self._error(400, "A http(s) URL is required.")
            clips = config.clips_dir()
            if not clips:
                return self._error(
                    400,
                    "No clips folder configured. "
                    "Run `kb clip --set-dir <folder>` first.",
                )
            doc = fetch_url(url)
            if doc is None:
                return self._error(
                    400,
                    "Could not fetch that URL. Check the address, your network, "
                    'and that the capture extra is installed: pip install -e ".[clip]"',
                )
            path = save_clip(doc, clips)
            if path is None:
                return self._json(200, {"path": None, "duplicate": True,
                                        "title": doc.title})
            ingest(clips, index_dir=index_dir)
            self._json(200, {"path": str(path), "duplicate": False,
                             "title": doc.title})

    return Handler


def make_server(
    index_dir: str | None = None,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    token: str | None = None,
) -> ThreadingHTTPServer:
    """Build (but do not start) the HTTP server.

    Separate from :func:`serve` so tests can drive a real socket on port 0 and
    shut it down deterministically.
    """
    if index_dir is None:
        index_dir = config.default_index_dir()
    return ThreadingHTTPServer((host, port), _build_handler(index_dir, token))


def _is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1")


def _lan_address() -> str | None:
    """Best-effort LAN IP for the "open this on your phone" hint."""
    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # No packets are sent for a UDP connect; it just picks the outbound
        # interface, which is the address a phone on the same wifi can reach.
        probe.connect(("192.0.2.1", 9))  # TEST-NET-1, guaranteed unroutable
        return probe.getsockname()[0]
    except OSError:
        return None
    finally:
        probe.close()


def serve(
    index_dir: str | None = None,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    token: str | None = None,
) -> None:
    """Run the web UI until interrupted.

    A non-loopback *host* exposes the knowledge base to the local network, so a
    token is minted when one was not supplied and printed as part of the URL.
    ``Ctrl-C`` stops the server cleanly.
    """
    if not _is_loopback(host) and not token:
        token = secrets.token_urlsafe(12)

    server = make_server(index_dir=index_dir, host=host, port=port, token=token)
    suffix = f"?t={token}" if token else ""

    print(f"kb is serving at  http://127.0.0.1:{server.server_port}/{suffix}")
    if not _is_loopback(host):
        lan = _lan_address()
        if lan:
            print(f"On your phone:    http://{lan}:{server.server_port}/{suffix}")
        print(
            "Reachable from your local network. The token keeps casual visitors "
            "out; do not expose this to the internet."
        )
    print("Ctrl-C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.shutdown()
        server.server_close()
