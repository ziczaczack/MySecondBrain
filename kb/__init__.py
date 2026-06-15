"""kb -- a pure-local personal knowledge base CLI.

Ingest .md/.txt notes into a local numpy-backed vector index, then run
semantic queries against them. See ``kb.ingest`` and ``kb.query``.
"""

from __future__ import annotations

from .ingest import ingest
from .query import query

__all__ = ["ingest", "query"]
