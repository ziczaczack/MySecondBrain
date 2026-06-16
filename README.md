# kb — a pure-local personal knowledge base

`kb` is a tiny command-line tool that turns a folder of `.md`/`.txt` notes into a
searchable knowledge base. It runs an **ingest → embedding → semantic query**
pipeline entirely on your machine: it reads your notes, turns each one into a
vector with a local sentence-transformer model, and answers natural-language
questions by ranking notes with cosine similarity.

Everything is **pure CPU** and **offline** after the first run. The only network
access is a one-time download of the embedding model weights — there are no cloud
APIs, no accounts, and no telemetry. Your notes never leave your machine.

## Key design facts

- **Python 3.11+**.
- Embeddings come from **sentence-transformers** using the
  **`all-MiniLM-L6-v2`** model, loaded on **CPU** (384-dimensional, L2-normalised
  vectors).
- The index is stored as a **numpy `.npy` file** (`vectors.npy`) plus a **JSON
  metadata file** (`meta.json`) in a single index directory.
- Search is **brute-force cosine similarity** over all vectors — there is **no
  FAISS, no sqlite-vec, and no native compilation** to build or install.
- **One chunk per file**: each note is embedded as a single document. There is no
  paragraph or section splitting in this version.

## Quickstart

### 1. Create and activate a virtual environment

POSIX (macOS / Linux):

```sh
python -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```sh
pip install -r requirements.txt
```

> **Note:** the first run downloads ~80MB of `all-MiniLM-L6-v2` model weights.
> The weights are cached locally, so afterwards `kb` runs fully offline.

### 3. Run the tests

```sh
python -m pytest
```

This runs the acceptance test (`tests/test_query.py`), which ingests the sample
notes in `fixtures/` and proves that the Rust-async note ranks **#1** for the
query `"Rust 异步运行时"` (verifying cross-lingual semantic search works).

### 4. Ingest your notes

```sh
python -m kb ingest fixtures
```

This walks the directory recursively, embeds every `.md`/`.txt` file, and writes
the index. By default the index is written to **`.kb_index`** in the current
directory (override with `--index-dir`).

### 5. Query

```sh
python -m kb query "Rust 异步运行时"
```

The output lists the Top-5 matches, each with a similarity score and the note's
first non-empty line as a summary. It looks roughly like this:

```text
Top-5 results for: Rust 异步运行时
1. rust-async.md   (0.5xxx)
   # Rust Async Runtime: Tokio Deep Dive
2. postgres-indexing.md   (0.1xxx)
   ...
3. hiking-gear.txt   (0.0xxx)
   ...
4. sourdough.md   (0.0xxx)
   ...
5. tax-deadlines.md   (0.0xxx)
   ...
```

(Exact scores vary slightly by platform and model version; the ranking is what
matters — `rust-async.md` comes first.)

## Commands

All commands are invoked as `python -m kb <subcommand>`.

### `ingest`

Embed and index a directory of `.md`/`.txt` notes.

```sh
python -m kb ingest <dir> [--index-dir DIR]
```

| Argument / flag | Required | Default     | Description                                              |
|-----------------|----------|-------------|----------------------------------------------------------|
| `dir`           | yes      | —           | Directory to walk **recursively** for `.md`/`.txt` notes. |
| `--index-dir`   | no       | `.kb_index` | Directory the index is written into (created if missing). |

Empty or whitespace-only files are skipped. On success it prints how many notes
were indexed.

### `query`

Search a previously built index.

```sh
python -m kb query <question> [--index-dir DIR] [-k N] [--hybrid]
```

| Argument / flag | Required | Default     | Description                                          |
|-----------------|----------|-------------|------------------------------------------------------|
| `question`      | yes      | —           | Natural-language query string.                       |
| `--index-dir`   | no       | `.kb_index` | Index directory to search.                           |
| `-k`            | no       | `5`         | Number of results to return.                         |
| `--hybrid`      | no       | off         | Fuse semantic + keyword (BM25) ranking via RRF.      |

If no index exists in the given directory, `query` prints a helpful message
telling you to run `ingest` first and exits with a non-zero status.

#### Hybrid search

By default `query` ranks notes by pure semantic (embedding/cosine) similarity.
Passing `--hybrid` additionally runs a lexical **BM25** keyword ranking over the
same candidates and fuses the two rankings with **Reciprocal Rank Fusion
(RRF)**. Each result then carries `semantic_score` and `lexical_score` component
fields alongside the fused `score`.

Reach for `--hybrid` when you want to surface **exact terms, code symbols, or
function names** that a fuzzy embedding match can miss:

```sh
python -m kb query "frobnicate_8842 helper" --hybrid
```

It is **opt-in**: without the flag, search stays pure-semantic and existing
behavior is unchanged. `--hybrid` composes with the other query flags
(`--kind`, `--since`, `-k`, `--json`) just like any of them.

> **Caveat:** BM25 helps most for space-delimited tokens and code symbols.
> Text without word spacing (e.g. CJK) gains little lexically, but the semantic
> side already covers those queries — so hybrid is never worse than pure
> semantic, only sometimes better.

## How it works

1. **Ingest** (`kb/ingest.py`) walks the source directory recursively, collecting
   `.md`/`.txt` files in sorted (deterministic) order. Each non-empty file is read
   as UTF-8 and kept as a single document, along with metadata (path, filename,
   and a one-line summary taken from its first non-empty line).
2. **Embedding** (`kb/embedding.py`) runs the texts through `all-MiniLM-L6-v2` on
   CPU, producing L2-normalised float32 vectors.
3. **Store** (`kb/store.py`) saves the vector matrix to `vectors.npy` and the
   metadata to `meta.json` inside the index directory.
4. **Query** (`kb/query.py`) embeds the question with the same model, loads the
   stored vectors, and computes cosine similarity against every vector
   (brute force). The top-`k` notes are returned best-first.

## Scope / limitations

- Works only with **local `.md`/`.txt` notes** — no PDFs, web pages, or other
  formats, and no cloud sources.
- **One chunk per file**: long notes are embedded as a single vector, so search
  matches whole documents rather than specific paragraphs.
- Search is **brute force** over all vectors. This is perfectly fine for
  hundreds to thousands of notes; it is not optimised for very large corpora
  (no approximate-nearest-neighbour index).
