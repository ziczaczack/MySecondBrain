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
- **Chunked indexing**: each file is split into overlapping ~200-word windows (40
  words of overlap) and every chunk is embedded as its own vector, so a search
  matches the specific passage that answers your query rather than a whole file.

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

### 4. Add a folder and ask

`kb` is **zero-config**: you never have to name or manage an index path. Just
register a folder and ask questions of your **managed knowledge base**.

```sh
python -m kb add <folder>          # register a notes/code folder and index it
python -m kb ask "your question"   # ask the managed knowledge base
```

`add` registers the folder as a **source** and indexes it in one step; `ask`
searches the managed knowledge base. Neither command needs an index path — the
index lives in a per-user managed home (see *Where your data lives* below).

List what you have registered, and keep it fresh automatically:

```sh
python -m kb sources               # list registered sources
python -m kb watch                 # auto-reindex registered folders on change
```

`ask` prints the Top-5 matches, each with a similarity score and an `excerpt` —
the best-matching chunk passage from the note. It looks roughly like this:

```text
python -m kb ask "Rust 异步运行时"

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

### `add`

Register a folder (or a bookmarks file) as a **source** and index it in one step
— the primary, zero-config way to load content. No index path required.

```sh
python -m kb add <folder> [--bookmarks]
```

| Argument / flag | Required | Default                | Description                                                        |
|-----------------|----------|------------------------|--------------------------------------------------------------------|
| `path`          | yes      | —                      | Folder of notes/code, or a Chrome/Edge `Bookmarks` file with `--bookmarks`. |
| `--bookmarks`   | no       | off                    | Treat `<path>` as a Chrome/Edge `Bookmarks` JSON file.             |
| `--index-dir`   | no       | managed knowledge base | Override the managed index location (advanced/multi-corpus).        |

The source is recorded in the registry (`python -m kb sources`) so `watch` can
re-index it later.

### `ask`

Ask the managed knowledge base a natural-language question — a friendly alias
for `query` against the managed index, with the same flags.

```sh
python -m kb ask <question> [-k N] [--since WINDOW] [--kind code|note] [--hybrid] [--json]
```

| Argument / flag | Required | Default                | Description                                          |
|-----------------|----------|------------------------|------------------------------------------------------|
| `question`      | yes      | —                      | Natural-language query string.                       |
| `-k`            | no       | `5`                    | Number of results to return.                         |
| `--since`       | no       | —                      | Only results within a window, e.g. `7d`, `30d`, or `YYYY-MM-DD`. |
| `--kind`        | no       | —                      | Filter by kind: `code` or `note`.                    |
| `--hybrid`      | no       | off                    | Fuse semantic + keyword (BM25) ranking via RRF.      |
| `--json`        | no       | off                    | Output results as JSON.                              |
| `--index-dir`   | no       | managed knowledge base | Search a different index (advanced/multi-corpus).    |

### `sources`

List the sources you have registered with `add`.

```sh
python -m kb sources
```

Prints one line per source (its kind and path). With nothing registered yet, it
tells you to add one with `python -m kb add <folder>`.

### `watch`

Watch your registered folders and **auto-reindex on change**, so the managed
knowledge base stays current without re-running `add`.

```sh
python -m kb watch [--interval N]
```

| Argument / flag | Required | Default                | Description                                                |
|-----------------|----------|------------------------|------------------------------------------------------------|
| `--interval`    | no       | `3`                    | Polling interval in seconds.                               |
| `--index-dir`   | no       | managed knowledge base | Override the managed index location (advanced).            |

Runs until interrupted (Ctrl-C).

### `ingest`

Embed and index a directory of `.md`/`.txt` notes.

```sh
python -m kb ingest <dir> [--index-dir DIR]
```

| Argument / flag | Required | Default                | Description                                              |
|-----------------|----------|------------------------|----------------------------------------------------------|
| `dir`           | yes      | —                      | Directory to walk **recursively** for `.md`/`.txt` notes. |
| `--index-dir`   | no       | managed knowledge base | Directory the index is written into (created if missing). |

`--index-dir` now defaults to the **managed knowledge base** (not `.kb_index`).
Pass an explicit `--index-dir` only for the advanced/multi-corpus path where you
want a separate, self-contained index. For the everyday workflow prefer
`add` + `ask`.

Empty or whitespace-only files are skipped. On success it prints how many notes
were indexed.

### `ingest-bookmarks`

Index your browser bookmarks from a Chrome/Edge `Bookmarks` JSON file — the same
Source pipeline as `ingest`, just a different origin.

```sh
python -m kb ingest-bookmarks <path-to-Bookmarks> [--index-dir DIR] [--rebuild]
```

| Argument / flag | Required | Default                | Description                                                  |
|-----------------|----------|------------------------|--------------------------------------------------------------|
| `path`          | yes      | —                      | Path to the Chrome/Edge `Bookmarks` JSON file.               |
| `--index-dir`   | no       | managed knowledge base | Directory the index is written into (created if missing).    |
| `--rebuild`     | no       | off                    | Ignore any existing index and re-embed everything afresh.    |

For the simple path, `python -m kb add <Bookmarks> --bookmarks` registers and
indexes bookmarks in one step; `--index-dir` is the advanced/multi-corpus
override.

The `Bookmarks` file lives alongside your browser profile:

- **Windows:** `%LOCALAPPDATA%\Google\Chrome\User Data\Default\Bookmarks`
  (Edge: `%LOCALAPPDATA%\Microsoft\Edge\User Data\Default\Bookmarks`)
- **macOS:** `~/Library/Application Support/Google/Chrome/Default/Bookmarks`
- **Linux:** `~/.config/google-chrome/Default/Bookmarks`

Chrome may hold a lock on this file while running, so copy it out first and point
`kb` at the copy.

Each bookmark becomes a **note**-kind document embedding its title, URL, and
folder path; an incremental re-ingest reuses unchanged bookmarks.

### `query`

Search a previously built index.

```sh
python -m kb query <question> [--index-dir DIR] [-k N] [--hybrid]
```

| Argument / flag | Required | Default                | Description                                          |
|-----------------|----------|------------------------|------------------------------------------------------|
| `question`      | yes      | —                      | Natural-language query string.                       |
| `--index-dir`   | no       | managed knowledge base | Index directory to search.                           |
| `-k`            | no       | `5`                    | Number of results to return.                         |
| `--hybrid`      | no       | off                    | Fuse semantic + keyword (BM25) ranking via RRF.      |

`--index-dir` defaults to the **managed knowledge base** (not `.kb_index`); pass
it explicitly only for the advanced/multi-corpus path. For the everyday workflow,
`ask` is the friendly equivalent of `query` against the managed index.

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

## Where your data lives

`kb` keeps your index and source registry in a single **managed home**
directory, so you never have to track an index path. The location is per-OS:

- **Windows:** `%APPDATA%\kb`
- **macOS:** `~/Library/Application Support/kb`
- **Linux:** `$XDG_DATA_HOME/kb` if set, else `~/.local/share/kb`

Inside the home live the vector index (`<home>/index`) and the JSON source
registry (`<home>/sources.json`, written by `add` and read by `sources` and
`watch`). Set the **`KB_HOME`** environment variable to override the home
directory (handy for tests, throwaway corpora, or keeping the data on another
drive).

## How it works

1. **Ingest** (`kb/ingest.py`) is driven by a **`Source`** (`kb/source.py`), the
   seam that decides where content comes from: a Source yields `Document` objects
   and the core ingest loop chunks and embeds them without knowing their origin.
   There are now two concrete sources: `FileSource` walks a directory in sorted
   (deterministic) order, decodes each file, and supplies the `(mtime, size)`
   change token used for incremental re-embedding, while `BookmarkSource` reads a
   Chrome/Edge `Bookmarks` JSON file and yields one document per bookmark. Further
   origins (Firefox via `places.sqlite`, chat-log exports, APIs, databases) can be
   added by implementing the `Source` protocol, with no change to the ingest loop.
2. **Embedding** (`kb/embedding.py`) runs the texts through `all-MiniLM-L6-v2` on
   CPU, producing L2-normalised float32 vectors.
3. **Store** (`kb/store.py`) saves the vector matrix to `vectors.npy` and the
   metadata to `meta.json` inside the index directory.
4. **Query** (`kb/query.py`) embeds the question with the same model, loads the
   stored vectors, and computes cosine similarity against every vector
   (brute force). The top-`k` notes are returned best-first.

## Scope / limitations

- Sources are **local `.md`/`.txt` notes** plus **Chrome/Edge browser
  bookmarks** — no PDFs or other formats, and no cloud sources (Firefox via
  `places.sqlite` and chat-log exports are still future work).
- **Chunked search**: long notes are split into overlapping ~200-word windows, so
  results point at the specific passage that matched rather than the whole note.
- Search is **brute force** over all vectors. This is perfectly fine for
  hundreds to thousands of notes; it is not optimised for very large corpora
  (no approximate-nearest-neighbour index).
