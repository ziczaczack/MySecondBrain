# kb — a pure-local personal knowledge base

`kb` ingests your notes and code, embeds them locally, and lets you search them
semantically — all on your own machine. An optional `kb ask` command synthesizes
a cited, natural-language answer from the passages it retrieves.

## Install

Install the package to get a `kb` command on your PATH:

```sh
pip install -e .
```

Optional features are pulled in as extras:

```sh
pip install -e ".[documents]"   # also index PDF and .docx files
pip install -e ".[synthesis]"   # enable `kb ask` (Anthropic SDK)
pip install -e ".[dev]"         # run the test suite
```

Every command below can be run either as `kb <cmd>` or `python -m kb <cmd>`.

## Quick start

```sh
# Index a folder of notes and code
python -m kb ingest ~/notes

# Search the index (pure-local retrieval, no network)
python -m kb query "postgres index types"

# Ask a question and get a synthesized, cited answer
python -m kb ask "how do partial indexes work in postgres?"
```

## Daily workflow

After `pip install -e .`, the `kb` command lives in your Python environment's
scripts directory (e.g. `.venv/Scripts/kb` on Windows, `.venv/bin/kb` on
macOS/Linux). Activate the environment — or add that directory to your `PATH` —
so you can call `kb` directly. Everything below also works as `python -m kb …`.

**1. Build your library once.** Prefer `add` over a bare `ingest`: `add`
*registers* the folder as a source, so you can re-index or `watch` it later with
no arguments. The first run downloads the embedding model (a few hundred MB,
one time), then embeds every note.

```sh
kb add ~/notes            # register + index a folder
kb add ~/code/my-project  # add as many sources as you like
kb sources                # list what's registered
kb status                 # how many chunks are indexed
```

**2. Search — pure-local, no network.**

```sh
kb query "voice-to-text desktop app"
kb query "Polymarket trading bot" --hybrid   # add --hybrid for keyword/proper-noun queries
kb query "multi-agent" -k 8 --kind note      # more hits, notes only
```

`--hybrid` is opt-in: it helps queries that hinge on a distinctive exact token,
but plain semantic search is the safe default for conceptual questions.

**3. Ask — the only step that hits the network** (see setup below).

```sh
export ANTHROPIC_API_KEY=sk-ant-...
kb ask "what was my plan for auto-editing videos?"
kb ask "..." --no-synthesis   # skip the LLM; return raw passages like `kb query`
```

**4. Keep the index fresh.** Re-running `add` on a registered source is
incremental — only changed files are re-embedded:

```sh
kb add ~/notes   # incremental refresh
kb watch         # or leave this running to auto-reindex on change (Ctrl+C to stop)
```

> **Switching embedding models?** The model that built an index is stamped into
> it, and `kb query`/`kb ask` refuse to search across a mismatch. After changing
> `KB_EMBED_MODEL`, rebuild once: `kb ingest <dir> --rebuild`.

## Commands

| Command            | What it does                                                        |
| ------------------ | ------------------------------------------------------------------ |
| `ingest <dir>`     | Embed and index a directory of notes, code, and (with `[documents]`) PDF/`.docx` files. |
| `ingest-bookmarks` | Index bookmarks from a Chrome/Edge `Bookmarks` JSON file.          |
| `add <path>`       | Register a folder (or `--bookmarks` file) as a source and index it.|
| `query "<q>"`      | Search the index. Pure-local retrieval, no API call.               |
| `ask "<q>"`        | Ask a question; an LLM synthesizes an answer with citations.        |
| `sources`          | List registered sources.                                           |
| `watch`            | Watch registered folders and auto-reindex on change.               |
| `status`           | Show statistics about an existing index.                           |

### Shared query flags

Both `query` and `ask` accept the same retrieval options:

| Flag                  | Meaning                                                            |
| --------------------- | ----------------------------------------------------------------- |
| `-k <n>`              | Number of results to retrieve (default: 5).                       |
| `--since <window>`    | Only results modified within a window, e.g. `7d`, `30d`, or `YYYY-MM-DD`. |
| `--kind <code\|note>` | Filter results by kind: `code` or `note`.                         |
| `--hybrid`            | Fuse semantic + keyword (BM25) ranking via RRF.                   |
| `--index-dir <dir>`   | Index directory to search (default: the managed knowledge base).  |
| `--json`              | Output results as JSON.                                           |

## Asking questions: `kb ask`

`kb ask` runs the same local retrieval as `kb query`, then sends the question and
the retrieved passages to the Claude API to synthesize a grounded answer. The
answer carries inline `[n]` citations, followed by a numbered **Sources** list
that maps each marker back to a file and line.

```sh
python -m kb ask "what should I bring for a cold-weather day hike?"
```

Example output:

```
For a cold-weather day hike, layer with a moisture-wicking base, an insulating
mid-layer, and a waterproof shell [1]. Carry extra food, water, and a headlamp,
and pack a map and compass even if you have GPS [2]. Tell someone your route and
expected return time before you leave [1][2].

Sources:
  [1] hiking-gear.txt:12
  [2] hiking-gear.txt:48
```

`kb ask` accepts every shared query flag, so you can scope the retrieval before
synthesis:

```sh
python -m kb ask "recent tax deadlines?" --since 30d --kind note -k 8
python -m kb ask "async runtime tradeoffs" --hybrid
```

With `--json`, `ask` emits a machine-readable object containing the `answer`
string and the `citations` list (each citation is `{n, filename, start_line}`):

```sh
python -m kb ask "..." --json
```

### Raw retrieval without synthesis

`--no-synthesis` skips the LLM call entirely and returns the raw retrieval
results — identical to running `kb query`. Use it when you want the matching
passages without contacting any API:

```sh
python -m kb ask "..." --no-synthesis
```

## Setup for synthesis

Synthesis requires an Anthropic API key in your environment:

```sh
export ANTHROPIC_API_KEY=sk-ant-...
```

If the key is unset, `kb ask` fails with a friendly message instead of a stack
trace:

```
ANTHROPIC_API_KEY is not set. Export it before running kb:
  export ANTHROPIC_API_KEY=sk-ant-...
```

The key is read from the environment on every run and is **never written to
disk**.

### Choosing the synthesis model

The synthesis model defaults to `claude-opus-4-8`. Override it with the
`KB_MODEL` environment variable:

```sh
export KB_MODEL=claude-opus-4-8
```

## Embedding model

Retrieval embeds locally with a **multilingual** model
(`paraphrase-multilingual-MiniLM-L12-v2`) so Chinese/Japanese/Korean notes land
in the same vector space as English ones. Both the dense and the `--hybrid`
(BM25) paths are CJK-aware: CJK text is segmented at character granularity for
embedding and into 2-character shingles for keyword matching.

Override the model with `KB_EMBED_MODEL` (e.g. the smaller English-only
`all-MiniLM-L6-v2`):

```sh
export KB_EMBED_MODEL=all-MiniLM-L6-v2
```

The model that built an index is stamped into it. Switching models requires a
re-ingest — `kb query`/`kb ask` refuse to search an index built with a
different model rather than return garbage. Rebuild with:

```sh
python -m kb ingest <dir> --rebuild
```

## Privacy

`kb` is local-first by design:

- **Ingestion, embedding, and retrieval are 100% local.** Walking your files,
  computing embeddings, and ranking results — including semantic search and the
  `--hybrid` (BM25 + RRF) path — never leave your machine.
- **Only the final `kb ask` synthesis step contacts the network.** When you run
  `kb ask` (without `--no-synthesis`), the question plus the retrieved passages
  are sent to the Anthropic API to produce the answer.
- **`kb query` and `kb ask --no-synthesis` never call any API.** They perform
  pure-local retrieval only.

If you never run `kb ask`, no data ever leaves your machine.
