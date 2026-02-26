# rag-lette

Chat with your documents. Lightweight RAG CLI with swappable storage and embedding backends.

## Install

Install into a venv or similar:

```bash
pip install -e .
```

This installs the default setup with local LanceDB storage and Mistral for embeddings and LLM.

Then, copy `.env.example` to `.env` and add your API keys:

```bash
MISTRAL_API_KEY=...
```

See below for other provider options.

## Development checks

If you installed dev dependencies (`pip install -e ".[dev]"`), run:

```bash
make check
```

Useful targets:

- `make test` - run tests
- `make test-cov` - run tests with coverage summary

## Basic usage

**Ingest** a file or directory (default: LanceDB at `./db`):

```bash
rag ingest ./db ./docs/
rag ingest ./db paper.pdf --embed mistral --chunk-size 800
```

**Ask** a question:

```bash
rag ask ./db "What are the responsible AI principles?"
rag ask ./db "What is AFM?" --llm claude --top-k 8 --context
```

**Shorthand** — ask against `./db` without subcommands:

```bash
rag "What is AFM?"
```

## Architecture

`rag-lette` is organized as a small pipeline with interchangeable adapters:

- `cli.py` orchestrates ingestion and question-answer flows.
- `chunk.py` extracts and chunks documents into retrievable text units.
- `embed.py` creates vectors through provider-specific embedding adapters.
- `db.py` persists vectors and retrieves nearest context through DB adapters.
- `generate.py` turns retrieved context + user question into the final answer.
- `config.py` resolves defaults from config files, profile, flags, and env vars.

```mermaid
flowchart LR
    A[Documents] --> B[chunk.py]
    B --> C[embed.py]
    C --> D[db.py]
    Q[User question] --> E[ask command]
    E --> D
    D --> F[Retrieved context]
    F --> G[generate.py]
    Q --> G
    G --> H[Answer]
```

Extension points:

- Add a new vector backend by implementing `DbAdapter` in `db.py`.
- Add a new embedding provider by implementing `EmbedAdapter` in `embed.py`.
- Keep `cli.py` thin by routing provider-specific behavior into provider modules.

## Embedding providers

Pass `--embed <provider>` to `ingest` and `ask` (must match):

| Alias      | Provider  | Model                    | Key needed          | Status |
|------------|-----------|--------------------------|---------------------|--------|
| `mistral`  | Mistral   | mistral-embed            | `MISTRAL_API_KEY`   | ✅     |
| `voyage`   | VoyageAI  | voyage-3.5-lite          | `VOYAGE_API_KEY`    | ✅     |
| `openai`   | OpenAI    | text-embedding-3-small   | `OPENAI_API_KEY`    | ✅     |
| `gemini`   | Google    | gemini-embedding-001     | `GEMINI_API_KEY`    | ✅     |

You can also specify a model directly: `--embed voyageai/voyage-3.5` or `--embed openai/text-embedding-3-large`

Install the extra for your chosen provider:

```bash
pip install -e ".[voyageai]"   # VoyageAI
pip install -e ".[openai]"     # OpenAI
pip install -e ".[gemini]"    # Gemini (embeddings and/or LLM)
```

> Both `ingest` and `ask` must use the same embedding provider and model — the stored vectors must match the query vector dimensionality.

## LLM providers

Pass `--llm <provider>` to `ask`:

| Alias       | Model                    | Key needed           | Status |
|-------------|--------------------------|----------------------|--------|
| `mistral`   | ministral-3b-2512        | `MISTRAL_API_KEY`    | ✅     |
| `claude`    | claude-haiku-4-5         | `ANTHROPIC_API_KEY`  | ✅     |
| `openai`    | gpt-4o                   | `OPENAI_API_KEY`     | ✅     |
| `gemini`    | gemini-2.5-flash         | `GEMINI_API_KEY`     | ✅     |

Or specify a model directly: `--llm mistral/open-mistral-nemo`

Install the extra for your chosen provider:

```bash
pip install -e ".[anthropic]"  # Claude
pip install -e ".[openai]"     # OpenAI (also enables OpenAI embeddings)
pip install -e ".[gemini]"      # Gemini (LLM and/or embeddings)
```

### Streaming output

Streaming is enabled by default in `ask`, so answers appear token-by-token as they arrive:

```bash
rag ask ./db "Summarise the key findings"
rag ask ./db "Summarise the key findings" --llm claude
```

Use `--no-stream` if you prefer waiting for the full answer rendered as markdown (bold, code blocks, lists, etc.):

```bash
rag ask ./db "Summarise the key findings" --no-stream
```

Streaming is supported for Mistral, Anthropic, OpenAI, and Gemini.

## Document chunking

### Basic method (default)

The default chunker (`--chunk basic`) uses **pymupdf4llm** for PDFs and plain text reading for `.txt`/`.md`. pymupdf4llm converts PDFs to markdown before chunking, preserving document structure like headers, tables, and lists. **pymupdf-layout** is bundled alongside it to activate improved layout analysis — better multi-column reading order, table detection, and header/footer removal — with no GPU or heavy ML framework required.

Supported formats: `.pdf` `.txt` `.md`

### Unstructured — power document parsing

The **unstructured** method is the power option for business documents. It understands document structure (headings, sections, tables) and supports a wide range of formats out of the box.

#### Install

```bash
pip install -e ".[unstructured]"
```

#### Supported formats

`.pdf` `.docx` `.doc` `.pptx` `.ppt` `.xlsx` `.xls` `.odt` `.rtf` `.html` `.htm` `.eml` `.msg` `.csv` `.txt` `.md`

#### Usage

```bash
# Ingest a folder of mixed business documents
rag ingest ./db ./business-docs/ --chunk unstructured

# Single file
rag ingest ./db quarterly-report.pptx --chunk unstructured
```

#### PDF strategy

PDFs have a `--pdf-strategy` flag that controls the extraction engine. The other formats (DOCX, PPTX, XLSX, etc.) are XML-based and always fast — this flag is a no-op for them.

| Strategy | Speed | Use when |
|---|---|---|
| `fast` *(default)* | Fast | Digital PDFs — reports, exported slides, contracts |
| `hi-res` | Slow | Scanned PDFs, complex multi-column layouts, tables in images |
| `auto` | Varies | Let unstructured decide (adds detection overhead even for simple files) |

```bash
# Digital PDF — fast (default)
rag ingest ./db report.pdf --chunk unstructured

# Scanned or image-heavy PDF
rag ingest ./db scanned-invoice.pdf --chunk unstructured --pdf-strategy hi-res
```

> `hi-res` requires `detectron2` and/or `tesseract` to be installed. These are heavy dependencies not included in `rag[unstructured]` — see the [unstructured docs](https://docs.unstructured.io) for setup instructions.

#### How chunking works

Unlike the basic method's character splitter, unstructured chunks by document structure: it respects headings and section boundaries, combines short fragments, and caps chunks at `--chunk-size`. This tends to produce more coherent, semantically meaningful chunks for retrieval.

---

## Database backends

The default backend is **LanceDB**, a local vector database stored at `./db`. For shared or persistent deployments, Postgres with pgvector is also supported.

### Postgres backend

Postgres with [pgvector](https://github.com/pgvector/pgvector) is supported as an alternative to the default local LanceDB store. Useful for shared or persistent deployments.

#### Install extras

```bash
pip install -e ".[postgres]"
```

#### URI format

```
postgres://user:password@host:port/dbname
```

The adapter will `CREATE DATABASE` if it doesn't exist, connecting to an admin DB first (default: `postgres`). Override with the `admin_db` query param:

```
postgres://user:password@host:5432/mydb?admin_db=template1
```

#### Pre-flight validation

When using the Postgres backend, `rag ingest` performs a **pre-flight check** before any chunking or embedding work begins:

1. **Connectivity** — verifies it can reach the Postgres host
2. **Database access** — creates the target database if it doesn't exist
3. **Write permissions** — confirms the user has write access

This ensures you get a clear error message immediately rather than after waiting through potentially slow document processing.

```
Error: Cannot connect to Postgres at localhost:5432 (admin database: 'postgres').
Check that Postgres is running and credentials are correct.
```

#### Quickstart with Docker

```bash
docker run --rm --name rag-postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_DB=postgres \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

Then ingest and query:

```bash
rag ingest postgres://postgres:postgres@localhost/ragdb ./docs/
rag ask    postgres://postgres:postgres@localhost/ragdb "What is AFM?"
```

#### Quickstart on macOS (Homebrew)

```bash
brew services start postgresql@16
```

```bash
rag ingest postgres://$(whoami)@localhost/ragdb ./docs/
rag ask    postgres://$(whoami)@localhost/ragdb "What is AFM?"
```

> The user running the CLI needs permission to `CREATE DATABASE` and `CREATE EXTENSION` (i.e. superuser, or a pre-created DB with pgvector already enabled).

#### Options

| Option      | Description                                      |
|-------------|--------------------------------------------------|
| `--table`   | Table name (default: `embeddings`)               |
| `--embed`   | Embedding provider — must match at ingest/ask    |
| `--top-k`   | Chunks to retrieve (default: 5)                  |

Example with all options:

```bash
rag ingest 'postgres://user:pass@localhost/ragdb?admin_db=postgres' ./docs/ \
  --table my_docs --embed voyage --chunk-size 600

rag ask 'postgres://user:pass@localhost/ragdb' "What is AFM?" \
  --table my_docs --embed voyage --llm claude --top-k 8
```

## Config file

Defaults can be set in `./rag.toml` or `~/.rag.toml`:

```toml
[defaults]
embed = "mistral"
llm   = "claude"
top_k = 8
```

Named profiles let you switch between setups with a single flag:

```toml
[profiles.work]
db    = "postgres://user:pass@prod-host/ragdb"
embed = "voyage"
llm   = "claude"

[profiles.local]
db    = "./db"
embed = "mistral"
llm   = "mistral"
```

Activate a profile with `--profile`:

```bash
rag ingest --profile work ./docs/
rag ask    --profile work "What is the policy on X?"

# Profile values are overridden by explicit flags:
rag ask --profile work --llm openai "What is the policy on X?"
```

**Precedence (low → high):** `~/.rag.toml` < `./rag.toml` < `--profile` < CLI flags < env vars (`RAG_LLM`, `RAG_EMBED`, `RAG_DB`, etc.)

---

## Google Gemini and Vertex AI

Use Google's Gemini or Vertex AI in three different ways:

1. **Gemini as LLM or embedder** — Use Gemini for generation and/or embeddings with your existing local or Postgres store
2. **Gemini API** — Upload files to Gemini and ask in one shot; no vector store, no separate ingest
3. **Vertex AI RAG Engine** — Full managed RAG where Vertex handles chunking, embedding, and retrieval with persistent storage in Google Cloud

### Gemini as LLM or embedder

Use Gemini for **generation** and/or **embeddings** with your existing local or Postgres store. Chunking and storage stay the same; only the embed and LLM calls use Gemini.

**Requires:** `pip install -e ".[gemini]"` and `GEMINI_API_KEY` in `.env`.

```bash
# Embed with Gemini, store in LanceDB
rag ingest ./db ./docs/ --embed gemini

# Generate with Gemini (retrieve from existing DB)
rag ask ./db "What is AFM?" --llm gemini
```

### Gemini API

Upload files to Gemini and ask in one shot. No vector store, no separate ingest. Best for ad-hoc questions against a handful of documents.

**Requires:** `pip install -e ".[gemini]"` and `GEMINI_API_KEY`.

**Limitations:**

- Files are **temporary** on Google's servers (expire after 48 hours).
- No persistent data store — each `rag gemini` run re-uploads the files.
- Limited by model context window (~2M tokens for Gemini 2.5).
- Best for: quick questions over a few PDFs or docs.

```bash
rag gemini ./docs/ "What is AFM?"
rag gemini paper.pdf "Summarize the key findings" --model gemini-2.5-pro
```

### Vertex AI RAG Engine

Full managed RAG: Vertex handles chunking, embedding, and retrieval; you use a persistent **corpus** in Google Cloud. Generation uses Gemini.

**Requires:**

- GCP project with [Vertex AI API](https://cloud.google.com/vertex-ai/docs/quickstart) enabled.
- `gcloud auth application-default login` (Application Default Credentials).
- `pip install -e ".[vertex]"`.

**URI format:** `vertex://PROJECT_ID/CORPUS_NAME` (corpus is created automatically if it doesn't exist).

```bash
# Ingest: uploads files to Vertex; Vertex chunks and embeds
rag ingest vertex://my-gcp-project/my-corpus ./docs/

# Ask: retrieves from corpus and generates with Gemini (use --llm gemini)
rag ask vertex://my-gcp-project/my-corpus "What is AFM?" --llm gemini
```

Vertex ingest supports the same basic file types as local ingest (`.pdf`, `.txt`, `.md`). For ask with Vertex, you can omit `--embed`; retrieval uses Vertex's own embedding.

---

## References

### Supported Providers Summary

Complete reference of all configurable providers, models, and strategies:

| Category | Option | Alias | Provider | Default Model | API Key | Extra Install |
|----------|--------|-------|----------|---------------|---------|----------------|
| **Embedding** | `--embed mistral` | `mistral` | Mistral | `mistral-embed` | `MISTRAL_API_KEY` | `[default]` |
| | `--embed voyage` | `voyage` / `voyageai` | VoyageAI | `voyage-3.5-lite` | `VOYAGE_API_KEY` | `[voyageai]` |
| | `--embed openai` | `openai` | OpenAI | `text-embedding-3-small` | `OPENAI_API_KEY` | `[openai]` |
| | `--embed gemini` | `gemini` | Google | `gemini-embedding-001` | `GEMINI_API_KEY` | `[gemini]` |
| **LLM** | `--llm mistral` | `mistral` | Mistral | `ministral-3b-2512` | `MISTRAL_API_KEY` | `[default]` |
| | `--llm mistral-large` | `mistral-large` | Mistral | `mistral-large-2512` | `MISTRAL_API_KEY` | `[default]` |
| | `--llm claude` | `claude` / `anthropic` | Anthropic | `claude-haiku-4-5` | `ANTHROPIC_API_KEY` | `[anthropic]` |
| | `--llm gpt-4o` | `openai` | OpenAI | `gpt-4o` | `OPENAI_API_KEY` | `[openai]` |
| | `--llm gemini` | `gemini` | Google | `gemini-2.5-flash` | `GEMINI_API_KEY` | `[gemini]` |
| **Chunking** | `--chunk basic` | `basic` | Default | — | — | `[default]` |
| | `--chunk unstructured` | `unstructured` | Unstructured | — | — | `[unstructured]` |
| **PDF Strategy** | `--pdf-strategy fast` | `fast` | Unstructured | — | — | `[unstructured]` |
| | `--pdf-strategy hi-res` | `hi-res` | Unstructured | — | — | `[unstructured]` |
| | `--pdf-strategy auto` | `auto` | Unstructured | — | — | `[unstructured]` |
| **Database** | `./db` | — | LanceDB (default) | — | — | `[default]` |
| | `postgres://...` | — | Postgres + pgvector | — | — | `[postgres]` |
| **Gemini API** | `rag gemini` | — | Gemini File API | — | `GEMINI_API_KEY` | `[gemini]` |
| **Vertex** | `vertex://PROJECT/CORPUS` | — | Vertex AI RAG Engine | — | GCP auth | `[vertex]` |

**Notes:**
- Custom models can be specified with `provider/model` syntax: `--embed voyageai/voyage-3.5` or `--llm openai/gpt-4-turbo`
- Embedding provider and model must match between `ingest` and `ask` commands
- Default values are set in `RagConfig` with `--profile`, config files, or environment variables
- See [Document chunking](#document-chunking) for chunking strategies and [Database backends](#database-backends) for storage options
