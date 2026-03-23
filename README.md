# RAG Lette: Chat with your Documents

Lightweight RAG CLI with swappable storage and embedding backends.

## Highlights

Mix and match embeddings, LLMs, and storage — all from the CLI, no code required.

- **Embeddings:** Mistral · VoyageAI · OpenAI · Gemini · AWS Bedrock
- **LLMs:** Mistral · Claude · GPT-4o · Gemini · AWS Bedrock
- **Storage:** LanceDB (local) · Postgres/pgvector · Weaviate · S3 · Vertex AI RAG Engine · Bedrock Knowledge Bases
- **Parsing:** PDF, DOCX, PPTX, XLSX, and more via `basic` or `unstructured` chunkers

## Architecture

`rag-lette` is a small pipeline with interchangeable adapters:

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

- `cli.py` orchestrates ingestion and question-answer flows.
- `chunk.py` extracts and chunks documents into retrievable text units.
- `embed.py` creates vectors through provider-specific embedding adapters.
- `db.py` persists vectors and retrieves nearest context through DB adapters.
- `generate.py` turns retrieved context + user question into the final answer.
- `config.py` resolves defaults from config files, profile, flags, and env vars.

Ingest and query use the same core stages, but in reverse:

- Ingest: documents are chunked, embedded, and stored.
- Ask: the question is embedded, relevant chunks are retrieved, and the LLM generates the answer from that context.

Each stage uses swappable providers, so embeddings, LLMs, and storage backends can be chosen independently.

## Install

Install into a venv or similar. May need to **restart** venv for CLI.

```bash
pip install -e .
```

This installs the default setup, with local LanceDB storage and Mistral for embeddings and LLM.

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
- `make weaviate` - start local Weaviate container (`rag-weaviate`)

## Basic usage

**Ingest** a file or directory (default: LanceDB at `./db`):

```bash
rag ingest ./db ./docs/
rag ingest ./db paper.pdf --embed-provider mistral --embed-model mistral-embed --chunk-size 800
```

**List** ingested files:

```bash
rag list ./db
rag list s3://my-bucket/rag-db
```

**Ask** a question:

```bash
rag ask ./db "What are the responsible AI principles?"
rag -q ask ./db "What is AFM?" --llm-provider anthropic --llm-model claude-sonnet-4-5 --top-k 8 --context
rag ask ./db "list files"
```

`ask` now appends a per-query `Sources` footer with filename counts when the active backend returns source metadata for retrieved chunks. If a backend cannot expose filenames for retrieval results, the footer is omitted quietly.

`rag ask ... "list files"` is a built-in shortcut: it reuses the datastore source listing and prints the stored source names for the selected DB instead of running retrieval and generation.

**Shorthand** — ask against `./db` without subcommands:

```bash
rag "What is AFM?"
```

Extension points:

- Add a new vector backend by implementing `DbAdapter` in `db.py`.
- Add a new embedding provider by implementing `EmbedAdapter` in `embed.py`.
- Keep `cli.py` thin by routing provider-specific behavior into provider modules.

## Embedding providers

Pass `--embed-provider` and `--embed-model` to `ingest` and `ask`.
If you only set `--embed-model`, the CLI will infer the provider when it can:

| Alias      | Provider  | Model                    | Key needed          | Status |
|------------|-----------|--------------------------|---------------------|--------|
| `mistral`  | Mistral   | mistral-embed            | `MISTRAL_API_KEY`   | ✅     |
| `voyage`   | VoyageAI  | voyage-3.5-lite          | `VOYAGE_API_KEY`    | ✅     |
| `openai`   | OpenAI    | text-embedding-3-small   | `OPENAI_API_KEY`    | ✅     |
| `gemini`   | Google    | gemini-embedding-001     | `GEMINI_API_KEY`    | ✅     |
| `bedrock`  | AWS       | cohere.embed-english-v3      | AWS credentials | ✅     |

You can also specify a model directly: `--embed-model voyage-3.5-lite` or `--embed-model openai/text-embedding-3-large`

Install the extra for your chosen provider:

```bash
pip install -e ".[voyageai]"   # VoyageAI
pip install -e ".[openai]"     # OpenAI
pip install -e ".[gemini]"    # Gemini (embeddings and/or LLM)
pip install -e ".[bedrock]"   # AWS Bedrock
```

> Both `ingest` and `ask` must use the same embedding provider and model.
>
> For **LanceDB** and **Weaviate**, `rag ingest` now records the embedding provider/model in backend metadata and `rag ask` fails fast if you request a different embedding config later.
>
> Older LanceDB/Weaviate datasets created before this metadata was added do not have that protection, so keep the embedding settings consistent when querying them.

## LLM providers

Pass `--llm-provider` and `--llm-model` to `ask`.
If you only set `--llm-model`, the CLI will infer the provider when it can:

| Alias       | Model                    | Key needed           | Status |
|-------------|--------------------------|----------------------|--------|
| `mistral`   | ministral-3b-2512        | `MISTRAL_API_KEY`    | ✅     |
| `claude`    | claude-haiku-4-5         | `ANTHROPIC_API_KEY`  | ✅     |
| `claude-sonnet-4-5` | claude-sonnet-4-5 | `ANTHROPIC_API_KEY` | ✅ |
| `openai`    | gpt-4o                   | `OPENAI_API_KEY`     | ✅     |
| `gpt-5.4`   | gpt-5.4                  | `OPENAI_API_KEY`     | ✅     |
| `gemini`    | gemini-2.5-flash         | `GEMINI_API_KEY`     | ✅     |
| `bedrock`   | amazon.nova-lite-v1:0    | AWS credentials      | ✅     |

OpenAI also accepts `gpt-5.4-mini` and `gpt-5.4-nano`; those resolve to the current official mini/nano model IDs in the code.

Or specify a model directly: `--llm-model open-mistral-nemo`

Install the extra for your chosen provider:

```bash
pip install -e ".[anthropic]"  # Claude
pip install -e ".[openai]"     # OpenAI (also enables OpenAI embeddings)
pip install -e ".[gemini]"      # Gemini (LLM and/or embeddings)
pip install -e ".[bedrock]"    # AWS Bedrock
```

### Streaming output

Streaming is enabled by default in `ask`, so answers appear token-by-token as they arrive:

```bash
rag ask ./db "Summarise the key findings"
rag ask ./db "Summarise the key findings" --llm-provider anthropic --llm-model claude-sonnet-4-5
```

Use `--no-stream` if you prefer waiting for the full answer rendered as markdown (bold, code blocks, lists, etc.):

```bash
rag ask ./db "Summarise the key findings" --no-stream
```

Streaming is supported for Mistral, Anthropic, OpenAI, Gemini, and Bedrock.

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

### LanceDB backend

LanceDB is the default vector store for local workflows.

- Stores chunk text, source filename, and vectors.
- Records embedding provider/model metadata for new ingests.
- Validates the requested embedding config on `rag ask` for datasets created or updated with current versions of the CLI.

Older LanceDB datasets without stored embedding metadata still work, but the CLI cannot validate the original embedding config for them.

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
| `--embed-provider` / `--embed-model`   | Embedding provider/model — must match at ingest/ask    |
| `--top-k`   | Chunks to retrieve (default: 5)                  |

Example with all options:

```bash
rag ingest 'postgres://user:pass@localhost/ragdb?admin_db=postgres' ./docs/ \
  --table my_docs --embed-provider voyageai --embed-model voyage-3.5-lite --chunk-size 600

rag ask 'postgres://user:pass@localhost/ragdb' "What is AFM?" \
  --table my_docs --embed-provider voyageai --embed-model voyage-3.5-lite \
  --llm-provider anthropic --llm-model claude-sonnet-4-5 --top-k 8
```

### Weaviate backend

Weaviate is supported as a vector backend via `weaviate://...` URIs.

#### Install extras

```bash
pip install -e ".[weaviate]"
```

#### OrbStack + Weaviate quickstart (macOS)

1. Start OrbStack.
2. In this repo, start Weaviate:

```bash
make weaviate
```

3. Verify container status:

```bash
docker ps --filter name=rag-weaviate
```

4. Optional readiness check:

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8080/v1/.well-known/ready
```

Expected status: `200` (ready) or `503` (still starting).

The Makefile target creates/starts a local container named `rag-weaviate` on:

- HTTP: `localhost:8080`
- gRPC: `localhost:50051`

#### Step-by-step RAG workflow with Weaviate

Use the Weaviate URI as your DB and keep your embedding provider consistent between ingest and ask.

Weaviate ingests now record embedding provider/model metadata in a companion metadata collection. `rag ask` validates that the requested embedding config matches before querying.

1. Ingest documents:

```bash
rag ingest weaviate://localhost:8080 ./docs/ --embed-provider mistral --embed-model mistral-embed
```

2. Confirm what was ingested:

```bash
rag list weaviate://localhost:8080
```

3. Ask questions:

```bash
rag ask weaviate://localhost:8080 "What are the responsible AI principles?" \
  --embed-provider mistral --embed-model mistral-embed \
  --llm-provider anthropic --llm-model claude-sonnet-4-5 --top-k 5
```

4. Optional shorthand (uses default `./db`, so not for Weaviate):
   Use explicit `rag ask weaviate://...` for Weaviate workflows.

#### Notes

- Default table/collection is `embeddings`; override with `--table <name>` if needed.
- Collection names are sanitized for Weaviate automatically.
- Older Weaviate collections created before embedding metadata was added still work, but the CLI cannot validate their original embedding config.
- For local OrbStack usage, `weaviate://localhost:8080` is the simplest URI.

#### Container lifecycle

```bash
# Stop without deleting data inside the container
docker stop rag-weaviate

# Start again later
docker start rag-weaviate
```

## Config file

Defaults can be set in `./rag.toml` or `~/.rag.toml`:

```toml
[defaults]
embed-provider = "mistral"
embed-model = "mistral-embed"
llm-provider = "anthropic"
llm-model = "claude-sonnet-4-5"
top_k = 8
```

Named profiles let you switch between setups with a single flag:

```toml
[profiles.work]
db    = "postgres://user:pass@prod-host/ragdb"
embed-provider = "voyageai"
embed-model = "voyage-3.5-lite"
llm-provider = "anthropic"
llm-model = "claude-sonnet-4-5"

[profiles.local]
db    = "./db"
embed-provider = "mistral"
embed-model = "mistral-embed"
llm-provider = "mistral"
llm-model = "ministral-3b-2512"
```

Activate a profile with `--profile`:

```bash
rag ingest --profile work ./docs/
rag ask    --profile work "What is the policy on X?"

# Profile values are overridden by explicit flags:
rag ask --profile work --llm-provider openai --llm-model gpt-5.4 "What is the policy on X?"
```

**Precedence (low → high):** `~/.rag.toml` < `./rag.toml` < `--profile` < CLI flags < env vars (`RAG_LLM`, `RAG_LLM_PROVIDER`, `RAG_LLM_MODEL`, `RAG_EMBED`, `RAG_EMBED_PROVIDER`, `RAG_EMBED_MODEL`, `RAG_DB`, etc.)

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
rag ingest ./db ./docs/ --embed-provider gemini --embed-model gemini-embedding-001

# Generate with Gemini (retrieve from existing DB)
rag ask ./db "What is AFM?" --llm-provider gemini --llm-model gemini-2.5-flash
```

### Gemini API

Upload files to Gemini and ask in one shot. No vector store, no separate ingest. Best for ad-hoc questions against a handful of documents.

**Requires:** `pip install -e ".[gemini]"` and `GEMINI_API_KEY`.

**Supported file flow:**

- `--gemini-mode auto` (default): route to File Search when Office files are present, otherwise use direct file upload.
- `--gemini-mode direct`: force direct upload mode.
- `--gemini-mode file-search`: force File Search mode.

**Supported extensions in `rag gemini`:**

- Direct-compatible: `.pdf`, `.txt`, `.md`, `.html`, `.htm`, `.csv`
- Office (auto-routed through File Search import): `.doc`, `.docx`, `.ppt`, `.pptx`, `.xls`, `.xlsx`

**Limitations:**

- Files are **temporary** on Google's servers (expire after 48 hours).
- No persistent data store — each `rag gemini` run re-uploads the files.
- Limited by model context window (~2M tokens for Gemini 2.5).
- Best for: quick questions over a few PDFs or docs.
- Office files can spend time in Gemini `PROCESSING` before import/indexing completes.

```bash
rag gemini ./docs/ "What is AFM?"
rag gemini paper.pdf "Summarize the key findings" --model gemini-2.5-pro
rag gemini ./docs/ "What is AFM?" --gemini-mode auto
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

# Ask: retrieves from corpus and generates with Gemini (use --llm-provider gemini --llm-model gemini-2.5-flash)
rag ask vertex://my-gcp-project/my-corpus "What is AFM?" --llm-provider gemini --llm-model gemini-2.5-flash
```

Vertex ingest supports the same basic file types as local ingest (`.pdf`, `.txt`, `.md`). For ask with Vertex, you can omit `--embed`; retrieval uses Vertex's own embedding.

---

## AWS Bedrock

Use AWS Bedrock for a fully AWS-native RAG stack. Two modes are available:

1. **Bedrock Knowledge Bases (fully managed)** — AWS handles chunking, embedding, and retrieval; you connect rag-lette to an existing KB
2. **Bedrock with LanceDB on S3 (manual)** — chunking runs locally, vectors stored in LanceDB on S3; you control the embedding model

No third-party API keys needed — auth is via the standard AWS credential chain (IAM roles, `~/.aws/credentials`, environment variables, etc.).

### Bedrock Knowledge Bases (fully managed)

Amazon Bedrock Knowledge Bases is the native AWS managed RAG service. Create a Knowledge Base in the AWS Console (the wizard provisions the IAM role, OpenSearch Serverless collection, and S3 data source), then point rag-lette at it. AWS handles chunking, embedding, and retrieval.

#### Prerequisites

- AWS account with [Bedrock model access](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html) enabled
- A Knowledge Base created in the [AWS Console](https://console.aws.amazon.com/bedrock/) (use the wizard — it provisions IAM role, OpenSearch Serverless, and S3 data source automatically)
- IAM user/role with `AmazonBedrockFullAccess` and S3 read/write on the KB's data source bucket
- Install the extras:

```bash
pip install -e ".[bedrock]"
```

#### URI format

```
bedrock-kb://KNOWLEDGE_BASE_ID
bedrock-kb://KNOWLEDGE_BASE_ID/DATA_SOURCE_ID   # specify data source explicitly
```

The Knowledge Base ID is shown on the KB overview page in the AWS Console (e.g. `ABCDE12345`).

#### Usage

```bash
# Ingest: uploads files to KB's S3 bucket, triggers ingestion job, waits for indexing
rag ingest bedrock-kb://ABCDE12345 ./docs/

# Ask: retrieves via KB Retrieve API, generates with Bedrock LLM
rag ask bedrock-kb://ABCDE12345 "What is AFM?" --llm-provider bedrock --llm-model amazon.nova-lite-v1:0
```

No `--embed` flag needed — the KB uses the embedding model configured at creation time.

#### Profile example

```toml
[profiles.aws-kb]
db  = "bedrock-kb://ABCDE12345"
llm = "bedrock"
```

```bash
rag ingest --profile aws-kb ./docs/
rag ask    --profile aws-kb "What is AFM?"
```

---

### Bedrock with LanceDB on S3 (manual)

Use Bedrock embeddings and generation with LanceDB on S3 for vector storage. Chunking runs locally.

#### Prerequisites

- AWS account with [Bedrock model access](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html) enabled
  - Add `AmazonBedrockFullAccess` to IAM policy
- AWS credentials configured via `aws configure`, environment variables, etc.
- Install the extras:

```bash
pip install -e ".[bedrock]"
```

#### Usage

```bash
# Ingest: embed with Bedrock model, store in LanceDB on S3
rag ingest s3://my-bucket/rag-db ./docs/ --embed-provider bedrock --embed-model cohere.embed-english-v3

# Ask: embed query and use LLM, all from Bedrock
rag ask s3://my-bucket/rag-db "What is AFM?" --embed-provider bedrock --embed-model cohere.embed-english-v3 \
  --llm-provider bedrock --llm-model amazon.nova-lite-v1:0
```

The `--embed` provider must match between `ingest` and `ask`.

#### Embedding models

| Model | Dimensions | Notes |
|---|---|---|
| `cohere.embed-english-v3` *(default)* | 1024 | Strong retrieval performance |
| `amazon.titan-embed-text-v2:0` | 1024 | Good alternative; native AWS model |

```bash
rag ingest s3://my-bucket/rag-db ./docs/ --embed-provider bedrock --embed-model amazon.titan-embed-text-v2:0
rag ask    s3://my-bucket/rag-db "What is AFM?" --embed-provider bedrock --embed-model amazon.titan-embed-text-v2:0 --llm-provider bedrock --llm-model amazon.nova-lite-v1:0
```

#### LLM models

The default is `amazon.nova-lite-v1:0`. The Bedrock adapter uses the [Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-call.html) — any Converse-compatible model works.

| Model | Notes |
|---|---|
| `amazon.nova-lite-v1:0` *(default)* | Fast and cost-effective |
| `amazon.nova-pro-v1:0` | Higher quality reasoning |
| `anthropic.claude-sonnet-4-5-20250514-v1:0` | Strong instruction-following and citation quality |

```bash
rag ask s3://my-bucket/rag-db "Summarise findings" --llm-provider bedrock --llm-model amazon.nova-pro-v1:0
```

#### Profile example

```toml
[profiles.aws]
db    = "s3://my-bucket/rag-db"
embed = "bedrock"
llm   = "bedrock"
```

```bash
rag ingest --profile aws ./docs/
rag ask    --profile aws "What is AFM?"
```

---

## References

### Supported Providers Summary

Complete reference of all configurable providers, models, and strategies:

| Category | Option | Alias | Provider | Default Model | API Key | Extra Install |
|----------|--------|-------|----------|---------------|---------|----------------|
| **Embedding** | `--embed-provider mistral` + `--embed-model mistral-embed` | `mistral` | Mistral | `mistral-embed` | `MISTRAL_API_KEY` | — |
| | `--embed-provider voyageai` + `--embed-model voyage-3.5-lite` | `voyage` / `voyageai` | VoyageAI | `voyage-3.5-lite` | `VOYAGE_API_KEY` | `[voyageai]` |
| | `--embed-provider openai` + `--embed-model text-embedding-3-small` | `openai` | OpenAI | `text-embedding-3-small` | `OPENAI_API_KEY` | `[openai]` |
| | `--embed-provider gemini` + `--embed-model gemini-embedding-001` | `gemini` | Google | `gemini-embedding-001` | `GEMINI_API_KEY` | `[gemini]` |
| | `--embed-provider bedrock` + `--embed-model cohere.embed-english-v3` | `bedrock` | AWS Bedrock | `cohere.embed-english-v3` | AWS credentials | `[bedrock]` |
| **LLM** | `--llm-provider mistral` + `--llm-model ministral-3b-2512` | `mistral` | Mistral | `ministral-3b-2512` | `MISTRAL_API_KEY` | — |
| | `--llm-provider mistral` + `--llm-model mistral-large-2512` | `mistral-large` | Mistral | `mistral-large-2512` | `MISTRAL_API_KEY` | — |
| | `--llm-provider anthropic` + `--llm-model claude-sonnet-4-5` | `claude` / `anthropic` / `claude-sonnet-4-5` | Anthropic | `claude-sonnet-4-5` | `ANTHROPIC_API_KEY` | `[anthropic]` |
| | `--llm-provider openai` + `--llm-model gpt-4o` | `openai` | OpenAI | `gpt-4o` | `OPENAI_API_KEY` | `[openai]` |
| | `--llm-provider openai` + `--llm-model gpt-5.4` | `gpt-5.4` | OpenAI | `gpt-5.4` | `OPENAI_API_KEY` | `[openai]` |
| | `--llm-provider openai` + `--llm-model gpt-5.4-mini` | `gpt-5.4-mini` | OpenAI | `gpt-5-mini-2025-08-07` | `OPENAI_API_KEY` | `[openai]` |
| | `--llm-provider openai` + `--llm-model gpt-5.4-nano` | `gpt-5.4-nano` | OpenAI | `gpt-5-nano-2025-08-07` | `OPENAI_API_KEY` | `[openai]` |
| | `--llm-provider gemini` + `--llm-model gemini-2.5-flash` | `gemini` | Google | `gemini-2.5-flash` | `GEMINI_API_KEY` | `[gemini]` |
| | `--llm-provider bedrock` + `--llm-model amazon.nova-lite-v1:0` | `bedrock` | AWS Bedrock | `amazon.nova-lite-v1:0` | AWS credentials | `[bedrock]` |
| **Chunking** | `--chunk basic` | `basic` | Default | — | — | — |
| | `--chunk unstructured` | `unstructured` | Unstructured | — | — | `[unstructured]` |
| **PDF Strategy** | `--pdf-strategy fast` | `fast` | Unstructured | — | — | `[unstructured]` |
| | `--pdf-strategy hi-res` | `hi-res` | Unstructured | — | — | `[unstructured]` |
| | `--pdf-strategy auto` | `auto` | Unstructured | — | — | `[unstructured]` |
| **Database** | `./db` | — | LanceDB (default) | — | — | — |
| | `postgres://...` | — | Postgres + pgvector | — | — | `[postgres]` |
| | `weaviate://host:8080` | — | Weaviate | — | Optional (`WEAVIATE_API_KEY`) | `[weaviate]` |
| **Gemini API** | `rag gemini` | — | Gemini File API | — | `GEMINI_API_KEY` | `[gemini]` |
| **Vertex** | `vertex://PROJECT/CORPUS` | — | Vertex AI RAG Engine | — | GCP auth | `[vertex]` |
| **Bedrock KB** | `bedrock-kb://KB_ID` | — | Bedrock Knowledge Bases | — | AWS credentials | `[bedrock]` |
| **AWS Bedrock** | `s3://bucket/path` (DB) | — | LanceDB on S3 | — | AWS credentials | `[bedrock]` |

**Notes:**
- You can now split provider and model explicitly with `--embed-provider` / `--embed-model` and `--llm-provider` / `--llm-model`.
- Custom models can still be specified with `provider/model` syntax on the legacy combined flags, but the split flags are the preferred form.
- Embedding provider and model must match between `ingest` and `ask` commands.
- Default values are set in `RagConfig` with `--profile`, config files, or environment variables.
- See [Document chunking](#document-chunking) for chunking strategies and [Database backends](#database-backends) for storage options
