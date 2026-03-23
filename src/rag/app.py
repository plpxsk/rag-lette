"""Gradio front-end for the RAG services."""
from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from rag.config import EMBED_ALIASES, LLM_ALIASES, RagConfig, load_config
from rag.gemini import GeminiFileApiClient
from rag.generate import generate, generate_stream
from rag.services import IngestionResult, IngestionService, QueryService, RetrievalResult

ingestion_service = IngestionService()
query_service = QueryService()
_source_executor = ThreadPoolExecutor(max_workers=1)

DB_PROVIDER_LABELS = {
    "lancedb": "LanceDB",
    "postgres": "Postgres",
    "weaviate": "Weaviate",
    "vertex": "Vertex AI RAG",
    "bedrock-kb": "Amazon Bedrock KB",
}
DB_PROVIDER_CHOICES = [(label, key) for key, label in DB_PROVIDER_LABELS.items()]
MANAGED_DB_PROVIDERS = {"vertex", "bedrock-kb"}
LLM_CHOICES = list(LLM_ALIASES) + ["custom"]
EMBED_CHOICES = list(EMBED_ALIASES) + ["custom"]
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]


def _is_managed_provider(provider: str | None) -> bool:
    return provider in MANAGED_DB_PROVIDERS


def _infer_embed_from_llm(llm_value: str) -> str:
    base = llm_value.split("/")[0]
    llm_provider = LLM_ALIASES.get(base, ("mistral", ""))[0]
    for embed_alias, (embed_provider, _) in EMBED_ALIASES.items():
        if embed_provider == llm_provider:
            return embed_alias
    return "mistral"


def _build_db_uri(
    provider: str | None,
    lancedb_path: str,
    postgres_uri: str,
    weaviate_uri: str,
    vertex_project: str,
    vertex_corpus: str,
    bedrock_kb_id: str,
    bedrock_data_source_id: str,
) -> str:
    if not provider:
        raise ValueError("Choose a database provider first.")
    if provider == "lancedb":
        path = lancedb_path.strip()
        if not path:
            raise ValueError("Enter a LanceDB path or URI.")
        return path
    if provider == "postgres":
        uri = postgres_uri.strip()
        if not uri:
            raise ValueError("Enter a Postgres URI.")
        return uri
    if provider == "weaviate":
        uri = weaviate_uri.strip()
        if not uri:
            raise ValueError("Enter a Weaviate URI.")
        return uri
    if provider == "vertex":
        project = vertex_project.strip()
        corpus = vertex_corpus.strip()
        if not project or not corpus:
            raise ValueError("Enter both a Vertex project ID and corpus name.")
        return f"vertex://{project}/{corpus}"
    if provider == "bedrock-kb":
        kb_id = bedrock_kb_id.strip()
        data_source_id = bedrock_data_source_id.strip()
        if not kb_id:
            raise ValueError("Enter a Bedrock Knowledge Base ID.")
        return f"bedrock-kb://{kb_id}/{data_source_id}" if data_source_id else f"bedrock-kb://{kb_id}"
    raise ValueError(f"Unknown DB provider: {provider}")


def _resolve_model_choice(
    choice: str,
    custom_provider: str,
    custom_model: str,
    aliases: dict[str, tuple[str, str]],
) -> str:
    if choice == "custom":
        provider = custom_provider.strip()
        model = custom_model.strip()
        if not provider or not model:
            raise ValueError("Custom provider and model are both required.")
        return f"{provider}/{model}"
    if choice not in aliases:
        raise ValueError(f"Unknown model choice: {choice}")
    return choice


def _normalize_skip_extensions(raw: str) -> set[str]:
    values: set[str] = set()
    for item in raw.split(","):
        entry = item.strip().lower()
        if not entry:
            continue
        values.add(entry if entry.startswith(".") else f".{entry}")
    return values


def _set_runtime_credentials(values: dict[str, str]) -> None:
    for env_name, value in values.items():
        if value.strip():
            os.environ[env_name] = value.strip()


def _apply_log_level(log_level: str) -> None:
    logging.getLogger().setLevel(getattr(logging, log_level.upper(), logging.WARNING))


def _resolve_selected_path(selection: str | list[str] | None, root_dir: str) -> Path:
    if selection is None or selection == []:
        raise ValueError("Select a file or directory to ingest first.")
    selected = selection[0] if isinstance(selection, list) else selection
    path = Path(selected)
    if not path.is_absolute():
        path = Path(root_dir) / path
    return path.expanduser().resolve()


def _config_summary(cfg: RagConfig) -> str:
    return (
        f"- DB: `{cfg.db}`\n"
        f"- Table: `{cfg.table}`\n"
        f"- LLM: `{cfg.llm_provider}/{cfg.llm_model}`\n"
        f"- Embed: `{cfg.embed_provider}/{cfg.embed_model}`\n"
        f"- top_k: `{cfg.top_k}`\n"
        f"- max_tokens: `{cfg.max_tokens}`"
    )


def _db_label(provider: str | None) -> str:
    return DB_PROVIDER_LABELS.get(provider, provider)


def _summarize_db_selection(
    provider: str | None,
    lancedb_path: str,
    postgres_uri: str,
    weaviate_uri: str,
    vertex_project: str,
    vertex_corpus: str,
    bedrock_kb_id: str,
    bedrock_data_source_id: str,
    table: str,
) -> str:
    try:
        uri = _build_db_uri(
            provider,
            lancedb_path,
            postgres_uri,
            weaviate_uri,
            vertex_project,
            vertex_corpus,
            bedrock_kb_id,
            bedrock_data_source_id,
        )
    except Exception:
        return "## Current Database\nNo database selected yet."

    if not uri.strip():
        return "## Current Database\nNo database selected yet."

    lines = [
        "## Current Database",
        f"**Provider:** {_db_label(provider)}",
        f"**Location:** `{uri}`",
    ]
    if not _is_managed_provider(provider):
        lines.append(f"**Table / Collection:** `{table}`")
    return "\n".join(lines)


def _format_sources(sources: list[str]) -> list[list[str]]:
    return [[source] for source in sources]


def _format_context(result: RetrievalResult, show_context: bool) -> str:
    if not show_context:
        return "Retrieved context is hidden. Enable `Show Retrieved Context` in settings."
    if not result.chunks:
        return "No context retrieved."
    lines = ["### Retrieved Context"]
    for index, chunk in enumerate(result.chunks, start=1):
        label = f"`{chunk.source}`  " if chunk.source else ""
        lines.append(f"**[{index}]** {label}{chunk.text}")
    return "\n\n".join(lines)


def _format_source_footer(result: RetrievalResult) -> str:
    if not result.has_sources:
        return ""
    lines = ["### Sources"]
    lines.extend(f"- `{source}`: {count}" for source, count in result.source_counts)
    lines.append(f"Total source chunks: {sum(count for _, count in result.source_counts)}")
    return "\n".join(lines)


def _append_message(
    history: list[dict[str, str]],
    *,
    role: str,
    content: str,
) -> list[dict[str, str]]:
    return [*history, {"role": role, "content": content}]


def _format_ingestion_result(result: IngestionResult, cfg: RagConfig, verbose: bool) -> str:
    lines: list[str] = []
    if result.rows_written == 0:
        lines.append("No new content ingested.")
    elif result.mode == "vector":
        lines.append(
            f"Ingested `{result.rows_written}` chunks into `{cfg.db}` "
            f"(table `{cfg.table}`)."
        )
    else:
        lines.append(f"Uploaded `{result.rows_written}` file(s) to `{cfg.db}`.")

    if result.skipped_files:
        lines.append("")
        lines.append("Skipped existing sources:")
        lines.extend(f"- `{source}`" for source in result.skipped_files)

    if result.failures:
        lines.append("")
        lines.append("Failures:")
        lines.extend(f"- `{file_path.name}`: {exc}" for file_path, exc in result.failures)

    if verbose:
        lines.append("")
        lines.append("Resolved configuration:")
        lines.append(_config_summary(cfg))

    return "\n".join(lines)


def _load_sources(cfg: RagConfig) -> list[str]:
    from rag.db import get_db_adapter

    adapter = get_db_adapter(cfg.db, cfg.table)
    return adapter.list_sources()


def _build_cfg(
    *,
    provider: str,
    table: str,
    profile: str,
    lancedb_path: str,
    postgres_uri: str,
    weaviate_uri: str,
    vertex_project: str,
    vertex_corpus: str,
    bedrock_kb_id: str,
    bedrock_data_source_id: str,
    llm_choice: str,
    llm_custom_provider: str,
    llm_custom_model: str,
    embed_choice: str,
    embed_custom_provider: str,
    embed_custom_model: str,
    chunk_method: str,
    chunk_size: int,
    chunk_overlap: int,
    pdf_strategy: str,
    top_k: int,
    max_tokens: int,
) -> RagConfig:
    overrides: dict[str, Any] = {
        "db": _build_db_uri(
            provider,
            lancedb_path,
            postgres_uri,
            weaviate_uri,
            vertex_project,
            vertex_corpus,
            bedrock_kb_id,
            bedrock_data_source_id,
        ),
        "llm": _resolve_model_choice(llm_choice, llm_custom_provider, llm_custom_model, LLM_ALIASES),
        "top_k": top_k,
        "max_tokens": max_tokens,
    }

    if not _is_managed_provider(provider):
        overrides["table"] = table
        overrides["embed"] = _resolve_model_choice(
            embed_choice,
            embed_custom_provider,
            embed_custom_model,
            EMBED_ALIASES,
        )
        overrides["chunk_method"] = chunk_method
        overrides["chunk_size"] = chunk_size
        overrides["chunk_overlap"] = chunk_overlap
        overrides["pdf_strategy"] = pdf_strategy

    profile_value = profile.strip() or None
    return load_config(profile=profile_value, overrides=overrides)


def create_demo(root_dir: str | Path | None = None):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is required for the GUI.\n"
            "Install it with: pip install -e \".[gui]\"\n"
            f"Original error: {exc}"
        ) from exc

    browser_root = Path(root_dir or Path.cwd()).expanduser().resolve()

    def update_provider_controls(provider: str | None, embed_choice: str):
        managed = _is_managed_provider(provider)
        return (
            gr.update(visible=provider == "lancedb"),
            gr.update(visible=provider == "postgres"),
            gr.update(visible=provider == "weaviate"),
            gr.update(visible=provider == "vertex"),
            gr.update(visible=provider == "bedrock-kb"),
            gr.update(visible=not managed),
            gr.update(visible=not managed),
            gr.update(visible=not managed),
            gr.update(visible=not managed and embed_choice == "custom"),
            gr.update(visible=not managed and embed_choice == "custom"),
        )

    def update_db_summary(
        provider: str | None,
        lancedb_path: str,
        postgres_uri: str,
        weaviate_uri: str,
        vertex_project: str,
        vertex_corpus: str,
        bedrock_kb_id: str,
        bedrock_data_source_id: str,
        table: str,
    ):
        return _summarize_db_selection(
            provider,
            lancedb_path,
            postgres_uri,
            weaviate_uri,
            vertex_project,
            vertex_corpus,
            bedrock_kb_id,
            bedrock_data_source_id,
            table,
        )

    def update_pdf_strategy(chunk_method: str):
        return gr.update(visible=chunk_method == "unstructured")

    def update_llm_custom(choice: str):
        visible = choice == "custom"
        return gr.update(visible=visible), gr.update(visible=visible)

    def update_embed_custom(choice: str, provider: str | None):
        visible = choice == "custom" and not _is_managed_provider(provider)
        return gr.update(visible=visible), gr.update(visible=visible)

    def refresh_sources(
        provider: str | None,
        table: str,
        profile: str,
        lancedb_path: str,
        postgres_uri: str,
        weaviate_uri: str,
        vertex_project: str,
        vertex_corpus: str,
        bedrock_kb_id: str,
        bedrock_data_source_id: str,
        log_level: str,
    ):
        _apply_log_level(log_level)
        try:
            db_uri = _build_db_uri(
                provider,
                lancedb_path,
                postgres_uri,
                weaviate_uri,
                vertex_project,
                vertex_corpus,
                bedrock_kb_id,
                bedrock_data_source_id,
            )
            cfg = load_config(
                profile=profile.strip() or None,
                overrides={"db": db_uri, "table": table},
            )
            sources = _load_sources(cfg)
            summary = _summarize_db_selection(
                provider,
                lancedb_path,
                postgres_uri,
                weaviate_uri,
                vertex_project,
                vertex_corpus,
                bedrock_kb_id,
                bedrock_data_source_id,
                table,
            )
            summary += f"\n**Loaded Sources:** `{len(sources)}`"
            return _format_sources(sources), "Sources refreshed.", summary
        except Exception as exc:
            summary = _summarize_db_selection(
                provider,
                lancedb_path,
                postgres_uri,
                weaviate_uri,
                vertex_project,
                vertex_corpus,
                bedrock_kb_id,
                bedrock_data_source_id,
                table,
            )
            summary += "\n**Loaded Sources:** `Unavailable`"
            return [], f"Could not load sources: {exc}", summary

    def ingest_files(
        selection: str | list[str] | None,
        provider: str | None,
        table: str,
        profile: str,
        lancedb_path: str,
        postgres_uri: str,
        weaviate_uri: str,
        vertex_project: str,
        vertex_corpus: str,
        bedrock_kb_id: str,
        bedrock_data_source_id: str,
        llm_choice: str,
        llm_custom_provider: str,
        llm_custom_model: str,
        embed_choice: str,
        embed_custom_provider: str,
        embed_custom_model: str,
        chunk_method: str,
        chunk_size: int,
        chunk_overlap: int,
        pdf_strategy: str,
        top_k: int,
        max_tokens: int,
        skip_extensions: str,
        overwrite: bool,
        verbose: bool,
        log_level: str,
        mistral_api_key: str,
        anthropic_api_key: str,
        openai_api_key: str,
        gemini_api_key: str,
        voyage_api_key: str,
        weaviate_api_key: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        aws_session_token: str,
        aws_default_region: str,
        google_application_credentials: str,
    ):
        _apply_log_level(log_level)
        _set_runtime_credentials(
            {
                "MISTRAL_API_KEY": mistral_api_key,
                "ANTHROPIC_API_KEY": anthropic_api_key,
                "OPENAI_API_KEY": openai_api_key,
                "GEMINI_API_KEY": gemini_api_key,
                "VOYAGE_API_KEY": voyage_api_key,
                "WEAVIATE_API_KEY": weaviate_api_key,
                "AWS_ACCESS_KEY_ID": aws_access_key_id,
                "AWS_SECRET_ACCESS_KEY": aws_secret_access_key,
                "AWS_SESSION_TOKEN": aws_session_token,
                "AWS_DEFAULT_REGION": aws_default_region,
                "GOOGLE_APPLICATION_CREDENTIALS": google_application_credentials,
            }
        )

        try:
            source_path = _resolve_selected_path(selection, str(browser_root))
            cfg = _build_cfg(
                provider=provider,
                table=table,
                profile=profile,
                lancedb_path=lancedb_path,
                postgres_uri=postgres_uri,
                weaviate_uri=weaviate_uri,
                vertex_project=vertex_project,
                vertex_corpus=vertex_corpus,
                bedrock_kb_id=bedrock_kb_id,
                bedrock_data_source_id=bedrock_data_source_id,
                llm_choice=llm_choice,
                llm_custom_provider=llm_custom_provider,
                llm_custom_model=llm_custom_model,
                embed_choice=embed_choice,
                embed_custom_provider=embed_custom_provider,
                embed_custom_model=embed_custom_model,
                chunk_method=chunk_method,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                pdf_strategy=pdf_strategy,
                top_k=top_k,
                max_tokens=max_tokens,
            )
            result = ingestion_service.ingest(
                cfg,
                source_path,
                skip_extensions=_normalize_skip_extensions(skip_extensions),
                overwrite=overwrite,
            )
            loaded_sources = _load_sources(cfg)
            sources = _format_sources(loaded_sources)
            summary = _summarize_db_selection(
                provider,
                lancedb_path,
                postgres_uri,
                weaviate_uri,
                vertex_project,
                vertex_corpus,
                bedrock_kb_id,
                bedrock_data_source_id,
                table,
            )
            summary += f"\n**Loaded Sources:** `{len(loaded_sources)}`"
            return (
                sources,
                _format_ingestion_result(result, cfg, verbose),
                summary,
            )
        except Exception as exc:
            summary = _summarize_db_selection(
                provider,
                lancedb_path,
                postgres_uri,
                weaviate_uri,
                vertex_project,
                vertex_corpus,
                bedrock_kb_id,
                bedrock_data_source_id,
                table,
            )
            summary += "\n**Loaded Sources:** `Unavailable`"
            return [], f"Ingest failed: {exc}", summary

    def chat_with_rag(
        message: str,
        history: list[dict[str, str]],
        provider: str | None,
        table: str,
        profile: str,
        lancedb_path: str,
        postgres_uri: str,
        weaviate_uri: str,
        vertex_project: str,
        vertex_corpus: str,
        bedrock_kb_id: str,
        bedrock_data_source_id: str,
        llm_choice: str,
        llm_custom_provider: str,
        llm_custom_model: str,
        embed_choice: str,
        embed_custom_provider: str,
        embed_custom_model: str,
        chunk_method: str,
        chunk_size: int,
        chunk_overlap: int,
        pdf_strategy: str,
        top_k: int,
        max_tokens: int,
        show_context: bool,
        stream: bool,
        verbose: bool,
        log_level: str,
        mistral_api_key: str,
        anthropic_api_key: str,
        openai_api_key: str,
        gemini_api_key: str,
        voyage_api_key: str,
        weaviate_api_key: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        aws_session_token: str,
        aws_default_region: str,
        google_application_credentials: str,
    ):
        if not message.strip():
            yield "", history, history, "Ask a question to begin.", ""
            return

        _apply_log_level(log_level)
        _set_runtime_credentials(
            {
                "MISTRAL_API_KEY": mistral_api_key,
                "ANTHROPIC_API_KEY": anthropic_api_key,
                "OPENAI_API_KEY": openai_api_key,
                "GEMINI_API_KEY": gemini_api_key,
                "VOYAGE_API_KEY": voyage_api_key,
                "WEAVIATE_API_KEY": weaviate_api_key,
                "AWS_ACCESS_KEY_ID": aws_access_key_id,
                "AWS_SECRET_ACCESS_KEY": aws_secret_access_key,
                "AWS_SESSION_TOKEN": aws_session_token,
                "AWS_DEFAULT_REGION": aws_default_region,
                "GOOGLE_APPLICATION_CREDENTIALS": google_application_credentials,
            }
        )

        try:
            cfg = _build_cfg(
                provider=provider,
                table=table,
                profile=profile,
                lancedb_path=lancedb_path,
                postgres_uri=postgres_uri,
                weaviate_uri=weaviate_uri,
                vertex_project=vertex_project,
                vertex_corpus=vertex_corpus,
                bedrock_kb_id=bedrock_kb_id,
                bedrock_data_source_id=bedrock_data_source_id,
                llm_choice=llm_choice,
                llm_custom_provider=llm_custom_provider,
                llm_custom_model=llm_custom_model,
                embed_choice=embed_choice,
                embed_custom_provider=embed_custom_provider,
                embed_custom_model=embed_custom_model,
                chunk_method=chunk_method,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                pdf_strategy=pdf_strategy,
                top_k=top_k,
                max_tokens=max_tokens,
            )
            retrieved = query_service.retrieve(cfg, message)
            context_markdown = _format_context(retrieved, show_context)
            source_footer_future = _source_executor.submit(_format_source_footer, retrieved)
            status = "Generating answer..."
            if verbose:
                status += "\n\nResolved configuration:\n" + _config_summary(cfg)

            updated_history = _append_message(history, role="user", content=message)
            updated_history = _append_message(updated_history, role="assistant", content="")
            yield "", updated_history, updated_history, status, context_markdown

            if stream:
                answer = ""
                for chunk in generate_stream(
                    query=message,
                    context=retrieved.chunks,
                    provider=cfg.llm_provider,
                    model=cfg.llm_model,
                    max_tokens=cfg.max_tokens,
                ):
                    answer += chunk
                    updated_history[-1]["content"] = answer
                    yield "", updated_history, updated_history, status, context_markdown
                source_footer = source_footer_future.result()
                if source_footer:
                    updated_history[-1]["content"] = f"{answer}\n\n{source_footer}"
                    yield "", updated_history, updated_history, status, context_markdown
            else:
                answer = generate(
                    query=message,
                    context=retrieved.chunks,
                    provider=cfg.llm_provider,
                    model=cfg.llm_model,
                    max_tokens=cfg.max_tokens,
                )
                source_footer = source_footer_future.result()
                updated_history[-1]["content"] = (
                    f"{answer}\n\n{source_footer}" if source_footer else answer
                )
                yield "", updated_history, updated_history, status, context_markdown
        except Exception as exc:
            updated_history = _append_message(history, role="user", content=message)
            updated_history = _append_message(
                updated_history,
                role="assistant",
                content=f"Error: {exc}",
            )
            yield "", updated_history, updated_history, f"Query failed: {exc}", ""

    def clear_chat():
        return [], [], "Chat cleared.", ""

    def gemini_answer(
        message: str,
        history: list[dict[str, str]],
        selection: str | list[str] | None,
        gemini_model: str,
        gemini_max_tokens: int,
        gemini_api_key: str,
    ):
        if not message.strip():
            yield "", history, history, "Ask a question to begin."
            return

        try:
            if gemini_api_key.strip():
                os.environ["GEMINI_API_KEY"] = gemini_api_key.strip()
            source_path = _resolve_selected_path(selection, str(browser_root))
            client = GeminiFileApiClient(model=gemini_model)
            prepared = client.prepare(source_path)
            if prepared.uploaded_count == 0:
                raise RuntimeError("No supported files found at the selected path.")
            updated_history = _append_message(history, role="user", content=message)
            updated_history = _append_message(updated_history, role="assistant", content="")
            mode_label = "File Search" if prepared.mode == "file_search" else "Direct"
            status = f"Uploaded {prepared.uploaded_count} file(s) via Gemini {mode_label}. Generating answer..."
            yield "", updated_history, updated_history, status
            updated_history[-1]["content"] = client.ask_prepared(
                message,
                prepared,
                max_tokens=gemini_max_tokens,
            )
            yield "", updated_history, updated_history, status
        except Exception as exc:
            updated_history = _append_message(history, role="user", content=message)
            updated_history = _append_message(
                updated_history,
                role="assistant",
                content=f"Error: {exc}",
            )
            yield "", updated_history, updated_history, f"Gemini failed: {exc}"

    with gr.Blocks(title="RAG Lette GUI", fill_height=True) as demo:
        rag_history = gr.State([])
        gemini_history = gr.State([])

        gr.Markdown(
            """
            # RAG Lette
            Ingest documents, review stored sources, and chat against the selected backend.
            """
        )

        with gr.Sidebar(open=False):
            with gr.Accordion("Database", open=True):
                db_provider = gr.Dropdown(
                    choices=DB_PROVIDER_CHOICES,
                    value=None,
                    label="DB Provider",
                    info="Choose the storage backend for ingest and retrieval.",
                )
                lancedb_path = gr.Textbox(
                    label="LanceDB Path or URI",
                    value="",
                    info="Local path or s3:// URI for LanceDB.",
                    placeholder="./db or s3://bucket/path",
                )
                postgres_uri = gr.Textbox(
                    label="Postgres URI",
                    value="",
                    placeholder="postgres://user:pass@localhost:5432/ragdb",
                    visible=False,
                )
                weaviate_uri = gr.Textbox(
                    label="Weaviate URI",
                    value="",
                    placeholder="weaviate://localhost:8080",
                    visible=False,
                )
                vertex_project = gr.Textbox(label="Vertex Project ID", visible=False)
                vertex_corpus = gr.Textbox(label="Vertex Corpus Name", visible=False)
                bedrock_kb_id = gr.Textbox(label="Bedrock Knowledge Base ID", visible=False)
                bedrock_data_source_id = gr.Textbox(
                    label="Bedrock Data Source ID",
                    visible=False,
                )
                table = gr.Textbox(
                    label="Table / Collection Name",
                    value="embeddings",
                )
                profile = gr.Textbox(
                    label="Config Profile",
                    info="Optional profile from rag.toml.",
                )

            with gr.Accordion("Models", open=False):
                llm_choice = gr.Dropdown(
                    choices=LLM_CHOICES,
                    value="mistral",
                    label="LLM",
                )
                llm_custom_provider = gr.Textbox(
                    label="Custom LLM Provider",
                    visible=False,
                )
                llm_custom_model = gr.Textbox(
                    label="Custom LLM Model",
                    visible=False,
                )
                embed_choice = gr.Dropdown(
                    choices=EMBED_CHOICES,
                    value="mistral",
                    label="Embedding Model",
                )
                embed_custom_provider = gr.Textbox(
                    label="Custom Embed Provider",
                    visible=False,
                )
                embed_custom_model = gr.Textbox(
                    label="Custom Embed Model",
                    visible=False,
                )

            with gr.Accordion("Ingest Settings", open=False):
                chunk_method = gr.Dropdown(
                    choices=["basic", "unstructured"],
                    value="basic",
                    label="Chunking Method",
                )
                chunk_size = gr.Number(label="Chunk Size", value=1000, precision=0)
                chunk_overlap = gr.Number(label="Chunk Overlap", value=200, precision=0)
                pdf_strategy = gr.Dropdown(
                    choices=["fast", "hi-res", "auto"],
                    value="fast",
                    label="PDF Strategy",
                    visible=False,
                )
                skip_extensions = gr.Textbox(
                    label="Skip Extensions",
                    value="",
                    info="Comma-separated, e.g. .png,.jpg",
                )
                overwrite = gr.Checkbox(label="Overwrite Existing Sources", value=False)

            with gr.Accordion("Query Settings", open=False):
                top_k = gr.Number(label="Top K", value=5, precision=0)
                max_tokens = gr.Number(label="Max Tokens", value=4096, precision=0)
                show_context = gr.Checkbox(label="Show Retrieved Context", value=False)
                stream = gr.Checkbox(label="Stream Responses", value=True)

            with gr.Accordion("App Settings", open=False):
                verbose = gr.Checkbox(label="Verbose Config Summary", value=False)
                log_level = gr.Dropdown(
                    choices=LOG_LEVELS,
                    value="WARNING",
                    label="Log Level",
                )

            with gr.Accordion("Credentials", open=False):
                mistral_api_key = gr.Textbox(label="Mistral API Key", type="password")
                anthropic_api_key = gr.Textbox(label="Anthropic API Key", type="password")
                openai_api_key = gr.Textbox(label="OpenAI API Key", type="password")
                gemini_api_key = gr.Textbox(label="Gemini API Key", type="password")
                voyage_api_key = gr.Textbox(label="Voyage API Key", type="password")
                weaviate_api_key = gr.Textbox(label="Weaviate API Key", type="password")
                aws_access_key_id = gr.Textbox(label="AWS Access Key ID", type="password")
                aws_secret_access_key = gr.Textbox(label="AWS Secret Access Key", type="password")
                aws_session_token = gr.Textbox(label="AWS Session Token", type="password")
                aws_default_region = gr.Textbox(label="AWS Region")
                google_application_credentials = gr.Textbox(
                    label="GOOGLE_APPLICATION_CREDENTIALS",
                    info="Path to a service account JSON file if you are not using gcloud ADC.",
                )

        with gr.Tab("RAG Workspace"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=7):
                    current_db = gr.Markdown(
                        _summarize_db_selection(
                            None,
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "embeddings",
                        )
                    )
                    rag_chatbot = gr.Chatbot(
                        label="Document Chat",
                        height=680,
                        type="messages",
                    )
                    rag_message = gr.Textbox(
                        label="Question",
                        placeholder="Ask a question about your ingested documents",
                    )
                    with gr.Row():
                        rag_send = gr.Button("Ask", variant="primary")
                        rag_clear = gr.Button("Clear Chat")

                with gr.Column(scale=3):
                    ingest_selection = gr.FileExplorer(
                        label="Select File or Folder to Ingest",
                        root_dir=str(browser_root),
                        file_count="single",
                        glob="**/*",
                    )
                    with gr.Row():
                        ingest_button = gr.Button("Ingest", variant="primary")
                        refresh_button = gr.Button("Refresh Sources")
                    source_table = gr.Dataframe(
                        headers=["Ingested Files"],
                        datatype=["str"],
                        value=[],
                        interactive=False,
                        row_count=(0, "dynamic"),
                        col_count=(1, "fixed"),
                        wrap=True,
                    )
                    rag_status = gr.Markdown("Ready.")
                    with gr.Accordion("Retrieved Context", open=False):
                        rag_context = gr.Markdown("Retrieved context will appear here.")

        with gr.Tab("Gemini Direct"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=7):
                    gemini_chatbot = gr.Chatbot(
                        label="Gemini File API",
                        height=680,
                        type="messages",
                    )
                    gemini_message = gr.Textbox(
                        label="Question",
                        placeholder="Ask Gemini directly about the selected files",
                    )
                    with gr.Row():
                        gemini_send = gr.Button("Ask Gemini", variant="primary")
                        gemini_clear = gr.Button("Clear Chat")
                with gr.Column(scale=3):
                    gemini_selection = gr.FileExplorer(
                        label="Select File or Folder",
                        root_dir=str(browser_root),
                        file_count="single",
                        glob="**/*",
                    )
                    gemini_model = gr.Textbox(
                        label="Gemini Model",
                        value="gemini-2.5-flash",
                    )
                    gemini_max_tokens = gr.Number(
                        label="Max Tokens",
                        value=1024,
                        precision=0,
                    )
                    gemini_status = gr.Markdown(
                        "Uploads selected files to Gemini and asks without persisting anything."
                    )

        provider_inputs = [db_provider, embed_choice]
        provider_outputs = [
            lancedb_path,
            postgres_uri,
            weaviate_uri,
            vertex_project,
            vertex_corpus,
            bedrock_kb_id,
            bedrock_data_source_id,
            table,
            embed_custom_provider,
            embed_custom_model,
        ]
        db_provider.change(
            update_provider_controls,
            inputs=provider_inputs,
            outputs=provider_outputs,
        )
        embed_choice.change(
            update_provider_controls,
            inputs=provider_inputs,
            outputs=provider_outputs,
        )
        chunk_method.change(update_pdf_strategy, inputs=chunk_method, outputs=pdf_strategy)
        llm_choice.change(
            update_llm_custom,
            inputs=llm_choice,
            outputs=[llm_custom_provider, llm_custom_model],
        )
        embed_choice.change(
            update_embed_custom,
            inputs=[embed_choice, db_provider],
            outputs=[embed_custom_provider, embed_custom_model],
        )

        db_summary_inputs = [
            db_provider,
            lancedb_path,
            postgres_uri,
            weaviate_uri,
            vertex_project,
            vertex_corpus,
            bedrock_kb_id,
            bedrock_data_source_id,
            table,
        ]
        for component in [
            db_provider,
            lancedb_path,
            postgres_uri,
            weaviate_uri,
            vertex_project,
            vertex_corpus,
            bedrock_kb_id,
            bedrock_data_source_id,
            table,
        ]:
            component.change(
                update_db_summary,
                inputs=db_summary_inputs,
                outputs=current_db,
            )

        source_inputs = [
            db_provider,
            table,
            profile,
            lancedb_path,
            postgres_uri,
            weaviate_uri,
            vertex_project,
            vertex_corpus,
            bedrock_kb_id,
            bedrock_data_source_id,
            log_level,
        ]
        refresh_button.click(
            refresh_sources,
            inputs=source_inputs,
            outputs=[source_table, rag_status, current_db],
        )

        shared_inputs = [
            db_provider,
            table,
            profile,
            lancedb_path,
            postgres_uri,
            weaviate_uri,
            vertex_project,
            vertex_corpus,
            bedrock_kb_id,
            bedrock_data_source_id,
            llm_choice,
            llm_custom_provider,
            llm_custom_model,
            embed_choice,
            embed_custom_provider,
            embed_custom_model,
            chunk_method,
            chunk_size,
            chunk_overlap,
            pdf_strategy,
            top_k,
            max_tokens,
        ]
        credential_inputs = [
            mistral_api_key,
            anthropic_api_key,
            openai_api_key,
            gemini_api_key,
            voyage_api_key,
            weaviate_api_key,
            aws_access_key_id,
            aws_secret_access_key,
            aws_session_token,
            aws_default_region,
            google_application_credentials,
        ]

        ingest_button.click(
            ingest_files,
            inputs=[
                ingest_selection,
                *shared_inputs,
                skip_extensions,
                overwrite,
                verbose,
                log_level,
                *credential_inputs,
            ],
            outputs=[source_table, rag_status, current_db],
        )

        rag_submit_inputs = [
            rag_message,
            rag_history,
            *shared_inputs,
            show_context,
            stream,
            verbose,
            log_level,
            *credential_inputs,
        ]
        rag_submit_outputs = [
            rag_message,
            rag_chatbot,
            rag_history,
            rag_status,
            rag_context,
        ]
        rag_send.click(chat_with_rag, inputs=rag_submit_inputs, outputs=rag_submit_outputs)
        rag_message.submit(chat_with_rag, inputs=rag_submit_inputs, outputs=rag_submit_outputs)
        rag_clear.click(clear_chat, outputs=[rag_chatbot, rag_history, rag_status, rag_context])

        gemini_submit_inputs = [
            gemini_message,
            gemini_history,
            gemini_selection,
            gemini_model,
            gemini_max_tokens,
            gemini_api_key,
        ]
        gemini_submit_outputs = [
            gemini_message,
            gemini_chatbot,
            gemini_history,
            gemini_status,
        ]
        gemini_send.click(
            gemini_answer,
            inputs=gemini_submit_inputs,
            outputs=gemini_submit_outputs,
        )
        gemini_message.submit(
            gemini_answer,
            inputs=gemini_submit_inputs,
            outputs=gemini_submit_outputs,
        )
        gemini_clear.click(
            lambda: ([], [], "Chat cleared."),
            outputs=[gemini_chatbot, gemini_history, gemini_status],
        )

    return demo


def launch(
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
    inbrowser: bool = True,
    root_dir: str | Path | None = None,
) -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    demo = create_demo(root_dir=root_dir)
    demo.queue()
    demo.launch(server_name=host, server_port=port, inbrowser=inbrowser)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch the RAG Lette Gradio GUI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--root-dir", default=str(Path.cwd()))
    args = parser.parse_args(argv)
    launch(
        host=args.host,
        port=args.port,
        inbrowser=not args.no_browser,
        root_dir=args.root_dir,
    )
