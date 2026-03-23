from __future__ import annotations

from pathlib import Path

import pytest

from rag.app import (
    _add_ingest_targets,
    _build_db_uri,
    _format_context,
    _format_source_footer,
    _infer_embed_from_llm,
    _merge_ingestion_results,
    _resolve_provider_model_input,
    _resolve_selected_paths,
)
from rag.config import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER, EMBED_ALIASES, LLM_ALIASES
from rag.db import QueryChunk
from rag.services import IngestionResult, RetrievalResult


@pytest.mark.parametrize(
    ("provider", "kwargs", "expected"),
    [
        (
            "lancedb",
            {
                "lancedb_path": "./db",
                "postgres_uri": "",
                "weaviate_uri": "",
                "vertex_project": "",
                "vertex_corpus": "",
                "bedrock_kb_id": "",
                "bedrock_data_source_id": "",
            },
            "./db",
        ),
        (
            "postgres",
            {
                "lancedb_path": "",
                "postgres_uri": "postgres://user:pass@localhost:5432/ragdb",
                "weaviate_uri": "",
                "vertex_project": "",
                "vertex_corpus": "",
                "bedrock_kb_id": "",
                "bedrock_data_source_id": "",
            },
            "postgres://user:pass@localhost:5432/ragdb",
        ),
        (
            "vertex",
            {
                "lancedb_path": "",
                "postgres_uri": "",
                "weaviate_uri": "",
                "vertex_project": "proj-123",
                "vertex_corpus": "docs",
                "bedrock_kb_id": "",
                "bedrock_data_source_id": "",
            },
            "vertex://proj-123/docs",
        ),
        (
            "bedrock-kb",
            {
                "lancedb_path": "",
                "postgres_uri": "",
                "weaviate_uri": "",
                "vertex_project": "",
                "vertex_corpus": "",
                "bedrock_kb_id": "kb-123",
                "bedrock_data_source_id": "ds-456",
            },
            "bedrock-kb://kb-123/ds-456",
        ),
    ],
)
def test_build_db_uri(provider: str, kwargs: dict[str, str], expected: str) -> None:
    assert _build_db_uri(provider, **kwargs) == expected


def test_build_db_uri_requires_provider_selection() -> None:
    with pytest.raises(ValueError, match="Choose a database provider first"):
        _build_db_uri(None, "", "", "", "", "", "", "")


def test_build_db_uri_requires_lancedb_path() -> None:
    with pytest.raises(ValueError, match="Enter a LanceDB path or URI"):
        _build_db_uri("lancedb", "", "", "", "", "", "", "")


def test_resolve_provider_model_input_keeps_explicit_values() -> None:
    provider, model = _resolve_provider_model_input(
        "openai",
        "gpt-4o",
        LLM_ALIASES,
        DEFAULT_LLM_PROVIDER,
        DEFAULT_LLM_MODEL,
        label="LLM",
    )

    assert provider == "openai"
    assert model == "gpt-4o"


def test_resolve_provider_model_input_infers_provider_from_model() -> None:
    provider, model = _resolve_provider_model_input(
        "",
        "claude-sonnet-4-5",
        LLM_ALIASES,
        DEFAULT_LLM_PROVIDER,
        DEFAULT_LLM_MODEL,
        label="LLM",
    )

    assert provider == "anthropic"
    assert model == "claude-sonnet-4-5"


def test_resolve_provider_model_input_infers_provider_from_alias_model() -> None:
    provider, model = _resolve_provider_model_input(
        "",
        "gpt-5.4-mini",
        LLM_ALIASES,
        DEFAULT_LLM_PROVIDER,
        DEFAULT_LLM_MODEL,
        label="LLM",
    )

    assert provider == "openai"
    assert model == "gpt-5-mini-2025-08-07"


def test_resolve_provider_model_input_parses_provider_model_string() -> None:
    provider, model = _resolve_provider_model_input(
        "",
        "openai/gpt-4o",
        LLM_ALIASES,
        DEFAULT_LLM_PROVIDER,
        DEFAULT_LLM_MODEL,
        label="LLM",
    )

    assert provider == "openai"
    assert model == "gpt-4o"


def test_resolve_provider_model_input_defaults_when_blank() -> None:
    provider, model = _resolve_provider_model_input(
        "",
        "",
        LLM_ALIASES,
        DEFAULT_LLM_PROVIDER,
        DEFAULT_LLM_MODEL,
        label="LLM",
    )

    assert provider == DEFAULT_LLM_PROVIDER
    assert model == DEFAULT_LLM_MODEL


def test_resolve_provider_model_input_requires_model_for_unknown_provider() -> None:
    with pytest.raises(ValueError, match="model is required"):
        _resolve_provider_model_input(
            "custom-provider",
            "",
            EMBED_ALIASES,
            DEFAULT_LLM_PROVIDER,
            DEFAULT_LLM_MODEL,
            label="Embedding",
        )


def test_infer_embed_from_llm_matches_provider() -> None:
    assert _infer_embed_from_llm("openai") == "openai"


def test_resolve_selected_paths_supports_multiple_entries(tmp_path: Path) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("a", encoding="utf-8")
    second.write_text("b", encoding="utf-8")

    resolved = _resolve_selected_paths([str(first), str(second)], str(tmp_path))

    assert resolved == [first.resolve(), second.resolve()]


def test_merge_ingestion_results_sums_rows_and_keeps_skips() -> None:
    merged = _merge_ingestion_results(
        [
            IngestionResult("vector", False, [], 2, ["one.txt"], [], []),
            IngestionResult("vector", True, [], 3, ["two.txt"], [], []),
        ]
    )

    assert merged.rows_written == 5
    assert merged.db_existed is True
    assert merged.skipped_files == ["one.txt", "two.txt"]


def test_add_ingest_targets_appends_new_paths_once(tmp_path: Path) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("a", encoding="utf-8")
    second.write_text("b", encoding="utf-8")

    targets, rows, message = _add_ingest_targets(str(first), [str(first)], str(tmp_path))
    assert targets == [str(first.resolve())]
    assert rows == [[str(first.resolve())]]
    assert "already queued" in message

    targets, rows, message = _add_ingest_targets(str(second), targets, str(tmp_path))
    assert targets == [str(first.resolve()), str(second.resolve())]
    assert rows == [[str(first.resolve())], [str(second.resolve())]]
    assert "Queued 1 item" in message


def test_format_context_includes_source_labels_when_present() -> None:
    result = RetrievalResult(
        [QueryChunk(text="Chunk body", source="report.pdf"), QueryChunk(text="Other body")]
    )

    rendered = _format_context(result, show_context=True)

    assert "`report.pdf`" in rendered
    assert "Chunk body" in rendered
    assert "Other body" in rendered


def test_format_source_footer_omits_missing_sources() -> None:
    assert _format_source_footer(RetrievalResult([QueryChunk(text="Chunk body")])) == ""


def test_format_source_footer_includes_total_chunk_count() -> None:
    footer = _format_source_footer(
        RetrievalResult(
            [
                QueryChunk(text="Chunk A", source="report.pdf"),
                QueryChunk(text="Chunk B", source="report.pdf"),
                QueryChunk(text="Chunk C", source="notes.md"),
            ]
        )
    )

    assert "Total source chunks: 3" in footer
