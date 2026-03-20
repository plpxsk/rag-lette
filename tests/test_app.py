from __future__ import annotations

import pytest

from rag.app import _build_db_uri, _infer_embed_from_llm, _resolve_model_choice
from rag.config import EMBED_ALIASES, LLM_ALIASES


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


def test_resolve_model_choice_returns_alias_when_known() -> None:
    assert _resolve_model_choice("mistral", "", "", LLM_ALIASES) == "mistral"


def test_resolve_model_choice_builds_custom_provider_and_model() -> None:
    assert (
        _resolve_model_choice("custom", "openai", "gpt-4.1", LLM_ALIASES)
        == "openai/gpt-4.1"
    )


def test_resolve_model_choice_requires_custom_parts() -> None:
    with pytest.raises(ValueError, match="both required"):
        _resolve_model_choice("custom", "openai", "", EMBED_ALIASES)


def test_infer_embed_from_llm_matches_provider() -> None:
    assert _infer_embed_from_llm("openai") == "openai"
