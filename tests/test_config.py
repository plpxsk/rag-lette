from __future__ import annotations

from pathlib import Path

import pytest

from rag.config import load_config


def test_load_config_merges_global_local_profile_overrides_and_env(
    monkeypatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    (home / ".rag.toml").write_text(
        """
[defaults]
db = "global-db"
table = "global_table"
embed = "voyage"
chunk_size = 800

[profiles.work]
top_k = 4
llm = "mistral-large"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (workdir / "rag.toml").write_text(
        """
[defaults]
table = "local_table"
chunk_size = 900

[profiles.work]
chunk_overlap = 75
embed = "openai"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("RAG_TABLE", "env_table")
    monkeypatch.setenv("RAG_LLM", "claude")
    monkeypatch.setenv("RAG_MAX_TOKENS", "2048")

    cfg = load_config(
        profile="work",
        overrides={"db": "override-db", "embed": "openai/text-embedding-3-large"},
    )

    assert cfg.db == "override-db"
    assert cfg.table == "env_table"
    assert cfg.embed_provider == "openai"
    assert cfg.embed_model == "text-embedding-3-large"
    assert cfg.llm_provider == "anthropic"
    assert cfg.llm_model == "claude-haiku-4-5"
    assert cfg.top_k == 4
    assert cfg.max_tokens == 2048
    assert cfg.chunk_size == 900
    assert cfg.chunk_overlap == 75


def test_load_config_keeps_custom_provider_model_strings(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    cfg = load_config(
        overrides={
            "embed": "openai/text-embedding-3-large",
            "llm": "mistral/open-mistral-nemo",
        }
    )

    assert cfg.embed_provider == "openai"
    assert cfg.embed_model == "text-embedding-3-large"
    assert cfg.llm_provider == "mistral"
    assert cfg.llm_model == "open-mistral-nemo"


@pytest.mark.parametrize(
    ("llm_value", "expected_provider", "expected_model"),
    [
        ("claude-sonnet-4-5", "anthropic", "claude-sonnet-4-5"),
        ("gpt-5.4", "openai", "gpt-5.4"),
        ("gpt-5.4-mini", "openai", "gpt-5-mini-2025-08-07"),
        ("gpt-5.4-nano", "openai", "gpt-5-nano-2025-08-07"),
    ],
)
def test_load_config_accepts_new_llm_aliases(
    monkeypatch,
    tmp_path: Path,
    llm_value: str,
    expected_provider: str,
    expected_model: str,
) -> None:
    monkeypatch.chdir(tmp_path)

    cfg = load_config(overrides={"llm": llm_value})

    assert cfg.llm_provider == expected_provider
    assert cfg.llm_model == expected_model


def test_load_config_accepts_split_provider_and_model_fields(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    cfg = load_config(
        overrides={
            "embed_provider": "openai",
            "embed_model": "text-embedding-3-large",
            "llm_provider": "anthropic",
            "llm_model": "claude-sonnet-4-5",
        }
    )

    assert cfg.embed_provider == "openai"
    assert cfg.embed_model == "text-embedding-3-large"
    assert cfg.llm_provider == "anthropic"
    assert cfg.llm_model == "claude-sonnet-4-5"


def test_load_config_infers_provider_from_model_only(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    llm_cfg = load_config(overrides={"llm_model": "claude-sonnet-4-5"})
    embed_cfg = load_config(overrides={"embed_model": "text-embedding-3-small"})
    llm_alias_cfg = load_config(overrides={"llm_model": "gpt-5.4-mini"})
    embed_alias_cfg = load_config(overrides={"embed_model": "voyage"})

    assert llm_cfg.llm_provider == "anthropic"
    assert llm_cfg.llm_model == "claude-sonnet-4-5"
    assert embed_cfg.embed_provider == "openai"
    assert embed_cfg.embed_model == "text-embedding-3-small"
    assert llm_alias_cfg.llm_provider == "openai"
    assert llm_alias_cfg.llm_model == "gpt-5-mini-2025-08-07"
    assert embed_alias_cfg.embed_provider == "voyageai"
    assert embed_alias_cfg.embed_model == "voyage-3.5-lite"
