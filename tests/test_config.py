from __future__ import annotations

from pathlib import Path

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
