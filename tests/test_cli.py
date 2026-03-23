from __future__ import annotations

import logging
from pathlib import Path

from click.testing import CliRunner

from rag import cli
from rag.cli import main
from rag.db import QueryChunk
from rag.services import RetrievalResult


def test_cli_root_help_shows_global_log_option() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "--log LEVEL" in result.output
    assert "--log-level" not in result.output


def test_cli_global_log_option_configures_logging_from_group(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_basic_config(*, level, format, force):
        captured["level"] = level
        captured["format"] = format
        captured["force"] = force

    monkeypatch.setattr(cli.logging, "basicConfig", fake_basic_config)
    monkeypatch.setattr(
        "rag.db.get_db_adapter",
        lambda db, table: type("Adapter", (), {"list_sources": lambda self: []})(),
    )

    result = CliRunner().invoke(main, ["--log", "info", "list", "./db"])

    assert result.exit_code == 0
    assert captured == {
        "level": logging.INFO,
        "format": "%(name)s: %(message)s",
        "force": True,
    }


def test_cli_global_log_option_configures_logging_from_subcommand_position(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_basic_config(*, level, format, force):
        captured["level"] = level
        captured["format"] = format
        captured["force"] = force

    def fake_generate(*, query, context, provider, model, max_tokens):
        return "final answer"

    monkeypatch.setattr(cli.logging, "basicConfig", fake_basic_config)
    monkeypatch.setattr(
        "rag.cli.query_service.retrieve",
        lambda cfg, question, *, progress=None: RetrievalResult([QueryChunk(text="ctx", source="doc.txt")]),
    )
    monkeypatch.setattr("rag.cli.generate", fake_generate)

    result = CliRunner().invoke(main, ["ask", "./db", "Explain AFM", "--no-stream", "--log", "info"])

    assert result.exit_code == 0
    assert captured == {
        "level": logging.INFO,
        "format": "%(name)s: %(message)s",
        "force": True,
    }


def test_cli_bare_question_routes_to_default_db_and_infers_embed(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_retrieve(cfg, question, *, progress=None):
        captured["db"] = cfg.db
        captured["embed_provider"] = cfg.embed_provider
        captured["llm_provider"] = cfg.llm_provider
        captured["question"] = question
        return RetrievalResult(
            [
                QueryChunk(text="retrieved context", source="alpha.txt"),
                QueryChunk(text="retrieved context two", source="alpha.txt"),
                QueryChunk(text="retrieved context three", source="beta.md"),
            ]
        )

    def fake_stream(*, query, context, provider, model, max_tokens):
        captured["stream_provider"] = provider
        captured["stream_model"] = model
        assert query == "What is AFM?"
        assert context == [
            QueryChunk(text="retrieved context", source="alpha.txt"),
            QueryChunk(text="retrieved context two", source="alpha.txt"),
            QueryChunk(text="retrieved context three", source="beta.md"),
        ]
        assert max_tokens == 4096
        yield "streamed answer"

    monkeypatch.setattr("rag.cli.query_service.retrieve", fake_retrieve)
    monkeypatch.setattr("rag.cli.generate_stream", fake_stream)

    result = CliRunner().invoke(main, ["What is AFM?"])

    assert result.exit_code == 0
    assert "streamed answer" in result.output
    assert "alpha.txt: 2" in result.output
    assert "beta.md: 1" in result.output
    assert "Total source chunks: 3" in result.output
    assert captured == {
        "db": "./db",
        "embed_provider": "mistral",
        "llm_provider": "mistral",
        "question": "What is AFM?",
        "stream_provider": "mistral",
        "stream_model": "ministral-3b-2512",
    }


def test_cli_quiet_suppresses_config_summary(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("notes", encoding="utf-8")

    def fake_ingest(*args, **kwargs):
        class Result:
            skipped_files = []
            ignored_files = []
            failures = []
            rows_written = 1
            mode = "vector"
            chunks = []
            db_existed = False

        return Result()

    monkeypatch.setattr("rag.cli.ingestion_service.ingest", fake_ingest)

    result = CliRunner().invoke(main, ["-q", "ingest", "./db", str(source)])

    assert result.exit_code == 0
    assert "config  llm=" not in result.output


def test_cli_ingest_verbose_shows_only_ingest_relevant_config(
    monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("notes", encoding="utf-8")

    def fake_ingest(*args, **kwargs):
        class Result:
            skipped_files = []
            ignored_files = []
            failures = []
            rows_written = 1
            mode = "vector"
            chunks = []
            db_existed = False

        return Result()

    monkeypatch.setattr("rag.cli.ingestion_service.ingest", fake_ingest)

    result = CliRunner().invoke(
        main,
        [
            "ingest",
            "./db",
            str(source),
            "--embed-model",
            "voyage-3.5-lite",
        ],
    )

    assert result.exit_code == 0
    assert "config" in result.output
    assert "embed=voyageai/voyage-3.5-lite" in result.output
    assert "chunk=basic" in result.output
    assert "chunk_size=1000" in result.output
    assert "chunk_overlap=200" in result.output
    assert "llm=" not in result.output
    assert "top_k=" not in result.output
    assert "max_tokens=" not in result.output


def test_cli_ingest_shows_unstructured_hint_for_unsupported_files(
    monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "slides.pptx"
    source.write_text("slides", encoding="utf-8")

    def fake_ingest(*args, **kwargs):
        raise RuntimeError("Unsupported file type: '.pptx' (slides.pptx)")

    monkeypatch.setattr("rag.cli.ingestion_service.ingest", fake_ingest)

    result = CliRunner().invoke(main, ["ingest", "./db", str(source)])

    assert result.exit_code != 0
    assert "use --chunk unstructured" in result.output


def test_cli_ingest_reports_ignored_directory_files(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "docs"
    source.mkdir()

    def fake_ingest(*args, **kwargs):
        class Result:
            skipped_files = []
            ignored_files = [
                type("Ignored", (), {"path": source / "slides.pptx", "reason": "unsupported file type (.pptx)"}),
                type("Ignored", (), {"path": source / ".hidden.txt", "reason": "hidden file"}),
            ]
            failures = []
            rows_written = 1
            mode = "vector"
            chunks = []
            db_existed = False

        return Result()

    monkeypatch.setattr("rag.cli.ingestion_service.ingest", fake_ingest)

    result = CliRunner().invoke(main, ["ingest", "./db", str(source)])

    assert result.exit_code == 0
    assert "skipped slides.pptx: unsupported file type (.pptx)" in result.output
    assert "use --chunk unstructured" in result.output
    assert "skipped .hidden.txt: hidden file" in result.output


def test_cli_ask_uses_explicit_embed_and_non_stream_generation(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_retrieve(cfg, question, *, progress=None):
        captured["embed_provider"] = cfg.embed_provider
        captured["embed_model"] = cfg.embed_model
        captured["llm_provider"] = cfg.llm_provider
        captured["llm_model"] = cfg.llm_model
        captured["question"] = question
        return RetrievalResult([QueryChunk(text="ctx", source="doc.txt")])

    def fake_generate(*, query, context, provider, model, max_tokens):
        captured["provider"] = provider
        captured["model"] = model
        assert query == "Explain AFM"
        assert context == [QueryChunk(text="ctx", source="doc.txt")]
        assert max_tokens == 512
        return "final answer"

    monkeypatch.setattr("rag.cli.query_service.retrieve", fake_retrieve)
    monkeypatch.setattr("rag.cli.generate", fake_generate)

    result = CliRunner().invoke(
        main,
        [
            "ask",
            "./db",
            "Explain AFM",
            "--embed-provider",
            "openai",
            "--embed-model",
            "text-embedding-3-small",
            "--llm-provider",
            "anthropic",
            "--llm-model",
            "claude-sonnet-4-5",
            "--max-tokens",
            "512",
            "--no-stream",
        ],
    )

    assert result.exit_code == 0
    assert "final answer" in result.output
    assert "doc.txt: 1" in result.output
    assert "Total source chunks: 1" in result.output
    assert captured == {
        "embed_provider": "openai",
        "embed_model": "text-embedding-3-small",
        "llm_provider": "anthropic",
        "llm_model": "claude-sonnet-4-5",
        "question": "Explain AFM",
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
    }


def test_cli_ask_infers_provider_from_model_only(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_retrieve(cfg, question, *, progress=None):
        captured["llm_provider"] = cfg.llm_provider
        captured["llm_model"] = cfg.llm_model
        return RetrievalResult([QueryChunk(text="ctx", source="doc.txt")])

    def fake_generate(*, query, context, provider, model, max_tokens):
        captured["provider"] = provider
        captured["model"] = model
        return "final answer"

    monkeypatch.setattr("rag.cli.query_service.retrieve", fake_retrieve)
    monkeypatch.setattr("rag.cli.generate", fake_generate)

    result = CliRunner().invoke(
        main,
        [
            "ask",
            "./db",
            "Explain AFM",
            "--llm-model",
            "claude-sonnet-4-5",
            "--no-stream",
        ],
    )

    assert result.exit_code == 0
    assert "final answer" in result.output
    assert captured["llm_provider"] == "anthropic"
    assert captured["llm_model"] == "claude-sonnet-4-5"
    assert captured["provider"] == "anthropic"
    assert captured["model"] == "claude-sonnet-4-5"


def test_cli_ask_infers_provider_from_alias_model_only(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_retrieve(cfg, question, *, progress=None):
        captured["llm_provider"] = cfg.llm_provider
        captured["llm_model"] = cfg.llm_model
        return RetrievalResult([QueryChunk(text="ctx", source="doc.txt")])

    def fake_generate(*, query, context, provider, model, max_tokens):
        captured["provider"] = provider
        captured["model"] = model
        return "final answer"

    monkeypatch.setattr("rag.cli.query_service.retrieve", fake_retrieve)
    monkeypatch.setattr("rag.cli.generate", fake_generate)

    result = CliRunner().invoke(
        main,
        [
            "ask",
            "./db",
            "Explain AFM",
            "--llm-model",
            "gpt-5.4-mini",
            "--no-stream",
        ],
    )

    assert result.exit_code == 0
    assert "final answer" in result.output
    assert captured["llm_provider"] == "openai"
    assert captured["llm_model"] == "gpt-5-mini-2025-08-07"
    assert captured["provider"] == "openai"
    assert captured["model"] == "gpt-5-mini-2025-08-07"


def test_cli_ask_verbose_shows_retrieval_and_generation_config(monkeypatch) -> None:
    def fake_retrieve(cfg, question, *, progress=None):
        return RetrievalResult([QueryChunk(text="ctx", source="doc.txt")])

    def fake_generate(*, query, context, provider, model, max_tokens):
        return "final answer"

    monkeypatch.setattr("rag.cli.query_service.retrieve", fake_retrieve)
    monkeypatch.setattr("rag.cli.generate", fake_generate)

    result = CliRunner().invoke(
        main,
        [
            "ask",
            "./db",
            "Explain AFM",
            "--embed-provider",
            "openai",
            "--embed-model",
            "text-embedding-3-small",
            "--llm-provider",
            "anthropic",
            "--llm-model",
            "claude-sonnet-4-5",
            "--top-k",
            "7",
            "--max-tokens",
            "512",
            "--no-stream",
        ],
    )

    assert result.exit_code == 0
    assert "config" in result.output
    assert "embed=openai/text-embedding-3-small" in result.output
    assert "llm=anthropic/claude-sonnet-4-5" in result.output
    assert "top_k=7" in result.output
    assert "max_tokens=512" in result.output


def test_cli_ask_verbose_omits_embed_for_managed_backends(monkeypatch) -> None:
    def fake_retrieve(cfg, question, *, progress=None):
        return RetrievalResult([QueryChunk(text="ctx", source="doc.txt")])

    def fake_generate(*, query, context, provider, model, max_tokens):
        return "final answer"

    monkeypatch.setattr("rag.cli.query_service.retrieve", fake_retrieve)
    monkeypatch.setattr("rag.cli.generate", fake_generate)

    result = CliRunner().invoke(
        main,
        [
            "ask",
            "vertex://project/corpus",
            "Explain AFM",
            "--llm-provider",
            "google",
            "--llm-model",
            "gemini-2.5-flash",
            "--top-k",
            "3",
            "--max-tokens",
            "256",
            "--no-stream",
        ],
    )

    assert result.exit_code == 0
    assert "config" in result.output
    assert "embed=" not in result.output
    assert "llm=google/gemini-2.5-flash" in result.output
    assert "top_k=3" in result.output
    assert "max_tokens=256" in result.output


def test_cli_ask_omits_sources_when_retrieval_has_no_filenames(monkeypatch) -> None:
    def fake_retrieve(cfg, question, *, progress=None):
        return RetrievalResult([QueryChunk(text="ctx without source")])

    def fake_generate(*, query, context, provider, model, max_tokens):
        return "answer without sources"

    monkeypatch.setattr("rag.cli.query_service.retrieve", fake_retrieve)
    monkeypatch.setattr("rag.cli.generate", fake_generate)

    result = CliRunner().invoke(main, ["ask", "./db", "Explain AFM", "--no-stream"])

    assert result.exit_code == 0
    assert "answer without sources" in result.output
    assert "Sources" not in result.output


def test_cli_ask_list_files_shortcut_prints_sources_without_retrieval(monkeypatch) -> None:
    calls: list[str] = []

    class FakeAdapter:
        def list_sources(self) -> list[str]:
            calls.append("list_sources")
            return ["alpha.txt", "beta.md"]

    def fail_retrieve(*args, **kwargs):
        raise AssertionError("retrieve should not be called for list files")

    def fail_generate(*args, **kwargs):
        raise AssertionError("generate should not be called for list files")

    monkeypatch.setattr("rag.cli.get_db_adapter", lambda db, table: FakeAdapter())
    monkeypatch.setattr("rag.cli.query_service.retrieve", fail_retrieve)
    monkeypatch.setattr("rag.cli.generate", fail_generate)

    result = CliRunner().invoke(main, ["ask", "./db", "list files", "--no-stream"])

    assert result.exit_code == 0
    assert "alpha.txt" in result.output
    assert "beta.md" in result.output
    assert calls == ["list_sources"]


def test_cli_ask_list_files_shortcut_is_casefolded_and_trimmed(monkeypatch) -> None:
    class FakeAdapter:
        def list_sources(self) -> list[str]:
            return ["alpha.txt"]

    monkeypatch.setattr("rag.cli.get_db_adapter", lambda db, table: FakeAdapter())

    result = CliRunner().invoke(main, ["ask", "./db", "  LIST FILES  ", "--no-stream"])

    assert result.exit_code == 0
    assert "alpha.txt" in result.output


def test_cli_ask_list_files_shortcut_prints_empty_state(monkeypatch) -> None:
    class FakeAdapter:
        def list_sources(self) -> list[str]:
            return []

    monkeypatch.setattr("rag.cli.get_db_adapter", lambda db, table: FakeAdapter())

    result = CliRunner().invoke(main, ["ask", "./db", "list files", "--no-stream"])

    assert result.exit_code == 0
    assert "No files ingested yet." in result.output


def test_cli_ask_near_miss_list_files_query_uses_normal_rag_flow(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_retrieve(cfg, question, *, progress=None):
        captured["question"] = question
        return RetrievalResult([QueryChunk(text="ctx", source="doc.txt")])

    def fake_generate(*, query, context, provider, model, max_tokens):
        captured["generated"] = query
        return "final answer"

    monkeypatch.setattr("rag.cli.query_service.retrieve", fake_retrieve)
    monkeypatch.setattr("rag.cli.generate", fake_generate)

    result = CliRunner().invoke(main, ["ask", "./db", "list documents", "--no-stream"])

    assert result.exit_code == 0
    assert "final answer" in result.output
    assert captured == {
        "question": "list documents",
        "generated": "list documents",
    }
