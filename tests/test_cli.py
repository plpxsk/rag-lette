from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from rag.cli import main


def test_cli_bare_question_routes_to_default_db_and_infers_embed(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_retrieve(cfg, question, *, progress=None):
        captured["db"] = cfg.db
        captured["embed_provider"] = cfg.embed_provider
        captured["llm_provider"] = cfg.llm_provider
        captured["question"] = question
        return ["retrieved context"]

    def fake_stream(*, query, context, provider, model, max_tokens):
        captured["stream_provider"] = provider
        captured["stream_model"] = model
        assert query == "What is AFM?"
        assert context == ["retrieved context"]
        assert max_tokens == 4096
        yield "streamed answer"

    monkeypatch.setattr("rag.cli.query_service.retrieve", fake_retrieve)
    monkeypatch.setattr("rag.cli.generate_stream", fake_stream)

    result = CliRunner().invoke(main, ["What is AFM?"])

    assert result.exit_code == 0
    assert "streamed answer" in result.output
    assert captured == {
        "db": "./db",
        "embed_provider": "mistral",
        "llm_provider": "mistral",
        "question": "What is AFM?",
        "stream_provider": "mistral",
        "stream_model": "ministral-3b-2512",
    }


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


def test_cli_ask_uses_explicit_embed_and_non_stream_generation(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_retrieve(cfg, question, *, progress=None):
        captured["embed_provider"] = cfg.embed_provider
        captured["embed_model"] = cfg.embed_model
        captured["llm_provider"] = cfg.llm_provider
        captured["llm_model"] = cfg.llm_model
        captured["question"] = question
        return ["ctx"]

    def fake_generate(*, query, context, provider, model, max_tokens):
        captured["provider"] = provider
        captured["model"] = model
        assert query == "Explain AFM"
        assert context == ["ctx"]
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
            "--embed",
            "openai",
            "--llm",
            "claude",
            "--max-tokens",
            "512",
            "--no-stream",
        ],
    )

    assert result.exit_code == 0
    assert "final answer" in result.output
    assert captured == {
        "embed_provider": "openai",
        "embed_model": "text-embedding-3-small",
        "llm_provider": "anthropic",
        "llm_model": "claude-haiku-4-5",
        "question": "Explain AFM",
        "provider": "anthropic",
        "model": "claude-haiku-4-5",
    }
