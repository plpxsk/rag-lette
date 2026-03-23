"""CLI entry point: `rag ingest` and `rag ask`."""
from __future__ import annotations

import difflib
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule

from rag.config import EMBED_ALIASES, LLM_ALIASES, load_config
from rag.db import QueryChunk
from rag.gemini import GeminiFileApiClient
from rag.generate import generate, generate_stream
from rag.services import IngestionService, QueryService, RetrievalResult

console = Console()
ingestion_service = IngestionService()
query_service = QueryService()
_source_executor = ThreadPoolExecutor(max_workers=1)


def _infer_embed_from_llm(llm_alias: str) -> str:
    """Pick an embed alias whose provider matches the LLM provider, or fall back to mistral."""
    base = llm_alias.split("/")[0]
    llm_provider = LLM_ALIASES.get(base, ("mistral", ""))[0]
    for embed_alias, (embed_provider, _) in EMBED_ALIASES.items():
        if embed_provider == llm_provider:
            return embed_alias
    return "mistral"


def _fuzzy_option(value: str, known: list[str], label: str) -> str:
    """Validate value against known aliases; suggest close matches on typo."""
    if value in known:
        return value
    provider = value.split("/")[0]
    if provider in known:
        return value
    matches = difflib.get_close_matches(provider, known, n=1, cutoff=0.6)
    hint = f" Did you mean '{matches[0]}'?" if matches else f" Valid options: {', '.join(known)}."
    raise click.BadParameter(f"'{value}' is not a known {label}.{hint}", param_hint=f"--{label}")


def _service_progress(event: str, stage: str) -> None:
    start_messages = {
        "chunking": "Chunking documents...",
        "embedding": "Embedding chunks...",
        "writing": "Writing to database...",
        "uploading": "Uploading files...",
        "indexing": "Waiting for indexing to complete...",
        "searching": "Searching...",
    }
    end_messages = {
        "chunking": "Chunking complete.",
        "embedding": "Embedding complete.",
        "writing": "Write complete.",
        "uploading": "Upload complete.",
        "indexing": "Indexing complete.",
        "searching": "Search complete.",
    }
    if event == "start":
        console.print(f"[dim]{start_messages.get(stage, stage)}[/dim]")
    elif event == "end":
        console.print(f"[dim]{end_messages.get(stage, stage)}[/dim]")


def _format_context_chunk(index: int, chunk: QueryChunk) -> str:
    label = f"[{chunk.source}] " if chunk.source else ""
    return f"[dim][{index}][/dim] {label}{chunk.text}\n"


def _format_source_summary(result: RetrievalResult) -> str:
    if not result.has_sources:
        return ""
    lines = ["Sources"]
    lines.extend(f"- {source}: {count}" for source, count in result.source_counts)
    lines.append(f"Total source chunks: {sum(count for _, count in result.source_counts)}")
    return "\n".join(lines)


def _start_source_summary(result: RetrievalResult) -> Future[str]:
    return _source_executor.submit(_format_source_summary, result)


class SmartGroup(click.Group):
    """Click group that routes bare questions to `ask ./db <question>`."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        # Shorthand: bare questions treated as `ask ./db <question>`
        if args and not args[0].startswith("-") and args[0] not in self.commands:
            args = ["ask", "./db"] + args
        return super().parse_args(ctx, args)

    def invoke(self, ctx: click.Context) -> object:
        try:
            return super().invoke(ctx)
        except NotImplementedError as exc:
            raise click.ClickException(str(exc))
        except RuntimeError as exc:
            raise click.ClickException(str(exc))


@click.group(cls=SmartGroup)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress the resolved config summary before running.",
)
@click.option(
    "--log-level",
    default="WARNING",
    show_default=True,
    metavar="LEVEL",
    help="Python log level (DEBUG, INFO, WARNING, ERROR).",
)
@click.pass_context
def main(ctx: click.Context, quiet: bool, log_level: str) -> None:
    """RAG — chat with your documents.

    \b
    Shorthand — ask a question against the default ./db database:
      rag 'What is AFM?'
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = not quiet

    numeric = getattr(logging, log_level.upper(), logging.WARNING)
    logging.basicConfig(level=numeric, format="%(name)s: %(message)s")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def _print_config(cfg: object) -> None:
    """Print a one-line config summary to the console."""
    from rag.config import RagConfig
    config: RagConfig = cfg  # type: ignore[assignment]
    console.print(
        f"[dim]config  llm=[bold]{config.llm_provider}/{config.llm_model}[/bold]"
        f"  embed={config.embed_provider}/{config.embed_model}"
        f"  db={config.db}  table={config.table}  top_k={config.top_k}[/dim]"
    )


@main.command()
@click.argument("db")
@click.argument("path", type=click.Path(exists=True))
@click.option("--table", default="embeddings", show_default=True, help="Table name in the DB.")
@click.option("--embed", default="mistral", show_default=True, help="Embedding provider/model (e.g. mistral, voyageai).")
@click.option("--chunk", "chunk_method", default="basic", show_default=True, type=click.Choice(["basic", "unstructured"]), help="Chunking method.")
@click.option("--chunk-size", default=1000, show_default=True, help="Max characters per chunk.")
@click.option("--chunk-overlap", default=200, show_default=True, help="Overlap between chunks.")
@click.option("--pdf-strategy", default="fast", show_default=True, type=click.Choice(["fast", "hi-res", "auto"]), help="PDF extraction strategy for --chunk unstructured. 'fast' uses pdfminer (quick, digital PDFs). 'hi-res' uses layout detection + OCR (slow, needed for scanned PDFs).")
@click.option("--skip", "-s", "skip_extensions", multiple=True, metavar="EXT", help="Extensions to skip (e.g. .png).")
@click.option("--overwrite", is_flag=True, help="Re-ingest files that are already in the database.")
@click.option("--profile", default=None, help="Config profile from rag.toml to use (e.g. --profile work).")
@click.pass_context
def ingest(
    ctx: click.Context,
    db: str,
    path: str,
    table: str,
    embed: str,
    chunk_method: str,
    chunk_size: int,
    chunk_overlap: int,
    pdf_strategy: str,
    skip_extensions: tuple[str, ...],
    overwrite: bool,
    profile: str | None,
) -> None:
    """Ingest a file or directory into DB.

    Files already present in the database are skipped. Use --overwrite to
    delete existing records and re-ingest.

    \b
    Examples:
      rag ingest ./db ./docs/
      rag ingest ./db paper.pdf --embed mistral --chunk-size 800
      rag ingest ./db paper.pdf --overwrite
    """
    embed = _fuzzy_option(embed, list(EMBED_ALIASES), "embed")
    cfg = load_config(profile=profile, overrides={
        "db": db, "table": table, "embed": embed,
        "chunk_method": chunk_method, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
        "pdf_strategy": pdf_strategy,
    })

    if ctx.obj and ctx.obj.get("verbose"):
        _print_config(cfg)

    try:
        result = ingestion_service.ingest(
            cfg,
            Path(path),
            skip_extensions=set(skip_extensions),
            overwrite=overwrite,
            progress=_service_progress,
        )
    except RuntimeError as exc:
        msg = str(exc)
        if "Unsupported file type" in msg:
            raise click.ClickException(msg + "\nTip: use --chunk unstructured for .docx, .pptx, .xlsx and other Office formats.")
        raise click.ClickException(msg)

    for source in result.skipped_files:
        console.print(f"[dim]  skipping {source} (already ingested)[/dim]")
    for file_path, exc in result.failures:
        msg = str(exc)
        hint = "\n  Tip: use --chunk unstructured for .docx, .pptx, .xlsx and other Office formats." if "Unsupported file type" in msg else ""
        console.print(f"[yellow]  skipped {file_path.name}: {msg}{hint}[/yellow]")

    if result.rows_written == 0:
        console.print("[yellow]All files already ingested — nothing to do.[/yellow]")
        if result.mode == "vector":
            console.print("Use [bold]--overwrite[/bold] to force re-ingestion.")
        return

    if result.mode == "vector":
        console.print(f"Chunked into [bold]{len(result.chunks)}[/bold] chunks.")
        console.print(
            f"[green]{'Created' if not result.db_existed else 'Updated'}[/green] "
            f"[bold]{cfg.db}[/bold] (table: {cfg.table}) "
            f"— stored {result.rows_written} chunks."
        )
        return

    console.print(
        f"[green]{'Created' if not result.db_existed else 'Updated'}[/green] "
        f"[bold]{cfg.db}[/bold] — uploaded {result.rows_written} file(s)."
    )


@main.command()
@click.argument("db")
@click.argument("question")
@click.option("--table", default="embeddings", show_default=True)
@click.option("--embed", default=None, help="Embedding provider/model. Defaults to match --llm when possible.")
@click.option("--llm", default="mistral", show_default=True, help="LLM provider/model (e.g. mistral, mistral-large, claude).")
@click.option("--top-k", default=5, show_default=True, help="Number of chunks to retrieve.")
@click.option("--max-tokens", default=4096, show_default=True, help="Max tokens in the LLM response (increase if answers are cut off).")
@click.option("--context", "-c", "show_context", is_flag=True, help="Print retrieved chunks before the answer.")
@click.option(
    "--stream/--no-stream",
    default=True,
    show_default=True,
    help=(
        "Stream the answer token-by-token for compatible LLMs. "
        "Use --no-stream for full markdown-rendered output."
    ),
)
@click.option("--profile", default=None, help="Config profile from rag.toml to use (e.g. --profile work).")
@click.pass_context
def ask(
    ctx: click.Context,
    db: str,
    question: str,
    table: str,
    embed: str,
    llm: str,
    top_k: int,
    max_tokens: int,
    show_context: bool,
    stream: bool,
    profile: str | None,
) -> None:
    """Ask a question against an ingested database.

    \b
    Examples:
      rag ask ./db "What is AFM?"
      rag ask ./db "What is AFM?" --llm mistral --top-k 8 --context
      rag ask ./db "What is AFM?" --stream
    """
    llm = _fuzzy_option(llm, list(LLM_ALIASES), "llm")
    if embed is None:
        embed = _infer_embed_from_llm(llm)
    embed = _fuzzy_option(embed, list(EMBED_ALIASES), "embed")
    cfg = load_config(profile=profile, overrides={
        "db": db, "table": table, "embed": embed, "llm": llm,
        "top_k": top_k, "max_tokens": max_tokens,
    })

    if ctx.obj and ctx.obj.get("verbose"):
        _print_config(cfg)

    try:
        retrieved = query_service.retrieve(cfg, question, progress=_service_progress)
    except Exception as exc:
        msg = str(exc)
        if "query dim" in msg and "doesn't match" in msg:
            raise click.ClickException(
                f"{msg}\nHint: make sure --embed matches the model used at ingest time."
            )
        if isinstance(exc, RuntimeError):
            raise click.ClickException(msg)
        raise

    source_summary_future = _start_source_summary(retrieved)

    if show_context and retrieved.chunks:
        console.print(Rule("Retrieved context"))
        for i, chunk in enumerate(retrieved.chunks, 1):
            console.print(_format_context_chunk(i, chunk))
        console.print(Rule())

    if stream:
        stream_gen = generate_stream(
            query=question,
            context=retrieved.chunks,
            provider=cfg.llm_provider,
            model=cfg.llm_model,
            max_tokens=cfg.max_tokens,
        )
        with console.status(f"Connecting to [bold]{cfg.llm_model}[/bold]..."):
            first = next(stream_gen, None)
        if first is not None:
            print(first, end="", flush=True)
            for chunk in stream_gen:
                print(chunk, end="", flush=True)
            print()
    else:
        with console.status(f"Generating answer with [bold]{cfg.llm_model}[/bold]..."):
            answer = generate(
                query=question,
                context=retrieved.chunks,
                provider=cfg.llm_provider,
                model=cfg.llm_model,
                max_tokens=cfg.max_tokens,
            )
        console.print(Markdown(answer))

    source_summary = source_summary_future.result()
    if source_summary:
        console.print()
        console.print(Rule("Sources"))
        for line in source_summary.splitlines()[1:]:
            console.print(line)


@main.command("list")
@click.argument("db")
@click.option("--table", default="embeddings", show_default=True, help="Table name in the DB.")
@click.option("--profile", default=None, help="Config profile from rag.toml to use.")
@click.pass_context
def list_cmd(ctx: click.Context, db: str, table: str, profile: str | None) -> None:
    """List files ingested into DB.

    \b
    Examples:
      rag list ./db
      rag list s3://my-bucket/rag-db
      rag list postgres://user:pass@localhost/ragdb
    """
    from rag.config import load_config
    from rag.db import get_db_adapter
    cfg = load_config(profile=profile, overrides={"db": db, "table": table})
    adapter = get_db_adapter(cfg.db, cfg.table)
    try:
        sources = adapter.list_sources()
    except RuntimeError as exc:
        raise click.ClickException(str(exc))
    if not sources:
        console.print("[dim]No files ingested yet.[/dim]")
        return
    for source in sources:
        console.print(source)


@main.command("gemini")
@click.argument("path", type=click.Path(exists=True))
@click.argument("question")
@click.option("--model", default="gemini-2.5-flash", show_default=True, help="Gemini model (e.g. gemini-2.5-flash, gemini-2.5-pro).")
@click.option("--max-tokens", default=1024, show_default=True)
def gemini_cmd(path: str, question: str, model: str, max_tokens: int) -> None:
    """Ask a question using Gemini direct mode or File Search mode.

    Routes to Gemini File Search when Office files are present, otherwise uses
    direct file uploads. Best for ad-hoc questions against a handful of
    documents. Requires GEMINI_API_KEY.

    \b
    Examples:
      rag gemini ./docs/ "What is AFM?"
      rag gemini paper.pdf "Summarize the key findings" --model gemini-2.5-pro
    """
    src = Path(path)
    direct = GeminiFileApiClient(model=model)
    with console.status("Uploading files to Gemini..."):
        prepared = direct.prepare(src)
    if prepared.uploaded_count == 0:
        raise click.ClickException(
            f"No supported files found under {path}. "
            "Supported: .pdf, .txt, .md, .html, .htm, .csv, .doc, .docx, .ppt, .pptx, .xls, .xlsx"
        )
    mode_label = "File Search" if prepared.mode == "file_search" else "Direct"
    console.print(f"Uploaded [bold]{prepared.uploaded_count}[/bold] file(s) via Gemini {mode_label}.")
    with console.status(f"Generating answer with [bold]{model}[/bold]..."):
        answer = direct.ask_prepared(question, prepared, max_tokens=max_tokens)
    console.print(Markdown(answer))


@main.command("gui")
@click.option("--host", default="127.0.0.1", show_default=True, help="Host interface to bind.")
@click.option("--port", default=7860, show_default=True, help="Port to listen on.")
@click.option("--no-browser", is_flag=True, help="Do not open the GUI in a browser automatically.")
@click.option(
    "--root-dir",
    default=str(Path.cwd()),
    show_default=True,
    type=click.Path(exists=True, file_okay=False),
    help="Root directory shown in the file browser.",
)
def gui_cmd(host: str, port: int, no_browser: bool, root_dir: str) -> None:
    """Launch the Gradio GUI front-end."""
    try:
        from rag.app import launch
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    try:
        launch(host=host, port=port, inbrowser=not no_browser, root_dir=root_dir)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
