"""CLI entry point: `rag ingest` and `rag ask`."""
from __future__ import annotations

import logging
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule

from rag.config import load_config
from rag.gemini import GeminiFileApiClient
from rag.generate import generate, generate_stream
from rag.services import IngestionService, QueryService

console = Console()
ingestion_service = IngestionService()
query_service = QueryService()


def _service_progress(event: str, stage: str) -> None:
    start_messages = {
        "chunking": "Chunking documents...",
        "embedding": "Embedding chunks...",
        "writing": "Writing to database...",
        "uploading": "Uploading to Vertex AI RAG corpus...",
        "searching": "Searching...",
    }
    end_messages = {
        "chunking": "Chunking complete.",
        "embedding": "Embedding complete.",
        "writing": "Write complete.",
        "uploading": "Upload complete.",
        "searching": "Search complete.",
    }
    if event == "start":
        console.print(f"[dim]{start_messages.get(stage, stage)}[/dim]")
    elif event == "end":
        console.print(f"[dim]{end_messages.get(stage, stage)}[/dim]")


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
@click.option("--verbose", "-v", is_flag=True, help="Print resolved config before running.")
@click.option(
    "--log-level",
    default="WARNING",
    show_default=True,
    metavar="LEVEL",
    help="Python log level (DEBUG, INFO, WARNING, ERROR).",
)
@click.pass_context
def main(ctx: click.Context, verbose: bool, log_level: str) -> None:
    """RAG — chat with your documents.

    \b
    Shorthand — ask a question against the default ./db database:
      rag 'What is AFM?'
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

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
        raise click.ClickException(str(exc))

    for source in result.skipped_files:
        console.print(f"[dim]  skipping {source} (already ingested)[/dim]")
    for file_path, exc in result.failures:
        console.print(f"[yellow]  skipped {file_path.name}: {exc}[/yellow]")

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
@click.option("--embed", default="mistral", show_default=True)
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
    cfg = load_config(profile=profile, overrides={
        "db": db, "table": table, "embed": embed, "llm": llm,
        "top_k": top_k, "max_tokens": max_tokens,
    })

    if ctx.obj and ctx.obj.get("verbose"):
        _print_config(cfg)

    try:
        retrieved = query_service.retrieve(cfg, question, progress=_service_progress)
    except RuntimeError as exc:
        raise click.ClickException(str(exc))

    if show_context and retrieved:
        console.print(Rule("Retrieved context"))
        for i, chunk in enumerate(retrieved, 1):
            console.print(f"[dim][{i}][/dim] {chunk}\n")
        console.print(Rule())

    if stream:
        stream_gen = generate_stream(
            query=question,
            context=retrieved,
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
                context=retrieved,
                provider=cfg.llm_provider,
                model=cfg.llm_model,
                max_tokens=cfg.max_tokens,
            )
        console.print(Markdown(answer))


@main.command("gemini")
@click.argument("path", type=click.Path(exists=True))
@click.argument("question")
@click.option("--model", default="gemini-2.5-flash", show_default=True, help="Gemini model (e.g. gemini-2.5-flash, gemini-2.5-pro).")
@click.option("--max-tokens", default=1024, show_default=True)
def gemini_cmd(path: str, question: str, model: str, max_tokens: int) -> None:
    """Ask a question using Gemini File API (no ingest step, no persistent store).

    Uploads files to Gemini and uses them as context. Best for ad-hoc questions
    against a handful of documents. Requires GEMINI_API_KEY.

    \b
    Examples:
      rag gemini ./docs/ "What is AFM?"
      rag gemini paper.pdf "Summarize the key findings" --model gemini-2.5-pro
    """
    src = Path(path)
    direct = GeminiFileApiClient(model=model)
    with console.status("Uploading files to Gemini..."):
        files = direct.upload(src)
    if not files:
        raise click.ClickException(
            f"No supported files found under {path}. "
            "Supported: .pdf, .txt, .md, .doc, .docx, .ppt, .pptx, .xls, .xlsx, .html, .csv"
        )
    console.print(f"Uploaded [bold]{len(files)}[/bold] file(s).")
    with console.status(f"Generating answer with [bold]{model}[/bold]..."):
        answer = direct.ask(question, files, max_tokens=max_tokens)
    console.print(Markdown(answer))
