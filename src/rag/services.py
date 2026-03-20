"""Service layer for ingestion and retrieval orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from rag.chunk import BASIC_EXTENSIONS, Chunk, chunk_file, list_chunkable_files
from rag.config import RagConfig
from rag.db import get_db_adapter
from rag.embed import get_embed_adapter

ProgressStage = Literal["chunking", "embedding", "writing", "uploading", "indexing", "searching"]
ProgressEvent = Literal["start", "end"]


class ProgressCallback(Protocol):
    def __call__(self, event: ProgressEvent, stage: ProgressStage) -> None: ...


@dataclass
class IngestionResult:
    mode: str
    db_existed: bool
    chunks: list[Chunk]
    rows_written: int
    skipped_files: list[str]
    failures: list[tuple[Path, Exception]]


class IngestionService:
    """Coordinates ingestion across chunking, embedding, and storage adapters."""

    def ingest(
        self,
        cfg: RagConfig,
        source_path: Path,
        *,
        skip_extensions: set[str],
        overwrite: bool,
        progress: ProgressCallback | None = None,
    ) -> IngestionResult:
        adapter = get_db_adapter(cfg.db, cfg.table)
        adapter.preflight()

        db_existed = adapter.exists()
        is_managed = cfg.db.startswith(("vertex://", "bedrock-kb://"))
        if is_managed:
            return self._ingest_managed(
                cfg,
                source_path,
                skip_extensions=skip_extensions,
                overwrite=overwrite,
                db_existed=db_existed,
                progress=progress,
            )
        return self._ingest_vector(
            cfg,
            source_path,
            skip_extensions=skip_extensions,
            overwrite=overwrite,
            db_existed=db_existed,
            progress=progress,
        )

    def _ingest_managed(
        self,
        cfg: RagConfig,
        source_path: Path,
        *,
        skip_extensions: set[str],
        overwrite: bool,
        db_existed: bool,
        progress: ProgressCallback | None,
    ) -> IngestionResult:
        adapter = get_db_adapter(cfg.db, cfg.table)
        mode = "bedrock-kb" if cfg.db.startswith("bedrock-kb://") else "vertex"
        normalized_skip = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in skip_extensions
        }

        files_to_process: list[Path] = []
        skipped_files: list[str] = []

        if source_path.is_dir():
            all_files = [
                file_path for file_path in sorted(source_path.iterdir())
                if file_path.is_file()
                and not file_path.name.startswith(".")
                and file_path.suffix.lower() in BASIC_EXTENSIONS
                and file_path.suffix.lower() not in normalized_skip
            ]
            for file_path in all_files:
                if db_existed and adapter.has_source(file_path.name):
                    if overwrite:
                        adapter.delete_source(file_path.name)
                        files_to_process.append(file_path)
                    else:
                        skipped_files.append(file_path.name)
                else:
                    files_to_process.append(file_path)
        else:
            if source_path.suffix.lower() not in BASIC_EXTENSIONS:
                raise RuntimeError(
                    f"Managed ingest supports .pdf, .txt, .md; got {source_path.suffix!r}"
                )
            if db_existed and adapter.has_source(source_path.name):
                if overwrite:
                    adapter.delete_source(source_path.name)
                    files_to_process = [source_path]
                else:
                    skipped_files.append(source_path.name)
            else:
                files_to_process = [source_path]

        if not files_to_process:
            return IngestionResult(
                mode=mode,
                db_existed=db_existed,
                chunks=[],
                rows_written=0,
                skipped_files=skipped_files,
                failures=[],
            )

        self._emit(progress, "start", "uploading")
        adapter.setup(embedding_dim=768)
        rows = [{"path": str(path.resolve()), "source": path.name} for path in files_to_process]
        adapter.add(rows)
        self._emit(progress, "end", "uploading")

        if hasattr(adapter, "wait_for_ingestion"):
            self._emit(progress, "start", "indexing")
            adapter.wait_for_ingestion()
            self._emit(progress, "end", "indexing")

        return IngestionResult(
            mode=mode,
            db_existed=db_existed,
            chunks=[],
            rows_written=len(rows),
            skipped_files=skipped_files,
            failures=[],
        )

    def _ingest_vector(
        self,
        cfg: RagConfig,
        source_path: Path,
        *,
        skip_extensions: set[str],
        overwrite: bool,
        db_existed: bool,
        progress: ProgressCallback | None,
    ) -> IngestionResult:
        adapter = get_db_adapter(cfg.db, cfg.table)
        chunks: list[Chunk] = []
        failures: list[tuple[Path, Exception]] = []
        skipped_files: list[str] = []

        if source_path.is_dir():
            all_files = list_chunkable_files(source_path, cfg.chunk_method, skip_extensions)
            files_to_process: list[Path] = []
            to_overwrite: set[str] = set()
            for file_path in all_files:
                if db_existed and adapter.has_source(file_path.name):
                    if overwrite:
                        to_overwrite.add(file_path.name)
                        files_to_process.append(file_path)
                    else:
                        skipped_files.append(file_path.name)
                else:
                    files_to_process.append(file_path)

            if not files_to_process:
                return IngestionResult(
                    mode="vector",
                    db_existed=db_existed,
                    chunks=[],
                    rows_written=0,
                    skipped_files=skipped_files,
                    failures=[],
                )

            # Chunk first; only delete existing records for files that chunk successfully
            self._emit(progress, "start", "chunking")
            for file_path in files_to_process:
                try:
                    file_chunks = chunk_file(
                        file_path,
                        chunk_size=cfg.chunk_size,
                        overlap=cfg.chunk_overlap,
                        chunk_method=cfg.chunk_method,
                        pdf_strategy=cfg.pdf_strategy,
                    )
                    if file_path.name in to_overwrite:
                        adapter.delete_source(file_path.name)
                    chunks.extend(file_chunks)
                except Exception as exc:
                    failures.append((file_path, exc))
            self._emit(progress, "end", "chunking")
        else:
            already_exists = db_existed and adapter.has_source(source_path.name)
            if already_exists and not overwrite:
                skipped_files.append(source_path.name)
                return IngestionResult(
                    mode="vector",
                    db_existed=db_existed,
                    chunks=[],
                    rows_written=0,
                    skipped_files=skipped_files,
                    failures=[],
                )

            try:
                self._emit(progress, "start", "chunking")
                chunks = chunk_file(
                    source_path,
                    chunk_size=cfg.chunk_size,
                    overlap=cfg.chunk_overlap,
                    chunk_method=cfg.chunk_method,
                    pdf_strategy=cfg.pdf_strategy,
                )
                self._emit(progress, "end", "chunking")
            except Exception as exc:
                raise RuntimeError(f"Failed to chunk {source_path.name}: {exc}") from exc

            # Delete only after successful chunk so a failed re-ingest preserves old records
            if already_exists:
                adapter.delete_source(source_path.name)

        if not chunks:
            raise RuntimeError("No chunks extracted — nothing to ingest.")

        self._emit(progress, "start", "embedding")
        embedder = get_embed_adapter(cfg.embed_provider, cfg.embed_model)
        vectors = embedder.embed([chunk.text for chunk in chunks])
        self._emit(progress, "end", "embedding")

        if len(vectors) != len(chunks):
            raise RuntimeError(
                "Embedding adapter returned a different number of vectors than chunks."
            )

        self._emit(progress, "start", "writing")
        adapter.setup(embedding_dim=embedder.dim)
        rows = [
            {"vector": vector, "text": chunk.text, "source": chunk.source}
            for chunk, vector in zip(chunks, vectors)
        ]
        adapter.add(rows)
        self._emit(progress, "end", "writing")

        return IngestionResult(
            mode="vector",
            db_existed=db_existed,
            chunks=chunks,
            rows_written=len(rows),
            skipped_files=skipped_files,
            failures=failures,
        )

    @staticmethod
    def _emit(
        progress: ProgressCallback | None,
        event: ProgressEvent,
        stage: ProgressStage,
    ) -> None:
        if progress is not None:
            progress(event, stage)


class QueryService:
    """Coordinates retrieval across storage and embedding adapters."""

    def retrieve(
        self,
        cfg: RagConfig,
        question: str,
        *,
        progress: ProgressCallback | None = None,
    ) -> list[str]:
        adapter = get_db_adapter(cfg.db, cfg.table)
        if not adapter.exists():
            raise RuntimeError(
                f"No database found at {cfg.db!r} (table: {cfg.table}). Run 'rag ingest' first."
            )

        self._emit(progress, "start", "searching")
        if hasattr(adapter, "query_by_text"):
            result = adapter.query_by_text(question, cfg.top_k)
            self._emit(progress, "end", "searching")
            return result

        embedder = get_embed_adapter(cfg.embed_provider, cfg.embed_model)
        question_vector = embedder.embed([question])[0]
        result = adapter.query(query_vector=question_vector, k=cfg.top_k)
        self._emit(progress, "end", "searching")
        return result

    @staticmethod
    def _emit(
        progress: ProgressCallback | None,
        event: ProgressEvent,
        stage: ProgressStage,
    ) -> None:
        if progress is not None:
            progress(event, stage)
