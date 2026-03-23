"""Gemini end-to-end helpers for direct uploads and File Search routing.

Requires: pip install 'rag[gemini]' and GEMINI_API_KEY.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

from rag.generate import _build_prompt

if TYPE_CHECKING:
    from google.genai import types as genai_types

GEMINI_DIRECT_EXTENSIONS: frozenset[str] = frozenset({
    ".pdf", ".txt", ".md",
    ".html", ".htm",
    ".csv",
})
GEMINI_FILE_SEARCH_EXTENSIONS: frozenset[str] = frozenset({
    ".doc", ".docx",
    ".ppt", ".pptx",
    ".xls", ".xlsx",
})
GEMINI_SUPPORTED_EXTENSIONS: frozenset[str] = (
    GEMINI_DIRECT_EXTENSIONS | GEMINI_FILE_SEARCH_EXTENSIONS
)
GEMINI_FILE_SEARCH_MIME_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/plain",
    ".html": "text/plain",
    ".htm": "text/plain",
    ".csv": "text/plain",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


@dataclass(frozen=True)
class GeminiPreparedContext:
    mode: Literal["direct", "file_search"]
    uploaded_count: int
    files: list[genai_types.File] | None = None
    file_search_store_name: str | None = None
    upload_summary: str | None = None


GeminiMode = Literal["auto", "direct", "file_search"]


class GeminiFileApiClient:
    """Upload files to Gemini File API and ask questions (no persistent store)."""

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash") -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "The 'google-genai' package is required for Gemini direct mode.\n"
                "Install it with:  pip install \"rag[gemini]\"\n"
                f"Original error: {exc}"
            ) from exc
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set")
        self.client = genai.Client(api_key=key)
        self.model = model

    def upload(self, path: Path) -> list[genai_types.File]:
        """Upload file(s) to Gemini File API. Returns list of file references for generate_content."""
        path = path.expanduser().resolve()
        if path.is_file():
            if path.suffix.lower() not in GEMINI_DIRECT_EXTENSIONS:
                return []
            return [self._upload_one(path)]
        if path.is_dir():
            return self._upload_dir(path)
        raise FileNotFoundError(f"Not a file or directory: {path}")

    def _upload_one(self, path: Path) -> genai_types.File:
        suffix = path.suffix.lower()
        mime_type = "application/pdf" if suffix == ".pdf" else "text/plain"
        return self.client.files.upload(file=str(path), config={"mime_type": mime_type})

    def _upload_dir(self, directory: Path) -> list[genai_types.File]:
        files: list[genai_types.File] = []
        for p in sorted(directory.iterdir()):
            if not p.is_file() or p.name.startswith("."):
                continue
            if p.suffix.lower() in GEMINI_DIRECT_EXTENSIONS:
                files.append(self._upload_one(p))
        return files

    def ask(
        self,
        question: str,
        files: list[genai_types.File],
        *,
        max_tokens: int | None = None,
    ) -> str:
        """Send question with uploaded file context to Gemini. Returns answer text."""
        from google.genai import types
        if not files:
            raise ValueError("No files provided; upload files first.")
        contents: list[Any] = [_build_prompt(question, ())]
        contents.extend(files)
        config_kwargs: dict[str, int] = {}
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return response.text or ""

    def prepare(
        self,
        path: Path,
        mode: GeminiMode = "auto",
        *,
        progress: Callable[[str], None] | None = None,
    ) -> GeminiPreparedContext:
        files_to_consider = self._collect_supported_files(path)
        if not files_to_consider:
            return GeminiPreparedContext(mode="direct", uploaded_count=0, files=[])
        total = len(files_to_consider)
        self._emit_progress(progress, f"Found {total} supported file(s).")
        if mode == "direct":
            direct_files = [
                p for p in files_to_consider if p.suffix.lower() in GEMINI_DIRECT_EXTENSIONS
            ]
            uploaded: list[genai_types.File] = []
            for idx, p in enumerate(direct_files, start=1):
                self._emit_progress(progress, f"[{idx}/{len(direct_files)}] Uploading {p.name} (direct)")
                uploaded.append(self._upload_one(p))
            return GeminiPreparedContext(
                mode="direct",
                uploaded_count=len(uploaded),
                files=uploaded,
                upload_summary=f"Direct uploads: {len(uploaded)}",
            )
        has_file_search_extension = any(
            p.suffix.lower() in GEMINI_FILE_SEARCH_EXTENSIONS for p in files_to_consider
        )
        if mode == "file_search" or has_file_search_extension:
            office_count = sum(
                1 for p in files_to_consider if p.suffix.lower() in GEMINI_FILE_SEARCH_EXTENSIONS
            )
            regular_count = len(files_to_consider) - office_count
            self._emit_progress(progress, "Using Gemini File Search mode.")
            store_name = self._upload_to_file_search_store(files_to_consider, progress=progress)
            return GeminiPreparedContext(
                mode="file_search",
                uploaded_count=len(files_to_consider),
                file_search_store_name=store_name,
                upload_summary=(
                    f"File Search uploads: {regular_count} regular, {office_count} office import"
                ),
            )
        uploaded: list[genai_types.File] = []
        for idx, p in enumerate(files_to_consider, start=1):
            self._emit_progress(progress, f"[{idx}/{total}] Uploading {p.name} (direct)")
            uploaded.append(self._upload_one(p))
        return GeminiPreparedContext(
            mode="direct",
            uploaded_count=len(uploaded),
            files=uploaded,
            upload_summary=f"Direct uploads: {len(uploaded)}",
        )

    def ask_prepared(
        self,
        question: str,
        prepared: GeminiPreparedContext,
        *,
        max_tokens: int | None = None,
    ) -> str:
        if prepared.mode == "direct":
            return self.ask(question, prepared.files or [], max_tokens=max_tokens)
        if not prepared.file_search_store_name:
            raise RuntimeError("Missing file search store name.")
        try:
            return self._ask_with_file_search(
                question,
                prepared.file_search_store_name,
                max_tokens=max_tokens,
            )
        finally:
            self._delete_file_search_store(prepared.file_search_store_name)

    def _collect_supported_files(self, path: Path) -> list[Path]:
        path = path.expanduser().resolve()
        if path.is_file():
            return [path] if path.suffix.lower() in GEMINI_SUPPORTED_EXTENSIONS else []
        if path.is_dir():
            collected: list[Path] = []
            for p in sorted(path.iterdir()):
                if not p.is_file() or p.name.startswith("."):
                    continue
                if p.suffix.lower() in GEMINI_SUPPORTED_EXTENSIONS:
                    collected.append(p)
            return collected
        raise FileNotFoundError(f"Not a file or directory: {path}")

    def _ask_with_file_search(
        self,
        question: str,
        store_name: str,
        *,
        max_tokens: int | None = None,
    ) -> str:
        from google.genai import types
        config_kwargs: dict[str, Any] = {
            "tools": [
                types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[store_name]
                    )
                )
            ]
        }
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        response = self.client.models.generate_content(
            model=self.model,
            contents=_build_prompt(question, ()),
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return response.text or ""

    def _upload_to_file_search_store(
        self,
        paths: list[Path],
        *,
        progress: Callable[[str], None] | None = None,
    ) -> str:
        store = self.client.file_search_stores.create(
            config={"display_name": f"rag-lette-{int(time.time())}"}
        )
        if not store.name:
            raise RuntimeError("Gemini File Search store creation returned an empty name.")
        total = len(paths)
        try:
            for idx, path in enumerate(paths, start=1):
                if path.suffix.lower() in GEMINI_FILE_SEARCH_EXTENSIONS:
                    self._emit_progress(progress, f"[{idx}/{total}] Uploading {path.name} (office import)")
                    self._import_office_file_to_file_search_store(
                        store.name,
                        path,
                        progress=progress,
                    )
                else:
                    self._emit_progress(progress, f"[{idx}/{total}] Uploading {path.name} (file search)")
                    operation = self._upload_one_to_file_search_store(store.name, path)
                    self._wait_for_operation(
                        operation,
                        progress=progress,
                        label=f"Indexing {path.name}",
                    )
        except Exception:
            self._delete_file_search_store(store.name)
            raise
        return store.name

    def _import_office_file_to_file_search_store(
        self,
        store_name: str,
        path: Path,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        mime_type = GEMINI_FILE_SEARCH_MIME_TYPES.get(path.suffix.lower())
        if not mime_type:
            raise RuntimeError(f"Unsupported Office file extension: {path.suffix}")
        uploaded = self.client.files.upload(
            file=str(path),
            config={
                "mime_type": mime_type,
                "display_name": path.name,
            },
        )
        if not uploaded.name:
            raise RuntimeError(f"Gemini file upload returned empty name for {path.name}.")
        file_name = uploaded.name
        try:
            self._emit_progress(progress, f"Processing {path.name} in Gemini...")
            self._wait_for_file_processing(file_name, progress=progress, label=path.name)
            self._emit_progress(progress, f"Importing {path.name} into File Search...")
            operation = self._import_file_to_search_store(store_name, file_name, path.name)
            self._wait_for_operation(
                operation,
                progress=progress,
                label=f"Importing {path.name}",
            )
        finally:
            self._delete_uploaded_file(file_name)

    def _upload_one_to_file_search_store(self, store_name: str, path: Path) -> Any:
        mime_type = GEMINI_FILE_SEARCH_MIME_TYPES.get(path.suffix.lower())
        if not mime_type:
            raise RuntimeError(f"Unsupported file extension for Gemini File Search: {path.suffix}")
        delay_seconds = 1.0
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                return self.client.file_search_stores.upload_to_file_search_store(
                    file_search_store_name=store_name,
                    file=str(path),
                    config={
                        "display_name": path.name,
                        "mime_type": mime_type,
                    },
                )
            except Exception as exc:
                last_error = exc
                if attempt == 3 or not self._is_retryable_server_error(exc):
                    raise RuntimeError(
                        f"Gemini File Search upload failed for {path.name}: {exc}"
                    ) from exc
                time.sleep(delay_seconds)
                delay_seconds *= 2.0
        if last_error is None:
            raise RuntimeError(f"Gemini File Search upload failed for {path.name}.")
        raise RuntimeError(
            f"Gemini File Search upload failed for {path.name}: {last_error}"
        ) from last_error

    def _import_file_to_search_store(
        self,
        store_name: str,
        file_name: str,
        display_name: str,
    ) -> Any:
        delay_seconds = 1.0
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                return self.client.file_search_stores.import_file(
                    file_search_store_name=store_name,
                    file_name=file_name,
                )
            except Exception as exc:
                last_error = exc
                if attempt == 3 or not self._is_retryable_server_error(exc):
                    raise RuntimeError(
                        f"Gemini File Search import failed for {display_name}: {exc}"
                    ) from exc
                time.sleep(delay_seconds)
                delay_seconds *= 2.0
        if last_error is None:
            raise RuntimeError(f"Gemini File Search import failed for {display_name}.")
        raise RuntimeError(
            f"Gemini File Search import failed for {display_name}: {last_error}"
        ) from last_error

    def _wait_for_file_processing(
        self,
        file_name: str,
        *,
        timeout_seconds: float = 300.0,
        progress: Callable[[str], None] | None = None,
        label: str | None = None,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        start = time.monotonic()
        last_report_bucket = -1
        while True:
            file_ref = self.client.files.get(name=file_name)
            raw_state = str(file_ref.state or "")
            state = raw_state.split(".")[-1] if raw_state else ""
            if state == "ACTIVE":
                elapsed = int(time.monotonic() - start)
                target = label or file_name
                self._emit_progress(progress, f"{target} processed ({elapsed}s)")
                return
            if state == "FAILED":
                raise RuntimeError(f"Gemini file processing failed for {file_name}.")
            elapsed = int(time.monotonic() - start)
            bucket = elapsed // 5
            if bucket > last_report_bucket:
                last_report_bucket = bucket
                target = label or file_name
                shown_state = state or "PENDING"
                self._emit_progress(progress, f"Waiting for {target}: {shown_state} ({elapsed}s)")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out while waiting for Gemini file processing: {file_name}")
            time.sleep(1.0)

    def _wait_for_operation(
        self,
        operation: Any,
        *,
        timeout_seconds: float = 300.0,
        progress: Callable[[str], None] | None = None,
        label: str = "Operation",
    ) -> Any:
        deadline = time.monotonic() + timeout_seconds
        start = time.monotonic()
        last_report_bucket = -1
        current = operation
        while not current.done:
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out while waiting for Gemini file operation.")
            elapsed = int(time.monotonic() - start)
            bucket = elapsed // 5
            if bucket > last_report_bucket:
                last_report_bucket = bucket
                self._emit_progress(progress, f"{label} in progress ({elapsed}s)")
            time.sleep(1.0)
            current = self.client.operations.get(current)
        if current.error:
            raise RuntimeError(f"Gemini file operation failed: {current.error}")
        elapsed = int(time.monotonic() - start)
        self._emit_progress(progress, f"{label} complete ({elapsed}s)")
        return current

    def _delete_file_search_store(self, name: str) -> None:
        try:
            self.client.file_search_stores.delete(name=name, config={"force": True})
        except Exception:
            return

    def _delete_uploaded_file(self, name: str) -> None:
        try:
            self.client.files.delete(name=name)
        except Exception:
            return

    @staticmethod
    def _is_retryable_server_error(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int) and status_code >= 500:
            return True
        return exc.__class__.__name__ == "ServerError"

    @staticmethod
    def _emit_progress(progress: Callable[[str], None] | None, message: str) -> None:
        if progress is not None:
            progress(message)


GeminiDirect = GeminiFileApiClient
