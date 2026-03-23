"""Gemini end-to-end helpers for direct uploads and File Search routing.

Requires: pip install 'rag[gemini]' and GEMINI_API_KEY.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

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


@dataclass(frozen=True)
class GeminiPreparedContext:
    mode: Literal["direct", "file_search"]
    uploaded_count: int
    files: list[genai_types.File] | None = None
    file_search_store_name: str | None = None


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
        contents: list[Any] = [_build_prompt(question)]
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

    def prepare(self, path: Path) -> GeminiPreparedContext:
        files_to_consider = self._collect_supported_files(path)
        if not files_to_consider:
            return GeminiPreparedContext(mode="direct", uploaded_count=0, files=[])
        has_file_search_extension = any(
            p.suffix.lower() in GEMINI_FILE_SEARCH_EXTENSIONS for p in files_to_consider
        )
        if has_file_search_extension:
            store_name = self._upload_to_file_search_store(files_to_consider)
            return GeminiPreparedContext(
                mode="file_search",
                uploaded_count=len(files_to_consider),
                file_search_store_name=store_name,
            )
        uploaded = [self._upload_one(p) for p in files_to_consider]
        return GeminiPreparedContext(
            mode="direct",
            uploaded_count=len(uploaded),
            files=uploaded,
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
            contents=_build_prompt(question),
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return response.text or ""

    def _upload_to_file_search_store(self, paths: list[Path]) -> str:
        store = self.client.file_search_stores.create(
            config={"display_name": f"rag-lette-{int(time.time())}"}
        )
        if not store.name:
            raise RuntimeError("Gemini File Search store creation returned an empty name.")
        try:
            for path in paths:
                operation = self.client.file_search_stores.upload_to_file_search_store(
                    file_search_store_name=store.name,
                    file=str(path),
                    config={"display_name": path.name},
                )
                self._wait_for_operation(operation)
        except Exception:
            self._delete_file_search_store(store.name)
            raise
        return store.name

    def _wait_for_operation(self, operation: Any, *, timeout_seconds: float = 300.0) -> Any:
        deadline = time.monotonic() + timeout_seconds
        current = operation
        while not current.done:
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out while waiting for Gemini file operation.")
            time.sleep(1.0)
            current = self.client.operations.get(current)
        if current.error:
            raise RuntimeError(f"Gemini file operation failed: {current.error}")
        return current

    def _delete_file_search_store(self, name: str) -> None:
        try:
            self.client.file_search_stores.delete(name=name, config={"force": True})
        except Exception:
            return


GeminiDirect = GeminiFileApiClient


def _build_prompt(query: str) -> str:
    return (
        "Answer the query using only the context provided below. "
        "Quote relevant passages to support your answer. "
        "Treat the source filename attached to each chunk as retrieval metadata, not as a claim. "
        "If the context does not contain enough information, say so.\n\n"
        "<context>\n\n</context>\n\n"
        f"<query>{query}</query>"
    )
