"""Gemini File API — direct end-to-end: upload files and ask questions.

No vector store; files are uploaded to Gemini and used as context.
Requires: pip install 'rag[gemini]'  and GEMINI_API_KEY.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google.genai import types as genai_types

GEMINI_DIRECT_EXTENSIONS: frozenset[str] = frozenset({
    ".pdf", ".txt", ".md",
    ".doc", ".docx",
    ".ppt", ".pptx",
    ".xls", ".xlsx",
    ".html", ".htm",
    ".csv",
})


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
            return [self._upload_one(path)]
        if path.is_dir():
            return self._upload_dir(path)
        raise FileNotFoundError(f"Not a file or directory: {path}")

    def _upload_one(self, path: Path) -> genai_types.File:
        return self.client.files.upload(file=str(path))

    def _upload_dir(self, directory: Path) -> list[genai_types.File]:
        files: list[genai_types.File] = []
        for p in sorted(directory.iterdir()):
            if not p.is_file() or p.name.startswith("."):
                continue
            if p.suffix.lower() in GEMINI_DIRECT_EXTENSIONS:
                files.append(self._upload_one(p))
        return files

    def ask(self, question: str, files: list[genai_types.File], *, max_tokens: int = 1024) -> str:
        """Send question with uploaded file context to Gemini. Returns answer text."""
        from google.genai import types
        if not files:
            raise ValueError("No files provided; upload files first.")
        contents: list[Any] = [question]
        contents.extend(files)
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(max_output_tokens=max_tokens),
        )
        return response.text or ""


GeminiDirect = GeminiFileApiClient
