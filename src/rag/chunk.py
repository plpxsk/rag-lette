"""Document chunking: extract and split text from supported files.

chunk_method controls both extraction and splitting strategy:
  "basic"        — pymupdf4llm for PDFs (markdown-aware), direct read for txt/md,
                   character-based splitter
  "unstructured" — unstructured library; supports PDF, DOCX, PPTX, XLSX, and more
                   Install: pip install "rag[unstructured]"
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

BASIC_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt", ".md"})

UNSTRUCTURED_EXTENSIONS: frozenset[str] = frozenset({
    ".pdf",
    ".doc", ".docx",
    ".ppt", ".pptx",
    ".xls", ".xlsx",
    ".odt",
    ".rtf",
    ".html", ".htm",
    ".eml", ".msg",
    ".csv",
    ".txt", ".md",
})

SUPPORTED_EXTENSIONS = BASIC_EXTENSIONS


@dataclass
class Chunk:
    text: str
    source: str  # filename
    heading: str = field(default="")


def chunk_file(
    path: Path,
    chunk_size: int = 1000,
    overlap: int = 200,
    chunk_method: str = "basic",
    pdf_strategy: str = "fast",
) -> list[Chunk]:
    """Extract text from a single file and split into overlapping chunks."""
    if chunk_method == "unstructured":
        return _chunk_file_unstructured(path, chunk_size, overlap, pdf_strategy)

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _extract_pdf(path)
    elif suffix in (".txt", ".md"):
        text = path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {path.suffix!r} ({path.name})")

    return _chunk_basic(text, chunk_size, overlap, path.name)


def list_chunkable_files(
    directory: Path,
    chunk_method: str = "basic",
    skip_extensions: set[str] | None = None,
) -> list[Path]:
    """Return sorted list of files in directory that can be chunked."""
    supported = UNSTRUCTURED_EXTENSIONS if chunk_method == "unstructured" else BASIC_EXTENSIONS
    skip = {
        e.lower() if e.startswith(".") else f".{e.lower()}"
        for e in (skip_extensions or set())
    }
    return [
        p for p in sorted(directory.iterdir())
        if p.is_file() and not p.name.startswith(".")
        and p.suffix.lower() in supported
        and p.suffix.lower() not in skip
    ]


def chunk_directory(
    directory: Path,
    chunk_size: int = 1000,
    overlap: int = 200,
    chunk_method: str = "basic",
    skip_extensions: set[str] | None = None,
    pdf_strategy: str = "fast",
) -> tuple[list[Chunk], list[tuple[Path, Exception]]]:
    """Chunk all supported files in a directory. Returns (chunks, failures)."""
    chunks: list[Chunk] = []
    failures: list[tuple[Path, Exception]] = []

    for path in list_chunkable_files(directory, chunk_method, skip_extensions):
        try:
            chunks.extend(
                chunk_file(
                    path,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    chunk_method=chunk_method,
                    pdf_strategy=pdf_strategy,
                )
            )
        except Exception as exc:
            failures.append((path, exc))

    return chunks, failures


def _extract_pdf(path: Path) -> str:
    """Extract text from a PDF using pymupdf4llm (basic method).

    pymupdf4llm converts PDFs to markdown, preserving structure like headers,
    tables, and lists. pymupdf_layout is imported first (when available) to
    activate improved layout analysis: better multi-column reading order, table
    detection, and header/footer removal — with no GPU or heavy ML dependencies.
    """
    try:
        import pymupdf.layout  # noqa: F401
    except ImportError:
        pass
    import pymupdf4llm
    return pymupdf4llm.to_markdown(str(path))


def _chunk_file_unstructured(
    path: Path,
    chunk_size: int,
    overlap: int,
    pdf_strategy: str = "fast",
) -> list[Chunk]:
    """Extract and chunk a file using the unstructured library.

    Supports PDF, DOCX, PPTX, XLSX, HTML, EML, and many more formats.
    Install the extra with: pip install "rag[unstructured]"

    pdf_strategy controls how PDFs are parsed (ignored for other formats):
      "fast"    — pdfminer text extraction; fast, good for digital PDFs (default)
      "hi_res"  — detectron2 layout model + optional OCR; slow, needed for scanned PDFs
      "auto"    — unstructured picks based on document characteristics
    """
    try:
        from unstructured.partition.auto import partition
        from unstructured.chunking.title import chunk_by_title
    except ImportError as exc:
        raise ImportError(
            "The 'unstructured' package is required for --chunk unstructured.\n"
            "Install it with:  pip install \"rag[unstructured]\"\n"
            f"Original error: {exc}"
        ) from exc

    suffix = path.suffix.lower()
    if suffix not in UNSTRUCTURED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type for unstructured method: {path.suffix!r} ({path.name})\n"
            f"Supported: {', '.join(sorted(UNSTRUCTURED_EXTENSIONS))}"
        )

    elements = partition(filename=str(path), strategy=pdf_strategy)

    chunked = chunk_by_title(
        elements,
        max_characters=chunk_size,
        overlap=overlap,
        combine_text_under_n_chars=min(200, chunk_size // 4),
    )

    return [
        Chunk(text=text, source=path.name)
        for el in chunked
        if (text := str(el).strip())
    ]


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """Split markdown text into (heading, section_text) pairs at heading boundaries."""
    matches = list(_MD_HEADING.finditer(text))
    if not matches:
        return [("", text)]
    sections = []
    for idx, m in enumerate(matches):
        heading = m.group(2).strip()
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections.append((heading, text[start:end]))
    return sections


def _chunk_basic(text: str, chunk_size: int, overlap: int, source: str) -> list[Chunk]:
    """Heading-aware chunking: split at markdown headings, then recursively split each
    section, then merge any trailing tiny chunks into the preceding chunk."""
    MIN_CHUNK_CHARS = 100
    raw: list[Chunk] = []
    for heading, sec_text in _split_by_headings(text):
        for piece in _split_text(sec_text, chunk_size, overlap):
            raw.append(Chunk(text=piece, source=source, heading=heading))

    merged: list[Chunk] = []
    for ch in raw:
        if merged and len(ch.text) < MIN_CHUNK_CHARS:
            prev = merged[-1]
            merged[-1] = Chunk(
                text=(prev.text + "\n\n" + ch.text).strip(),
                source=prev.source,
                heading=prev.heading or ch.heading,
            )
        else:
            merged.append(ch)
    return merged


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Recursive splitter: tries paragraph → line → word → character boundaries.

    Respects the markdown structure produced by pymupdf4llm (\\n\\n paragraph
    breaks, \\n line breaks) before falling back to word and character splits.
    """
    if not text.strip():
        return []
    return _recursive_split(text.strip(), chunk_size, overlap, ["\n\n", "\n", " "])


def _recursive_split(
    text: str,
    chunk_size: int,
    overlap: int,
    separators: list[str],
) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    sep = separators[0] if separators else None

    if not sep or sep not in text:
        if len(separators) > 1:
            return _recursive_split(text, chunk_size, overlap, separators[1:])
        return _char_split(text, chunk_size, overlap)

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for part in text.split(sep):
        part_len = len(part) + len(sep)
        if current_len + part_len > chunk_size and current_parts:
            chunk = sep.join(current_parts).strip()
            if chunk:
                if len(chunk) > chunk_size and len(separators) > 1:
                    chunks.extend(_recursive_split(chunk, chunk_size, overlap, separators[1:]))
                else:
                    chunks.append(chunk)
            overlap_parts: list[str] = []
            overlap_len = 0
            for p in reversed(current_parts):
                cost = len(p) + len(sep)
                if overlap_len + cost <= overlap:
                    overlap_parts.insert(0, p)
                    overlap_len += cost
                else:
                    break
            current_parts = overlap_parts
            current_len = overlap_len

        current_parts.append(part)
        current_len += part_len

    if current_parts:
        chunk = sep.join(current_parts).strip()
        if chunk:
            if len(chunk) > chunk_size and len(separators) > 1:
                chunks.extend(_recursive_split(chunk, chunk_size, overlap, separators[1:]))
            else:
                chunks.append(chunk)

    return chunks


def _char_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Character-based fallback when no separator is found."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks
