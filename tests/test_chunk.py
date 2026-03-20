from __future__ import annotations

from pathlib import Path

import pytest

from rag.chunk import _char_split, _chunk_basic, chunk_file, list_chunkable_files


def test_list_chunkable_files_respects_method_and_skip_extensions(tmp_path: Path) -> None:
    for name in ["a.txt", "b.md", "c.pdf", "d.docx", ".hidden.txt", "e.png"]:
        (tmp_path / name).write_text("x")

    basic = list_chunkable_files(tmp_path, "basic", {"PDF"})
    unstructured = list_chunkable_files(tmp_path, "unstructured", {".md"})

    assert [path.name for path in basic] == ["a.txt", "b.md"]
    assert [path.name for path in unstructured] == ["a.txt", "c.pdf", "d.docx"]


def test_chunk_file_preserves_headings_for_markdown_sections(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    path.write_text(
        "# Intro\n"
        + ("Alpha line. " * 12)
        + "\n\n"
        "## Details\n"
        + ("Beta line. " * 12)
        + "\n",
        encoding="utf-8",
    )

    chunks = chunk_file(path, chunk_size=1000, overlap=50)

    assert [chunk.heading for chunk in chunks] == ["Intro", "Details"]
    assert chunks[0].source == "doc.md"
    assert "Alpha line." in chunks[0].text
    assert "Beta line." in chunks[1].text


def test_chunk_basic_merges_tiny_trailing_sections() -> None:
    text = "# Intro\n" + ("A" * 140) + "\n\n## Tiny\nshort tail"

    chunks = _chunk_basic(text, chunk_size=1000, overlap=50, source="doc.md")

    assert len(chunks) == 1
    assert "short tail" in chunks[0].text
    assert chunks[0].heading == "Intro"


def test_char_split_respects_overlap() -> None:
    assert _char_split("abcdefghij", chunk_size=4, overlap=1) == ["abcd", "defg", "ghij"]


def test_chunk_file_rejects_unsupported_basic_file_types(tmp_path: Path) -> None:
    path = tmp_path / "deck.pptx"
    path.write_text("slides", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        chunk_file(path)
