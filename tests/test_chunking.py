import pytest
from app.services.chunking import chunk_text


def test_chunking_produces_overlapping_chunks():
    text = "Sentence one. " * 80
    chunks = chunk_text(text, max_chars=300, overlap_chars=50)
    assert len(chunks) > 2
    assert all(len(x["content"]) <= 300 for x in chunks)
    assert set(chunks[0]) == {"id", "content"}


def test_chunking_empty_text_returns_empty():
    assert chunk_text("  \n ") == []


def test_chunking_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello", max_chars=100, overlap_chars=100)
