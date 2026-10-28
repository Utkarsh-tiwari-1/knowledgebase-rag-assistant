import pytest
from fastapi import HTTPException

from app.services.extract import extract_text


def test_extracts_utf8_text():
    assert extract_text("notes.txt", "Hello RAG".encode()) == "Hello RAG"


def test_rejects_unsupported_extension():
    with pytest.raises(HTTPException) as error:
        extract_text("notes.exe", b"data")
    assert error.value.status_code == 415


def test_rejects_empty_document():
    with pytest.raises(HTTPException) as error:
        extract_text("empty.md", b"  ")
    assert error.value.status_code == 422
