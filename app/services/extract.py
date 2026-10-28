from io import BytesIO
from pathlib import Path

from docx import Document as DocxDocument
from fastapi import HTTPException
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


def extract_text(filename: str, content: bytes) -> str:
    """Extract text from a supported upload, rejecting empty or unsupported files."""
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Supported file types: .txt, .md, .pdf, .docx")

    try:
        if extension in {".txt", ".md"}:
            text = content.decode("utf-8-sig")
        elif extension == ".pdf":
            reader = PdfReader(BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            document = DocxDocument(BytesIO(content))
            text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Could not read the uploaded document") from exc

    text = text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="The document contains no extractable text")
    return text
