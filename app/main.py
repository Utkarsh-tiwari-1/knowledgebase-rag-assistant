import hashlib
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from openai import OpenAI

from app.config import Settings, get_settings
from app.schemas import (
    DeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.services.extract import extract_text
from app.services.rag import RAGService
from app.services.vector_store import VectorStore


@lru_cache
def build_services() -> tuple[Settings, VectorStore, RAGService]:
    settings = get_settings()
    api_key = settings.openai_api_key.get_secret_value()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    client = OpenAI(api_key=api_key)
    store = VectorStore(settings, client)
    return settings, store, RAGService(settings, store, client)


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.api_title,
        description="Upload documents, retrieve relevant passages, and ask grounded questions.",
        version="1.0.0",
    )

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        try:
            _, store, _ = build_services()
            return HealthResponse(
                status="ok", collection=store.collection.name, indexed_chunks=store.count()
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Service is not configured or available") from exc

    @application.post("/documents", response_model=IngestResponse, status_code=201, tags=["documents"])
    async def upload_document(file: UploadFile = File(...)) -> IngestResponse:
        filename = Path(file.filename or "").name
        content = await file.read()
        if not filename:
            raise HTTPException(status_code=400, detail="A filename is required")
        max_bytes = get_settings().max_upload_mb * 1024 * 1024
        if len(content) > max_bytes:
            raise HTTPException(status_code=413, detail=f"File exceeds {get_settings().max_upload_mb} MB")
        text = extract_text(filename, content)
        document_id = hashlib.sha256(content).hexdigest()[:24]
        try:
            _, store, _ = build_services()
            count = store.add_document(document_id, filename, text)
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Document indexing failed") from exc
        return IngestResponse(filename=filename, chunks_added=count, document_id=document_id)

    @application.post("/query", response_model=QueryResponse, tags=["query"])
    def query_documents(request: QueryRequest) -> QueryResponse:
        try:
            settings, _, rag = build_services()
            top_k = request.top_k or settings.default_top_k
            if top_k > settings.max_top_k:
                raise HTTPException(status_code=422, detail=f"top_k cannot exceed {settings.max_top_k}")
            return rag.answer(request.question, top_k)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Question answering failed") from exc

    @application.get("/documents", response_model=DocumentListResponse, tags=["documents"])
    def list_documents() -> DocumentListResponse:
        try:
            _, store, _ = build_services()
            return DocumentListResponse(documents=[DocumentInfo(**item) for item in store.list_documents()])
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Document index is unavailable") from exc

    @application.delete("/documents/{document_id}", response_model=DeleteResponse, tags=["documents"])
    def delete_document(document_id: str) -> DeleteResponse:
        try:
            _, store, _ = build_services()
            deleted = store.delete_document(document_id)
            if not deleted:
                raise HTTPException(status_code=404, detail="Document not found")
            return DeleteResponse(document_id=document_id, deleted=True)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Document index is unavailable") from exc

    return application


app = create_app()
