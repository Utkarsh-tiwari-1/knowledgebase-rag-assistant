from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)


class Source(BaseModel):
    filename: str
    chunk: int
    distance: float
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    abstained: bool


class IngestResponse(BaseModel):
    filename: str
    chunks_added: int
    document_id: str


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    chunks: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]


class DeleteResponse(BaseModel):
    document_id: str
    deleted: bool


class HealthResponse(BaseModel):
    status: str
    collection: str
    indexed_chunks: int
