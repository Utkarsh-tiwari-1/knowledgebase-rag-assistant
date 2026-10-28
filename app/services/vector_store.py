from collections import defaultdict
from pathlib import Path
from typing import Any

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings


class VectorStore:
    def __init__(self, settings: Settings, embedding_client: Any):
        Path(settings.chroma_persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedding_client = embedding_client
        self.embedding_model = settings.openai_embedding_model
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            add_start_index=True,
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = self.embedding_client.embeddings.create(model=self.embedding_model, input=texts)
        return [item.embedding for item in response.data]

    def add_document(self, document_id: str, filename: str, text: str) -> int:
        chunks = self.splitter.create_documents([text])
        if not chunks:
            return 0
        # Replacing an identical upload also removes any old trailing chunks if the text changed.
        existing = self.collection.get(where={"document_id": document_id}, include=[])
        if existing.get("ids"):
            self.collection.delete(ids=existing["ids"])
        ids = [f"{document_id}:{index}" for index in range(len(chunks))]
        embeddings = self._embed([chunk.page_content for chunk in chunks])
        metadata = [
            {
                "document_id": document_id,
                "filename": filename,
                "chunk_index": index,
                "start_index": int(chunk.metadata.get("start_index", 0)),
            }
            for index, chunk in enumerate(chunks)
        ]
        self.collection.upsert(
            ids=ids,
            documents=[chunk.page_content for chunk in chunks],
            metadatas=metadata,
            embeddings=embeddings,
        )
        return len(chunks)

    def search(self, question: str, top_k: int) -> list[dict[str, Any]]:
        if self.collection.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=self._embed([question]),
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        matches: list[dict[str, Any]] = []
        for document, metadata, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            matches.append({"text": document, "metadata": metadata, "distance": float(distance)})
        return matches

    def list_documents(self) -> list[dict[str, Any]]:
        records = self.collection.get(include=["metadatas"])
        grouped: dict[str, dict[str, Any]] = defaultdict(dict)
        for metadata in records.get("metadatas") or []:
            if not metadata:
                continue
            doc_id = metadata["document_id"]
            grouped[doc_id]["document_id"] = doc_id
            grouped[doc_id]["filename"] = metadata["filename"]
            grouped[doc_id]["chunks"] = grouped[doc_id].get("chunks", 0) + 1
        return sorted(grouped.values(), key=lambda item: item["filename"].lower())

    def delete_document(self, document_id: str) -> bool:
        records = self.collection.get(where={"document_id": document_id}, include=[])
        ids = records.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
        return bool(ids)

    def count(self) -> int:
        return self.collection.count()
