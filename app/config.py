from functools import lru_cache

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: SecretStr = SecretStr("")
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    chroma_persist_directory: str = "./chroma_data"
    chroma_collection_name: str = "documents"
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    chunk_overlap: int = Field(default=150, ge=0, le=2000)
    default_top_k: int = Field(default=4, ge=1, le=20)
    max_top_k: int = Field(default=10, ge=1, le=50)
    max_distance: float = Field(default=0.7, ge=0.0, le=2.0)
    max_upload_mb: int = Field(default=20, ge=1, le=200)
    api_title: str = "KnowledgeBase RAG Document Assistant"

    @model_validator(mode="after")
    def validate_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.default_top_k > self.max_top_k:
            raise ValueError("DEFAULT_TOP_K cannot exceed MAX_TOP_K")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
