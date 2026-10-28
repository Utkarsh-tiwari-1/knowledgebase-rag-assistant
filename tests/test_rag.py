from types import SimpleNamespace

from app.config import Settings
from app.services.rag import RAGService


class FakeStore:
    def __init__(self, matches):
        self.matches = matches

    def search(self, question, top_k):
        return self.matches[:top_k]


class FakeChat:
    def __init__(self, content="Answer [1]"):
        self.content = content
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


def test_abstains_when_no_relevant_chunks():
    service = RAGService(Settings(openai_api_key="test"), FakeStore([]), FakeChat())
    result = service.answer("What is this?")
    assert result.abstained is True
    assert result.sources == []


def test_returns_sources_for_retrieved_answer():
    match = {
        "text": "The project uses ChromaDB.",
        "metadata": {"filename": "readme.md", "chunk_index": 0},
        "distance": 0.2,
    }
    service = RAGService(Settings(openai_api_key="test"), FakeStore([match]), FakeChat())
    result = service.answer("Which vector database?")
    assert result.abstained is False
    assert result.sources[0].filename == "readme.md"
    assert result.answer == "Answer [1]"


def test_filters_distant_matches_and_abstains():
    match = {
        "text": "unrelated",
        "metadata": {"filename": "notes.txt", "chunk_index": 0},
        "distance": 1.1,
    }
    service = RAGService(Settings(openai_api_key="test", max_distance=0.7), FakeStore([match]), FakeChat())
    result = service.answer("Question")
    assert result.abstained is True
