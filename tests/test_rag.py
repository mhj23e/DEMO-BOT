import numpy as np
import pytest
from requests import Response

import app


@pytest.fixture(autouse=True)
def isolated_sources(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    app.load_documents.clear()
    app.build_index.clear()
    yield
    app.load_documents.clear()
    app.build_index.clear()


def test_chunk_text_preserves_source_and_overlap():
    assert app.chunk_text("  abcd\t efgh  ", "policy", chunk_size=5, overlap=2) == [
        {"text": "abcd ", "source": "policy"},
        {"text": "d efg", "source": "policy"},
        {"text": "fgh", "source": "policy"},
    ]


def test_blank_text_produces_no_chunks():
    assert app.chunk_text(" \n\t ", "empty") == []


def test_faiss_retrieval_ranks_by_cosine_similarity(monkeypatch):
    app.KNOWLEDGE_DIR.mkdir()
    (app.KNOWLEDGE_DIR / "a.txt").write_text("office supplies")
    (app.KNOWLEDGE_DIR / "b.txt").write_text("enrollment deadline")

    class FixedEmbeddings:
        def encode(self, texts):
            vectors = {
                "office supplies": [0.3, 0.4],
                "enrollment deadline": [0, 4],
                "When is enrollment?": [0, 8],
            }
            return np.array([vectors[text] for text in texts], dtype="float32")

    monkeypatch.setattr(app, "load_models", FixedEmbeddings)
    context, sources, failures = app.query_rag("When is enrollment?", [])
    assert context.split("\n\n---\n\n") == ["enrollment deadline", "office supplies"]
    assert sources == ["knowledge_base/b.txt", "knowledge_base/a.txt"]
    assert failures == []


def test_corrupt_pdf_is_reported_and_valid_text_is_indexed():
    app.KNOWLEDGE_DIR.mkdir()
    (app.KNOWLEDGE_DIR / "broken.pdf").write_bytes(b"not a pdf")
    (app.KNOWLEDGE_DIR / "policy.txt").write_text("Enrollment opens Monday.")

    documents, failures = app.load_documents(app.get_local_source_signature(), ())
    assert documents == [{"text": "Enrollment opens Monday.", "source": "knowledge_base/policy.txt"}]
    assert len(failures) == 1
    assert "broken.pdf" in failures[0]


def test_unavailable_url_is_reported_and_other_url_is_indexed(monkeypatch):
    bad_url = "https://example.org/missing"
    good_url = "https://example.org/policy"

    def get(url, **kwargs):
        response = Response()
        response.url = url
        response.status_code = 404 if url == bad_url else 200
        response._content = b"<main>Enrollment opens Monday.</main>"
        return response

    monkeypatch.setattr(app.requests, "get", get)
    documents, failures = app.load_documents((), (bad_url, good_url))
    assert documents == [{"text": "Enrollment opens Monday.", "source": good_url}]
    assert len(failures) == 1
    assert bad_url in failures[0]
    assert "404" in failures[0]


def test_empty_index_does_not_load_embedding_model(monkeypatch):
    def unexpected_model_load():
        pytest.fail("The embedding model should only load when sources contain text")

    monkeypatch.setattr(app, "load_models", unexpected_model_load)
    context, sources, failures = app.query_rag("Any policies?", [])
    assert "No indexed content" in context
    assert sources == failures == []
