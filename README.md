# DEMO-BOT

A Streamlit RAG demo that indexes TXT, PDF, and web pages with Sentence
Transformers and FAISS, then generates answers with local Ollama.

[![Tests](https://github.com/mhj23e/DEMO-BOT/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/mhj23e/DEMO-BOT/actions/workflows/tests.yml)

## Run locally

Use Python 3.12. Install and start Ollama, then run:

```sh
ollama pull gemma2:2b
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Put synthetic TXT or PDF files in `knowledge_base/`, or use the included
`data.txt` to try a question. The embedding model downloads on first use.

## Run tests

```sh
python -m pip install -r requirements-test.txt
python -m pytest -q
```

The test suite uses real FAISS, deterministic embeddings, and mocked web
responses. See [TEST_PLAN.md](TEST_PLAN.md) for pass criteria.
