from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, quote, urlparse

import faiss
import requests
import streamlit as st
from bs4 import BeautifulSoup
from pypdf import PdfReader
from pypdf.errors import PyPdfError

DATA_FILE = Path("data.txt")
KNOWLEDGE_DIR = Path("knowledge_base")
SUPPORTED_EXTENSIONS = {".txt", ".pdf"}

DEFAULT_HTTP_HEADERS = {
    # Wikimedia (including Wikipedia) may return 403 if a User-Agent is missing/vague.
    # Use a stable, descriptive UA; adjust the name/version as you like.
    "User-Agent": "DEMO-BOT/1.0 (+local streamlit rag demo; requests)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


@st.cache_resource
def load_models():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text(text, source_name, chunk_size=1000, overlap=200):
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    step = max(chunk_size - overlap, 1)
    chunks = []
    for start in range(0, len(cleaned), step):
        chunk = cleaned[start : start + chunk_size]
        if chunk.strip():
            chunks.append({"text": chunk, "source": source_name})
    return chunks


def read_text_file(path):
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(path):
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def extract_main_text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    for selector in ("main", "article", "[role='main']", ".wiki-content", ".documentation"):
        node = soup.select_one(selector)
        if node:
            return node.get_text(separator=" ", strip=True)

    return soup.get_text(separator=" ", strip=True)


def sanitize_pasted_url(url: str) -> str:
    url = (url or "").strip()
    # Common copy/paste artifacts from chat/markdown: trailing punctuation.
    while url.endswith((".", ",", ";", ":")):
        url = url[:-1]

    # Only strip a trailing bracket if we don't see a matching opener anywhere.
    if url.endswith(")") and "(" not in url:
        url = url[:-1]
    if url.endswith("]") and "[" not in url:
        url = url[:-1]
    if url.endswith("}") and "{" not in url:
        url = url[:-1]

    return url


def _wikipedia_rest_html_url(page_url: str) -> Optional[str]:
    """
    Convert a normal Wikipedia page URL into the REST endpoint that returns HTML:
    https://{lang}.wikipedia.org/api/rest_v1/page/html/{title}
    """
    parsed = urlparse(page_url)
    host = (parsed.netloc or "").lower()
    if not host.endswith("wikipedia.org"):
        return None

    title = None
    if parsed.path.startswith("/wiki/"):
        title = parsed.path[len("/wiki/") :]
    else:
        qs = parse_qs(parsed.query or "")
        title = (qs.get("title") or [None])[0]

    if not title:
        return None

    lang_host = host  # preserves en.wikipedia.org, etc.
    return f"https://{lang_host}/api/rest_v1/page/html/{quote(title, safe='')}"


def fetch_wiki_text(url):
    url = sanitize_pasted_url(url)
    response = requests.get(url, timeout=20, headers=DEFAULT_HTTP_HEADERS)
    if response.status_code in (403, 429):
        # Wikipedia in particular is happier when accessed via its REST API.
        rest_url = _wikipedia_rest_html_url(url)
        if rest_url:
            response = requests.get(rest_url, timeout=20, headers=DEFAULT_HTTP_HEADERS)

    response.raise_for_status()
    return extract_main_text_from_html(response.text)


def get_local_source_signature():
    files = []
    if DATA_FILE.exists():
        files.append(f"{DATA_FILE}:{DATA_FILE.stat().st_mtime_ns}")

    if KNOWLEDGE_DIR.exists():
        for path in sorted(KNOWLEDGE_DIR.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(f"{path}:{path.stat().st_mtime_ns}")

    return tuple(files)


@st.cache_data(show_spinner=False)
def load_documents(file_signature, wiki_urls):
    del file_signature

    documents = []

    failed_sources = []
    local_files = [DATA_FILE] if DATA_FILE.exists() else []
    local_files.extend(
        path for path in sorted(KNOWLEDGE_DIR.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    for path in local_files:
        source = str(path)
        try:
            reader = read_pdf_file if path.suffix.lower() == ".pdf" else read_text_file
            documents.extend(chunk_text(reader(path), source))
        except (OSError, PyPdfError) as exc:
            failed_sources.append(f"{source} ({exc})")

    for url in wiki_urls:
        try:
            wiki_text = fetch_wiki_text(url)
            documents.extend(chunk_text(wiki_text, url))
        except Exception as exc:
            failed_sources.append(f"{url} ({exc})")

    return documents, failed_sources


@st.cache_resource(show_spinner=False)
def build_index(file_signature, wiki_urls):
    documents, failed_sources = load_documents(file_signature, wiki_urls)
    if not documents:
        return None, None, [], failed_sources

    model = load_models()
    chunk_texts = [doc["text"] for doc in documents]
    embeddings = model.encode(chunk_texts)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype("float32"))
    return index, model, documents, failed_sources


def query_rag(question, wiki_urls):
    file_signature = get_local_source_signature()
    index, model, documents, failed_sources = build_index(file_signature, tuple(wiki_urls))
    if index is None or model is None:
        return "No indexed content yet. Add TXT/PDF files or wiki URLs first.", [], failed_sources

    query_emb = model.encode([question])
    faiss.normalize_L2(query_emb)
    scores, indices = index.search(query_emb.astype("float32"), k=min(4, len(documents)))

    selected_docs = []
    for i in indices[0]:
        if 0 <= i < len(documents):
            selected_docs.append(documents[i])

    context = "\n\n---\n\n".join(doc["text"] for doc in selected_docs)
    sources = [doc["source"] for doc in selected_docs]
    return context, sources, failed_sources


def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "wiki_urls" not in st.session_state:
        st.session_state.wiki_urls = ""



    st.title("Enterprise RAG Demo")
    st.info("Sessions auto-delete. Data stays local. Index TXT, PDF, and enterprise wiki/web pages.")

    with st.sidebar:
        st.header("Knowledge Sources")
        st.markdown(
            """
    1. Put `.txt` and `.pdf` files into `knowledge_base/`
    2. Or keep using `data.txt`
    3. Paste wiki page URLs below
            """
        )
        st.session_state.wiki_urls = st.text_area(
            "Wiki or intranet URLs",
            value=st.session_state.wiki_urls,
            height=140,
            placeholder="https://wiki.company.com/page-one\nhttps://wiki.company.com/page-two",
        )

        KNOWLEDGE_DIR.mkdir(exist_ok=True)
        local_files = []
        if DATA_FILE.exists():
            local_files.append(DATA_FILE.name)
        local_files.extend(
            str(path)
            for path in sorted(KNOWLEDGE_DIR.rglob("*"))
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        st.caption(f"Indexed local files: {len(local_files)}")
        if local_files:
            st.code("\n".join(local_files), language="text")
        else:
            st.caption("No local TXT/PDF files found yet.")


    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    if prompt := st.chat_input("Ask about your knowledge base..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        wiki_urls = [line.strip() for line in st.session_state.wiki_urls.splitlines() if line.strip()]

        with st.chat_message("assistant"):
            with st.spinner("Searching sources..."):
                context, sources, failed_sources = query_rag(prompt, wiki_urls)
                import ollama

                response = ollama.chat(
                    model="gemma2:2b",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Answer using only the provided context. "
                                "If the answer is not present, say you could not find it in the indexed sources.\n\n"
                                f"Context:\n{context}"
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                answer = response["message"]["content"]

            st.markdown(answer)
            if sources:
                st.caption("Sources: " + ", ".join(dict.fromkeys(sources)))
            if failed_sources:
                st.warning("Some sources could not be indexed:\n- " + "\n- ".join(failed_sources))

            st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
