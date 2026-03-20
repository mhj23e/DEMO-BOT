import streamlit as st
import ollama
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load model once
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return embedding_model

# Load and chunk documents
@st.cache_data
def load_documents():
    try:
        with open('data.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        # Simple chunking
        chunks = [text[i:i+1000] for i in range(0, len(text), 800)]
        return chunks
    except FileNotFoundError:
        return ["Demo ready! Add your content to data.txt"]

# Embed and build index
@st.cache_resource
def build_index():
    model = load_models()
    chunks = load_documents()
    if not chunks:
        return None, None, []
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype('float32'))
    return index, model, chunks

# RAG query
def query_rag(question):
    index, model, chunks = build_index()
    if index is None or model is None:
        return "Add content to data.txt first!"
    
    query_emb = model.encode([question])
    faiss.normalize_L2(query_emb)
    scores, indices = index.search(query_emb.astype('float32'), k=min(3, len(chunks)))
    context_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    return "\n\n---\n\n".join(context_chunks)

# Streamlit UI
st.title("🧠 Ephemeral RAG Demo")
st.info("💡 Sessions auto-delete. Data stays local. Edit data.txt to add your docs.")

# Show messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about your data..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context = query_rag(prompt)
            response = ollama.chat(
                model='gemma2:2b',
                messages=[
                    {'role': 'system', 'content': f"Answer using ONLY this context (don't hallucinate):\n\n{context}"},
                    {'role': 'user', 'content': prompt}
                ]
            )
            answer = response['message']['content']
        
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# Sidebar info
with st.sidebar:
    st.header("Quick Start")
    st.markdown("""
    1. Edit `data.txt` with your docs
    2. Refresh page
    3. Ask questions!
    
    **Test:** "What's this demo about?"
    """)
