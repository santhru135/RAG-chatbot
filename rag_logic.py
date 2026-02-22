"""
rag_logic.py — Backend logic for the RAG chatbot.

Handles: embedding initialization, file text extraction, vectorstore
management, LLM chain creation, and document retrieval.
"""

import os
import streamlit as st
import pandas as pd
from docx import Document
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_nvidia import ChatNVIDIA
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"
GEMMA_MODEL_NAME = "google/gemma-3n-e4b-it"

FILE_ICONS = {
    "pdf": "📕",
    "docx": "📘",
    "txt": "📄",
    "csv": "📊",
}


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def get_file_icon(filename: str) -> str:
    """Return an emoji icon based on file extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return FILE_ICONS.get(ext, "📎")


def format_file_size(size_bytes: int) -> str:
    """Format byte count into a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


# ═══════════════════════════════════════════════════════════════
# EMBEDDING + TEXT SPLITTER (cached by Streamlit)
# ═══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_embedding_and_splitter():
    """Load embedding model and text splitter once; cached across reruns."""
    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return embedding, splitter


# Module-level references used by other functions
embedding, text_splitter = get_embedding_and_splitter()


# ═══════════════════════════════════════════════════════════════
# FILE TEXT EXTRACTORS
# ═══════════════════════════════════════════════════════════════
def extract_text_from_pdf(file):
    """Extract text from a PDF file object."""
    try:
        reader = PdfReader(file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    except:
        return ""


def extract_text_from_docx(file_obj):
    """Extract text from a DOCX file object."""
    try:
        doc = Document(file_obj)
        return "\n".join(p.text for p in doc.paragraphs)
    except:
        return ""


def extract_text_from_txt(file_obj):
    """Extract text from a plain text file object."""
    try:
        raw = file_obj.read()
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="ignore")
        return str(raw)
    except:
        return ""


def extract_text_from_csv(file_obj):
    """Extract text from a CSV file by joining all rows."""
    try:
        df = pd.read_csv(file_obj)
        rows = []
        for _, row in df.iterrows():
            rows.append(" | ".join([f"{c}: {row[c]}" for c in df.columns]))
        return "\n".join(rows)
    except:
        return ""


def extract_text(uploaded):
    """Route to the correct extractor based on MIME type."""
    if uploaded.type == "application/pdf":
        return extract_text_from_pdf(uploaded)
    if uploaded.type == "text/plain":
        return extract_text_from_txt(uploaded)
    if uploaded.type == "text/csv":
        return extract_text_from_csv(uploaded)
    if uploaded.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(uploaded)
    try:
        return uploaded.read().decode("utf-8", errors="ignore")
    except:
        return ""


# ═══════════════════════════════════════════════════════════════
# VECTORSTORE
# ═══════════════════════════════════════════════════════════════
def add_document_to_vectorstore(text, name, store):
    """Split text into chunks and add to FAISS vectorstore."""
    chunks = text_splitter.split_text(text)
    metas = [{"source": name, "chunk": i} for i in range(len(chunks))]
    if store is None:
        store = FAISS.from_texts(chunks, embedding, metadatas=metas)
    else:
        store.add_texts(chunks, metadatas=metas)
    return store


def retrieve_documents(vectorstore, query, k=3):
    """Retrieve top-k relevant documents for a query."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.get_relevant_documents(query)


def collect_sources(docs):
    """Extract deduplicated source metadata from retrieved documents."""
    sources = []
    seen = set()
    for d in docs:
        key = f"{d.metadata['source']}_chunk{d.metadata['chunk']}"
        if key not in seen:
            sources.append({"source": d.metadata["source"], "chunk": d.metadata["chunk"]})
            seen.add(key)
    return sources


def build_context(docs):
    """Format retrieved documents into a single context string for the LLM."""
    return "\n\n".join(
        f"[{d.metadata['source']}::chunk{d.metadata['chunk']}] {d.page_content}"
        for d in docs
    )


# ═══════════════════════════════════════════════════════════════
# LLM CHAIN
# ═══════════════════════════════════════════════════════════════
prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant.\n\n"
        "Always follow this order of priority when answering:\n"
        "1. First, answer strictly using the provided context.\n"
        "2. If the context does not contain enough information, then answer using your\n"
        "   own general knowledge — but ONLY if the information is factual,\n"
        "   verified, and widely accepted.\n\n"
        "Rules:\n"
        "- Do NOT guess or make up details.\n"
        "- Do NOT hallucinate any facts.\n"
        "- If you are unsure or the information is not reliable, respond with:\n"
        '  "I\'m not fully sure, but based on my general knowledge: <answer>."\n'
        "- Always keep the answer truthful and relevant to the question.\n\n"
        "Context:\n{context}"
    ),
    ("user", "{question}"),
])


def get_chain():
    """Create and return the LLM chain. Returns None if API key is missing."""
    if not os.environ.get("NVIDIA_API_KEY"):
        st.error("⚠️ NVIDIA_API_KEY is missing! Please add it to your `.env` file.")
        return None
    model = ChatNVIDIA(model=GEMMA_MODEL_NAME)
    return prompt_template | model | StrOutputParser()
