# file: app.py
import os
import io
import streamlit as st
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_nvidia import ChatNVIDIA
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from docx import Document
from pypdf import PdfReader
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

st.set_page_config(page_title="RAG Chatbot", page_icon="🧠")
st.title("🧠 RAG Chatbot For Document Search")

# -----------------------
# CONSTANTS
# -----------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"
GEMMA_MODEL_NAME = "google/gemma-3n-e4b-it"

# -----------------------
# INIT embedding + splitter
# -----------------------
@st.cache_resource
def get_embedding_and_splitter():
    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE}
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return embedding, splitter

embedding, text_splitter = get_embedding_and_splitter()

# -----------------------
# MEMORY-ONLY VECTORSTORE (RAM)
# -----------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None   # stays only in RAM

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------
# File extractors
# -----------------------
def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    except:
        return ""

def extract_text_from_docx(file_obj):
    try:
        doc = Document(file_obj)
        return "\n".join(p.text for p in doc.paragraphs)
    except:
        return ""

def extract_text_from_txt(file_obj):
    try:
        raw = file_obj.read()
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="ignore")
        return str(raw)
    except:
        return ""

def extract_text_from_csv(file_obj):
    try:
        df = pd.read_csv(file_obj)
        rows = []
        for _, row in df.iterrows():
            rows.append(" | ".join([f"{c}: {row[c]}" for c in df.columns]))
        return "\n".join(rows)
    except:
        return ""

def extract_text(uploaded):
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

# -----------------------
# Add to vectorstore (RAM only)
# -----------------------
def add_document_to_vectorstore(text, name, store):
    chunks = text_splitter.split_text(text)
    metas = [{"source": name, "chunk": i} for i in range(len(chunks))]

    if store is None:
        store = FAISS.from_texts(chunks, embedding, metadatas=metas)
    else:
        store.add_texts(chunks, metadatas=metas)
    return store

# -----------------------
# Upload UI
# -----------------------
st.subheader("Upload files (Stored only in RAM)")
files = st.file_uploader("Choose files", type=["pdf", "docx", "txt", "csv"], accept_multiple_files=True)

if files:
    for uploaded in files:
        text = extract_text(uploaded)
        if text.strip() == "":
            st.warning(f"Could not extract text from {uploaded.name}")
            continue

        st.session_state.vectorstore = add_document_to_vectorstore(
            text, uploaded.name, st.session_state.vectorstore
        )

    st.success("Files added to memory successfully!")

# -----------------------
# LLM Chain
# -----------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer ONLY using the provided context.\n\n"
        "Context:\n{context}\n\n"
        "If answer is missing, say: 'I don't have that information.'"
    ),
    ("user", "{question}")
])

def get_chain():
    if not os.environ.get("NVIDIA_API_KEY"):
        st.error("NVIDIA_API_KEY missing!")
        return None
    model = ChatNVIDIA(model=GEMMA_MODEL_NAME)
    return (prompt | model | StrOutputParser())

# -----------------------
# Chat
# -----------------------
st.subheader("Ask a question")
query = st.chat_input("Type here...")

if query:
    if st.session_state.vectorstore is None:
        st.error("Upload documents first.")
        st.stop()

    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)

    if not docs:
        answer = "I don't have that information."
    else:
        context = "\n\n".join(
            f"[{d.metadata['source']}::chunk{d.metadata['chunk']}] {d.page_content}"
            for d in docs
        )
        chain = get_chain()
        answer = chain.invoke({"question": query, "context": context}).strip()

    st.session_state.chat_history.append(("user", query))
    st.session_state.chat_history.append(("assistant", answer))

# Show chat history
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

# -----------------------
# Sidebar: Clear memory
# -----------------------
st.sidebar.button("Clear Memory", on_click=lambda: st.session_state.clear())
