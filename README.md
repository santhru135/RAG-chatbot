# RAG Chatbot – Document-Based Information Retrieval

## Overview
This project implements a **Retrieval-Augmented Generation (RAG) chatbot** that can answer questions about your uploaded documents using AI-powered search. It retrieves relevant information from a document-based knowledge base and generates context-aware responses using a Large Language Model (LLM). All processing is performed entirely **in-memory** for fast, real-time responses.

The chatbot supports PDF, DOCX, TXT, and CSV files as input.

## Features
- Upload multiple documents and perform AI-powered Q&A.  
- In-memory vector storage using **FAISS** for fast document similarity search.  
- **LLM integration** with NVIDIA NIM and Gemma-3n-e4b-it for context-aware responses.  
- Strict rules to prevent hallucination and ensure factual answers.  
- Chat history support with the ability to clear it anytime.  
- Supports PDFs, Word documents, text files, and CSVs.  

## Technologies Used
- Python  
- Streamlit (Web interface)  
- LangChain (RAG pipeline)  
- FAISS (In-memory vector store)  
- NVIDIA NIM + Gemma-3n-e4b-it (LLM inference)  
- HuggingFace Embeddings (sentence-transformers/all-MiniLM-L6-v2)  
- Pandas (for CSV processing)  
- python-docx, pypdf (document parsing)  

## Getting Started
1. Clone the repository:
```bash
git clone <your-repo-link>

    Install dependencies:

pip install -r requirements.txt

    Set your NVIDIA API key in a .env file:

NVIDIA_API_KEY=your_api_key_here

    Run the Streamlit app:

streamlit run app.py

Usage

    Open the app in your browser.

    Upload your documents (PDF, DOCX, TXT, CSV).

    Type your question in the chat input.

    The chatbot retrieves relevant content from the uploaded documents and generates answers.

    Clear chat history using the Clear History button in the sidebar if needed.

File Support

    PDF: Extracts text from pages.

    DOCX: Extracts text from paragraphs.

    TXT: Reads plain text files.

    CSV: Converts rows into readable text format for Q&A.

Future Improvements

    Support for larger knowledge bases or multiple users.

    Multi-modal input support (images, tables, etc.).

    Enhanced document parsing for better accuracy.