RAG Chatbot – Document-Based Information Retrieval

Overview:
This project implements a Retrieval-Augmented Generation (RAG) chatbot that retrieves information from a document-based knowledge base and generates context-aware responses using a Large Language Model (LLM). All processing is performed entirely in-memory for fast, real-time answers.

Features:

In-memory vector storage using FAISS for fast document similarity search.

LLM integration with NVIDIA NIM and Gemma-3n-e4b-it for intelligent, context-aware responses.

Document-based retrieval ensures accurate answers from structured knowledge sources.

Optimized for low-latency queries, making it suitable for real-time applications.

Technologies Used:
Python, FAISS, NVIDIA NIM, LangChain, Gemma-3n-e4b-it, NLP

Getting Started:

Clone the repository:

git clone <your-repo-link>


Install dependencies:

pip install -r requirements.txt


Run the chatbot:

python app.py


Usage:

Provide documents (PDFs, text files, etc.) as input.

The chatbot processes the documents, stores them in memory, and answers questions based on their content.

Future Improvements:

Support for larger document collections.

Integration with external APIs or web sources for dynamic content.