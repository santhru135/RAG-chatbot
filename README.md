# RAG Chatbot using Firecrawl, FAISS, and NVIDIA NIM (Gemma-3n-e4b-it)
This project is an intelligent Retrieval-Augmented Generation (RAG) chatbot that answers user queries with high accuracy by combining web scraping, vector search, and LLM inference.
Built using Firecrawl for website scraping, FAISS for fast vector similarity search, and NVIDIA NIM with the Gemma-3n-e4b-it model for natural language understanding.

🚀 Features

Automated Website Scraping
Uses Firecrawl to crawl and extract clean text content from any website.

Efficient Vector Indexing
FAISS is used to convert scraped text into embeddings and store them for fast semantic search.

Accurate Response Generation
The chatbot uses NVIDIA NIM’s Gemma-3n-e4b-it model to generate precise and context-aware answers.

Context-Aware Retrieval
Relevant text chunks are retrieved and combined with the user's question to create high-quality responses.

Modular Code Structure
The main logic is inside gemini_rag_chatbot.py, making the project easy to understand and extend.

🧩 How It Works

Scrape
Firecrawl scrapes the target website and returns raw + structured text.

Chunk & Embed
The scraped text is converted into embeddings using the chosen embedding model.

Store in FAISS Index
FAISS stores the embeddings for fast nearest-neighbor search.

Query
When the user asks a question, FAISS returns the top relevant text passages.

Generate Response
The chatbot sends the retrieved text + question to the Gemma model via NVIDIA NIM API.

Final Output
The user receives an accurate, summarized answer with sources.
