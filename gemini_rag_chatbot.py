import os
import streamlit as st
from operator import itemgetter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_nvidia import ChatNVIDIA
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

os.environ["NVIDIA_API_KEY"] = "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"

VECTOR_CACHE = "vectorstore"

def load_vectorstore():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    try:
        retriever = FAISS.load_local(VECTOR_CACHE, embedding, allow_dangerous_deserialization=True)
        return retriever.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error("❌ Could not load vector store. Please ensure it's built and available.")
        st.stop()

def initialize_chain():
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a friendly, helpful assistant who responds in a casual, human-like tone and answers only based on the given context. If the context does not contain the answer, say 'I don't have that informations'. Keep responses short, warm, and engaging. Avoid sounding robotic or overly formal. Be conversational like you're chatting with a friend:\n<Documents>\n{context}\n</Documents>\nAlso consider the previous conversation:\n<ChatHistory>\n{history}\n</ChatHistory>\nSpeak only in this language: {language}"
        ),
        ("user", "{question}"),
    ])

    model = ChatNVIDIA(model="google/gemma-3n-e4b-it")

    return (prompt | model | StrOutputParser())

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot with Streamlit")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "language" not in st.session_state:
    st.session_state.language = "english"
if "retriever" not in st.session_state:
    st.session_state.retriever = load_vectorstore()
if "chain" not in st.session_state:
    st.session_state.chain = initialize_chain()

with st.sidebar:
    st.header("🛠️ Settings")
    st.session_state.language = st.selectbox("🌐 Select language", ["english", "tamil", "hindi", "french", "spanish"])
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

user_input = st.chat_input("💬 Ask something")

if user_input:
    formatted_history = ""
    for msg in st.session_state.chat_history:
        role = "User" if msg["role"] == "user" else "Bot"
        formatted_history += f"{role}: {msg['content']}\n"

    docs = st.session_state.retriever.get_relevant_documents(user_input)

    if not docs:
        response = "🤔 I don't know. That information isn't in my knowledge base."
    else:
        context = "\n".join([doc.page_content for doc in docs])

        with st.spinner("Thinking..."):
            response = st.session_state.chain.invoke({
                "question": user_input,
                "language": st.session_state.language,
                "history": formatted_history.strip(),
                "context": context
            })

        response = response.strip() if response else "🤔 I don't know."

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "bot", "content": response})

    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        st.markdown(response)