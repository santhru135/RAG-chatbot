"""
app.py — Entry point for the RAG chatbot.

Wires together the backend (rag_logic.py) and UI (ui.py).
Run with:  streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

# ── Backend & UI imports ──
from rag_logic import (
    extract_text,
    add_document_to_vectorstore,
    retrieve_documents,
    collect_sources,
    build_context,
    get_chain,
)
from ui import (
    inject_custom_css,
    init_session_state,
    render_sidebar,
    render_header,
    render_empty_state,
    render_chat_history,
    render_source_chips,
    render_footer,
)


# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🧿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Setup ──
inject_custom_css()
init_session_state()


# ═══════════════════════════════════════════════════════════════
# SIDEBAR — file upload & processing
# ═══════════════════════════════════════════════════════════════
files = render_sidebar()

if files:
    new_files = [f for f in files if f.file_id not in st.session_state.processed_file_keys]

    if new_files:
        st.session_state.processing = True
        with st.sidebar:
            with st.spinner("Embedding documents..."):
                for uploaded in new_files:
                    text = extract_text(uploaded)
                    if text.strip() == "":
                        st.warning(f"⚠️ Could not extract text from **{uploaded.name}**")
                        continue

                    st.session_state.vectorstore = add_document_to_vectorstore(
                        text, uploaded.name, st.session_state.vectorstore
                    )
                    st.session_state.uploaded_files_info.append({
                        "name": uploaded.name,
                        "size": uploaded.size,
                    })
                    st.session_state.processed_file_keys.add(uploaded.file_id)

        st.session_state.processing = False
        st.toast("✅ Documents processed successfully!", icon="🎉")
        st.rerun()


# ═══════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════
render_header()
render_empty_state()
render_chat_history()


# ═══════════════════════════════════════════════════════════════
# CHAT INPUT & RESPONSE
# ═══════════════════════════════════════════════════════════════
query = st.chat_input("Ask questions about your documents...")

if query:
    if st.session_state.vectorstore is None:
        st.error("📤 Please upload documents first using the sidebar.")
        st.stop()

    # Display user message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(query)

    # Retrieve relevant documents
    docs = retrieve_documents(st.session_state.vectorstore, query)
    sources = collect_sources(docs)

    if not docs:
        answer = "I don't have enough information in the uploaded documents to answer that question."
        with st.chat_message("assistant", avatar="🧿"):
            st.markdown(answer)
    else:
        context = build_context(docs)
        chain = get_chain()
        if chain is None:
            st.stop()

        # Streaming response
        with st.chat_message("assistant", avatar="🧿"):
            answer = st.write_stream(
                chain.stream({"question": query, "context": context})
            )

    # Show source references
    if sources:
        with st.expander(f"📚 Sources ({len(sources)} references)"):
            st.markdown(render_source_chips(sources), unsafe_allow_html=True)

    # Save to history
    st.session_state.chat_history.append(("user", query, []))
    st.session_state.chat_history.append(("assistant", answer, sources))


# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════
render_footer()