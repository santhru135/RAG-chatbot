"""
ui.py — All Streamlit UI rendering for the RAG chatbot.

Handles: CSS injection, session state init, sidebar, header,
empty state, chat history display, and footer.
"""

import streamlit as st
from rag_logic import get_file_icon, format_file_size


# ═══════════════════════════════════════════════════════════════
# CUSTOM CSS — DARK MODE, MODERN SAAS DESIGN
# ═══════════════════════════════════════════════════════════════
def inject_custom_css():
    """Inject the full custom CSS for dark-mode SaaS styling."""
    st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-tertiary: #1c2333;
    --bg-card: #1e2536;
    --accent: #58a6ff;
    --accent-hover: #79b8ff;
    --accent-glow: rgba(88, 166, 255, 0.15);
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --text-muted: #6e7681;
    --border: #30363d;
    --border-light: #21262d;
    --success: #3fb950;
    --warning: #d29922;
    --error: #f85149;
    --user-bubble: #1a3a5c;
    --assistant-bubble: #1c2333;
    --radius: 12px;
    --radius-lg: 16px;
    --shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.2);
}

/* ── Global Overrides ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1923 0%, #111927 50%, #0d1117 100%) !important;
    border-right: 1px solid var(--border) !important;
    padding-top: 0 !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdown"] p,
[data-testid="stSidebar"] [data-testid="stMarkdown"] li,
[data-testid="stSidebar"] label {
    color: var(--text-secondary) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Sidebar Logo Area ── */
.sidebar-logo {
    text-align: center;
    padding: 28px 16px 20px 16px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
}

.sidebar-logo .logo-icon {
    font-size: 44px;
    margin-bottom: 8px;
    display: block;
    filter: drop-shadow(0 0 12px rgba(88, 166, 255, 0.4));
}

.sidebar-logo .logo-title {
    font-size: 18px;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.3px;
    margin: 0;
    line-height: 1.3;
}

.sidebar-logo .logo-sub {
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 4px;
}

/* ── Status Badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    margin-top: 12px;
}

.status-ready {
    background: rgba(63, 185, 80, 0.12);
    color: var(--success);
    border: 1px solid rgba(63, 185, 80, 0.25);
}

.status-processing {
    background: rgba(210, 153, 34, 0.12);
    color: var(--warning);
    border: 1px solid rgba(210, 153, 34, 0.25);
    animation: pulse-badge 1.5s ease-in-out infinite;
}

@keyframes pulse-badge {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

/* ── Section Labels ── */
.section-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin: 24px 0 10px 0;
    padding: 0 4px;
}

/* ── File List ── */
.file-list-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 8px;
    transition: all 0.2s ease;
}

.file-list-item:hover {
    border-color: var(--accent);
    background: var(--bg-card);
}

.file-icon {
    font-size: 18px;
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 8px;
    background: rgba(88, 166, 255, 0.1);
    flex-shrink: 0;
}

.file-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.file-size {
    font-size: 11px;
    color: var(--text-muted);
    margin-left: auto;
    flex-shrink: 0;
}

/* ── Main Header ── */
.main-header {
    text-align: center;
    padding: 32px 20px 24px 20px;
    margin-bottom: 8px;
}

.main-header h1 {
    font-size: 32px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #58a6ff 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.main-header .subtitle {
    font-size: 14px;
    color: var(--text-secondary);
    margin-top: 6px;
    font-weight: 400;
}

/* ── Empty State ── */
.empty-state {
    text-align: center;
    padding: 80px 40px;
    max-width: 480px;
    margin: 40px auto;
}

.empty-state .empty-icon {
    font-size: 64px;
    margin-bottom: 20px;
    display: block;
    opacity: 0.6;
    filter: drop-shadow(0 0 20px rgba(88, 166, 255, 0.25));
}

.empty-state h3 {
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 10px 0;
}

.empty-state p {
    font-size: 14px;
    color: var(--text-secondary);
    line-height: 1.7;
    margin: 0;
}

.empty-state .steps {
    text-align: left;
    display: inline-block;
    margin-top: 24px;
    padding: 20px 28px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
}

.empty-state .steps .step {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0;
    font-size: 14px;
    color: var(--text-secondary);
}

.empty-state .steps .step-num {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: var(--accent-glow);
    border: 1px solid rgba(88, 166, 255, 0.3);
    color: var(--accent);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 600;
    flex-shrink: 0;
}

/* ── Chat Messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 6px 0 !important;
    max-width: 900px;
    margin: 0 auto;
}

/* ── User Messages ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] {
    background: linear-gradient(135deg, #1a3a5c, #1e3d5f) !important;
    border: 1px solid rgba(88, 166, 255, 0.15) !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 14px 18px !important;
    color: var(--text-primary) !important;
    box-shadow: var(--shadow-sm);
}

/* ── Assistant Messages ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 14px 18px !important;
    color: var(--text-primary) !important;
    box-shadow: var(--shadow-sm);
}

/* ── Chat Avatars ── */
[data-testid="chatAvatarIcon-user"] {
    background: linear-gradient(135deg, #58a6ff, #a78bfa) !important;
    border-radius: 50% !important;
}

[data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, #1c2333, #2d3548) !important;
    border: 1px solid var(--border) !important;
    border-radius: 50% !important;
}

/* ── Chat Input ── */
[data-testid="stChatInput"] {
    max-width: 900px;
    margin: 0 auto;
}

[data-testid="stChatInput"] textarea {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    padding: 14px 18px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
    outline: none !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
}

[data-testid="stChatInput"] button {
    background: var(--accent) !important;
    border: none !important;
    border-radius: 12px !important;
    transition: background 0.2s ease;
}

[data-testid="stChatInput"] button:hover {
    background: var(--accent-hover) !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
}

[data-testid="stFileUploader"] > div {
    background: transparent !important;
}

[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
    background: rgba(88, 166, 255, 0.04) !important;
    border: 2px dashed rgba(88, 166, 255, 0.2) !important;
    border-radius: var(--radius) !important;
    padding: 24px !important;
    transition: all 0.25s ease;
}

[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent) !important;
    background: rgba(88, 166, 255, 0.08) !important;
}

[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] span {
    color: var(--text-secondary) !important;
    font-size: 13px !important;
}

[data-testid="stFileUploader"] small {
    color: var(--text-muted) !important;
}

/* ── Buttons ── */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: var(--error) !important;
    border: 1px solid rgba(248, 81, 73, 0.3) !important;
    border-radius: var(--radius) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 8px 0 !important;
    width: 100%;
    transition: all 0.2s ease;
    margin-top: 12px;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(248, 81, 73, 0.1) !important;
    border-color: var(--error) !important;
}

/* ── Expanders (Source References) ── */
[data-testid="stExpander"] {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    margin-top: 8px;
    max-width: 900px;
    margin-left: auto;
    margin-right: auto;
}

[data-testid="stExpander"] summary {
    color: var(--text-secondary) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

[data-testid="stExpander"] summary:hover {
    color: var(--accent) !important;
}

[data-testid="stExpander"] [data-testid="stMarkdownContainer"] {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
    box-shadow: none !important;
}

/* ── Source Chip ── */
.source-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 12px;
    color: var(--text-secondary);
    margin: 4px 4px 4px 0;
}

.source-chip .src-icon {
    color: var(--accent);
}

/* ── Alerts / Toasts ── */
[data-testid="stAlert"] {
    border-radius: var(--radius) !important;
    font-family: 'Inter', sans-serif !important;
    border: none !important;
    box-shadow: var(--shadow-sm);
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: var(--accent) !important;
}

/* ── Typing Indicator ── */
.typing-indicator {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 12px 18px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 18px 18px 18px 4px;
    max-width: 900px;
    margin: 0 auto;
}

.typing-indicator .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--text-muted);
    animation: typing-bounce 1.4s infinite ease-in-out;
}

.typing-indicator .dot:nth-child(1) { animation-delay: 0s; }
.typing-indicator .dot:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator .dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes typing-bounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
    30% { transform: translateY(-6px); opacity: 1; }
}

/* ── Dividers ── */
hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 16px 0;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 24px 0 16px 0;
    color: var(--text-muted);
    font-size: 12px;
    letter-spacing: 0.3px;
    border-top: 1px solid var(--border);
    margin-top: 40px;
    max-width: 900px;
    margin-left: auto;
    margin-right: auto;
}

.app-footer a {
    color: var(--accent);
    text-decoration: none;
}

/* ── Scrollbar ── */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-track {
    background: var(--bg-primary);
}
::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--text-muted);
}

/* ── Hide default decorations ── */
[data-testid="stDecoration"] {
    display: none !important;
}

[data-testid="stToolbar"] {
    display: none !important;
}

/* ── Block container max width ── */
.block-container {
    max-width: 960px !important;
    padding-top: 0 !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════
def init_session_state():
    """Set up all required session state keys with default values."""
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []          # list of (role, message, sources)
    if "uploaded_files_info" not in st.session_state:
        st.session_state.uploaded_files_info = []    # list of {name, size}
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "processed_file_keys" not in st.session_state:
        st.session_state.processed_file_keys = set()


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
def render_sidebar():
    """
    Render the sidebar: logo, status badge, file uploader, file list,
    and clear-chat button. Returns the list of uploaded file objects.
    """
    with st.sidebar:
        # ── Logo & Title ──
        st.markdown("""
        <div class="sidebar-logo">
            <span class="logo-icon">🧿</span>
            <p class="logo-title">Document Intelligence AI</p>
            <div class="logo-sub">Enterprise Knowledge Search</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Status Badge ──
        if st.session_state.processing:
            st.markdown("""
            <div style="text-align:center;">
                <span class="status-badge status-processing">⏳ Processing</span>
            </div>
            """, unsafe_allow_html=True)
        elif st.session_state.vectorstore is not None:
            st.markdown("""
            <div style="text-align:center;">
                <span class="status-badge status-ready">🟢 Ready</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;">
                <span class="status-badge status-ready">⚪ Awaiting Documents</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")  # spacer

        # ── File Upload ──
        st.markdown('<div class="section-label">📂 Upload Documents</div>', unsafe_allow_html=True)
        files = st.file_uploader(
            "Drag and drop files here",
            type=["pdf", "docx", "txt", "csv"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        # ── Uploaded File List ──
        if st.session_state.uploaded_files_info:
            st.markdown('<div class="section-label">📑 Loaded Documents</div>', unsafe_allow_html=True)
            for finfo in st.session_state.uploaded_files_info:
                icon = get_file_icon(finfo["name"])
                size = format_file_size(finfo["size"])
                st.markdown(f"""
                <div class="file-list-item">
                    <div class="file-icon">{icon}</div>
                    <div class="file-name">{finfo["name"]}</div>
                    <div class="file-size">{size}</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Clear Button ──
        st.markdown("<br>" * 2, unsafe_allow_html=True)
        if st.button("🗑️  Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    return files


# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
def render_header():
    """Render the main area header with title and subtitle."""
    st.markdown("""
    <div class="main-header">
        <h1>AI Document Assistant</h1>
        <div class="subtitle">Powered by RAG and LLM  ·  Ask anything about your uploaded documents</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# EMPTY STATE
# ═══════════════════════════════════════════════════════════════
def render_empty_state():
    """Render the empty-state screen when no documents are loaded."""
    if st.session_state.vectorstore is None and not st.session_state.chat_history:
        st.markdown("""
        <div class="empty-state">
            <span class="empty-icon">📂</span>
            <h3>No documents loaded yet</h3>
            <p>Upload your documents in the sidebar to get started. I can answer questions about PDFs, Word docs, text files, and CSVs.</p>
            <div class="steps">
                <div class="step">
                    <span class="step-num">1</span>
                    Upload documents using the sidebar
                </div>
                <div class="step">
                    <span class="step-num">2</span>
                    Wait for processing to complete
                </div>
                <div class="step">
                    <span class="step-num">3</span>
                    Ask any question about your documents
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# CHAT HISTORY
# ═══════════════════════════════════════════════════════════════
def render_source_chips(sources):
    """Build HTML for source-reference chips."""
    chips = ""
    for src in sources:
        chips += (
            f'<span class="source-chip">'
            f'<span class="src-icon">📄</span> '
            f'{src["source"]} · chunk {src["chunk"]}'
            f'</span>'
        )
    return chips


def render_chat_history():
    """Display all past chat messages with source expanders."""
    for entry in st.session_state.chat_history:
        role, msg = entry[0], entry[1]
        sources = entry[2] if len(entry) > 2 else []

        with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🧿"):
            st.markdown(msg)

        # Show source references for assistant messages
        if role == "assistant" and sources:
            with st.expander(f"📚 Sources ({len(sources)} references)"):
                st.markdown(render_source_chips(sources), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════
def render_footer():
    """Render the page footer."""
    st.markdown("""
    <div class="app-footer">
        Built by <strong>Santhru Mohan</strong> · Powered by LangChain, FAISS & NVIDIA AI
    </div>
    """, unsafe_allow_html=True)
