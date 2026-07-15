"""ChatGPT-style Streamlit frontend for the RAG PDF Chatbot.

This module only implements the user interface. All retrieval-augmented
generation logic (document loading, chunking, embeddings, vector store,
retriever, LLM, memory, logging) lives in `backend.py` and is used here
without changing backend API contracts.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List, Optional, TypedDict

import streamlit as st

from backend import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    ConversationMemory,
    RAGLogger,
    RAGPipeline,
    build_vector_store,
    chunk_documents,
    create_embedding_model,
    create_retriever,
    initialize_llm,
    load_documents,
    reset_session,
)

# ==========================================================
# Constants
# ==========================================================

APP_TITLE: str = "DocQuery AI"
APP_ICON: str = "🤖"
APP_DESCRIPTION: str = "Chat with your documents using RAG & LLMs."
UPLOAD_FOLDER: str = "uploaded_pdfs"
TYPING_DELAY_SECONDS: float = 0.02
NO_SOURCES_TEXT: str = "No source information available."

KEY_CHATS: str = "chats"
KEY_CURRENT_CHAT_ID: str = "current_chat_id"
KEY_CHAT_COUNTER: str = "chat_counter"
KEY_LOGGER: str = "logger"
KEY_KB_READY: str = "knowledge_base_ready"
KEY_RETRIEVER: str = "retriever"
KEY_LLM: str = "llm"
KEY_KB_STATS: str = "kb_stats"
KEY_UPLOADED_PDF_NAMES: str = "uploaded_pdf_names"

CHAT_KEY_TITLE: str = "title"
CHAT_KEY_MESSAGES: str = "messages"
CHAT_KEY_MEMORY: str = "memory"
CHAT_KEY_PIPELINE: str = "pipeline"

ROLE_USER: str = "user"
ROLE_ASSISTANT: str = "assistant"


class ChatMessage(TypedDict, total=False):
    """A rendered chat message entry."""

    role: str
    content: str
    sources: Optional[str]


class ChatEntry(TypedDict):
    """Per-chat session data."""

    title: str
    messages: List[ChatMessage]
    memory: ConversationMemory
    pipeline: Optional[RAGPipeline]


def ss() -> Any:
    """Shorthand accessor for Streamlit session state.

    Returns:
        The Streamlit session state object.
    """
    return st.session_state


# ==========================================================
# Page Configuration
# ==========================================================


def configure_page() -> None:
    """Configure Streamlit page settings.

    This must be called once at app start before rendering widgets.
    """
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ==========================================================
# Custom CSS
# ==========================================================


def inject_custom_css() -> None:
    """Inject custom CSS for a modern ChatGPT-like appearance."""
    st.markdown(
        """
        <style>
        /* ---------- General ---------- */
        html, body, [class*="css"] {
            font-family: "Segoe UI", "Inter", sans-serif;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 6rem;
            max-width: 900px;
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background-color: #171821;
            border-right: 1px solid #2b2d3a;
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.2rem;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #f5f5f7;
        }

        /* ---------- Buttons ---------- */
        div.stButton > button {
            border-radius: 10px;
            border: 1px solid #3a3d4d;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.15s ease-in-out;
        }

        div.stButton > button:hover {
            border-color: #6c63ff;
            color: #6c63ff;
        }

        /* ---------- Cards ---------- */
        .kb-card {
            background-color: #1f2130;
            border: 1px solid #2b2d3a;
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.8rem;
        }

        .kb-card p {
            margin: 0.15rem 0;
            font-size: 0.88rem;
            color: #d7d8e0;
        }

        .kb-card .kb-status-ready {
            color: #3ddc84;
            font-weight: 600;
        }

        .kb-card .kb-status-pending {
            color: #ff6b6b;
            font-weight: 600;
        }

        /* ---------- Chat bubbles ---------- */
        div[data-testid="stChatMessage"] {
            background-color: #1f2130;
            border-radius: 14px;
            padding: 0.6rem 0.9rem;
            margin-bottom: 0.6rem;
            border: 1px solid #292b38;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15);
        }

        /* ---------- Expander (sources) ---------- */
        div[data-testid="stExpander"] {
            border-radius: 10px;
            border: 1px solid #2b2d3a;
            background-color: #191a24;
        }

        /* ---------- Metrics ---------- */
        div[data-testid="stMetric"] {
            background-color: #1f2130;
            border-radius: 10px;
            padding: 0.5rem;
            border: 1px solid #2b2d3a;
        }

        /* ---------- Chat input ---------- */
        div[data-testid="stChatInput"] {
            border-radius: 14px;
            border: 1px solid #3a3d4d;
        }

        /* ---------- Scrollbars ---------- */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-thumb {
            background-color: #3a3d4d;
            border-radius: 8px;
        }

        ::-webkit-scrollbar-track {
            background-color: transparent;
        }

        /* ---------- Footer ---------- */
        .app-footer {
            text-align: center;
            color: #8a8d9a;
            font-size: 0.8rem;
            padding-top: 0.5rem;
        }

        /* ---------- Welcome screen ---------- */
        .welcome-wrapper {
            text-align: center;
            padding: 3rem 1rem;
        }

        .welcome-wrapper h1 {
            font-size: 2.2rem;
            margin-bottom: 0.3rem;
        }

        .welcome-wrapper .feature-list {
            display: inline-block;
            text-align: left;
            margin-top: 1.2rem;
            font-size: 1.02rem;
            line-height: 2;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# Session State Initialization
# ==========================================================


def create_chat_entry(title: str) -> ChatEntry:
    """Create a new empty chat session entry.

    Args:
        title: Display title for the chat.

    Returns:
        A dictionary representing a single chat session.
    """
    return {
        CHAT_KEY_TITLE: title,
        CHAT_KEY_MESSAGES: [],
        CHAT_KEY_MEMORY: ConversationMemory(),
        CHAT_KEY_PIPELINE: None,
    }


def start_new_chat() -> None:
    """Create a fresh chat session without touching the knowledge base."""
    state = ss()
    state[KEY_CHAT_COUNTER] += 1
    chat_id = str(uuid.uuid4())
    state[KEY_CHATS][chat_id] = create_chat_entry(f"Chat {state[KEY_CHAT_COUNTER]}")
    state[KEY_CURRENT_CHAT_ID] = chat_id


def init_session_state() -> None:
    """Initialize all required Streamlit session state variables."""
    state = ss()

    defaults: Dict[str, Any] = {
        KEY_CHATS: {},
        KEY_CURRENT_CHAT_ID: None,
        KEY_CHAT_COUNTER: 0,
        KEY_LOGGER: RAGLogger(),
        KEY_KB_READY: False,
        KEY_RETRIEVER: None,
        KEY_LLM: None,
        KEY_KB_STATS: {"documents": 0, "chunks": 0},
        KEY_UPLOADED_PDF_NAMES: [],
    }

    for key, value in defaults.items():
        if key not in state:
            state[key] = value

    if not state[KEY_CHATS]:
        start_new_chat()


def get_current_chat() -> ChatEntry:
    """Return the currently active chat session dictionary.

    Returns:
        Current chat entry.
    """
    state = ss()
    return state[KEY_CHATS][state[KEY_CURRENT_CHAT_ID]]


def ensure_pipeline_for_chat(chat: ChatEntry) -> RAGPipeline:
    """Lazily build or reuse the RAG pipeline for a chat.

    Args:
        chat: The chat session dictionary.

    Returns:
        Ready-to-use RAGPipeline instance for this chat.
    """
    state = ss()
    if chat[CHAT_KEY_PIPELINE] is None:
        chat[CHAT_KEY_PIPELINE] = RAGPipeline(
            retriever=state[KEY_RETRIEVER],
            llm=state[KEY_LLM],
            memory=chat[CHAT_KEY_MEMORY],
            logger=state[KEY_LOGGER],
        )
    return chat[CHAT_KEY_PIPELINE]


# ==========================================================
# Cached Resources
# ==========================================================


@st.cache_resource(show_spinner=False)
def get_cached_embedding_model() -> Any:
    """Return a cached embedding model instance.

    Returns:
        Embedding model from backend.
    """
    return create_embedding_model()


@st.cache_resource(show_spinner=False)
def get_cached_llm() -> Any:
    """Return a cached LLM instance.

    Returns:
        LLM instance from backend.
    """
    return initialize_llm()


# ==========================================================
# Knowledge Base Building
# ==========================================================


def save_uploaded_pdfs(uploaded_files: List[Any]) -> List[str]:
    """Persist uploaded PDF files to disk and return their paths.

    Args:
        uploaded_files: Files coming from st.file_uploader.

    Returns:
        List of saved file paths.
    """
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    pdf_paths: List[str] = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())
        pdf_paths.append(file_path)

    return pdf_paths


def _set_kb_session_values(
    retriever: Any,
    llm: Any,
    documents_count: int,
    chunks_count: int,
    uploaded_files: List[Any],
) -> None:
    """Store knowledge-base artifacts in session state.

    Args:
        retriever: Built retriever instance.
        llm: Initialized LLM instance.
        documents_count: Number of loaded documents.
        chunks_count: Number of generated chunks.
        uploaded_files: Uploaded Streamlit files.
    """
    state = ss()
    state[KEY_RETRIEVER] = retriever
    state[KEY_LLM] = llm
    state[KEY_KB_READY] = True
    state[KEY_KB_STATS] = {"documents": documents_count, "chunks": chunks_count}
    state[KEY_UPLOADED_PDF_NAMES] = [file.name for file in uploaded_files]

    # Rebuild active chat pipeline lazily against the refreshed KB.
    get_current_chat()[CHAT_KEY_PIPELINE] = None


def _progress_update(
    progress_bar: Any,
    status_placeholder: Any,
    step_index: int,
    total_steps: int,
    text: str,
) -> None:
    """Update progress bar and status placeholder.

    Args:
        progress_bar: Streamlit progress bar object.
        status_placeholder: Streamlit placeholder object.
        step_index: Current step number starting from 1.
        total_steps: Total number of steps.
        text: Status text to display.
    """
    status_placeholder.info(text)
    progress_bar.progress(step_index / total_steps, text=text)


def build_knowledge_base(uploaded_files: Optional[List[Any]]) -> None:
    """Build the knowledge base step-by-step with visible progress.

    Args:
        uploaded_files: Files coming from st.file_uploader.
    """
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one PDF first.")
        return

    progress_bar = st.progress(0, text="Starting...")
    status_placeholder = st.empty()
    total_steps = 6

    try:
        pdf_paths = save_uploaded_pdfs(uploaded_files)

        _progress_update(progress_bar, status_placeholder, 1, total_steps, "📂 Loading PDFs...")
        documents = load_documents(pdf_paths)

        _progress_update(
            progress_bar,
            status_placeholder,
            2,
            total_steps,
            "✂️ Chunking Documents...",
        )
        chunks = chunk_documents(documents)

        _progress_update(
            progress_bar,
            status_placeholder,
            3,
            total_steps,
            "🧠 Creating Embeddings...",
        )
        embedding_model = get_cached_embedding_model()

        _progress_update(progress_bar, status_placeholder, 4, total_steps, "📚 Building FAISS...")
        vector_store = build_vector_store(chunks, embedding_model)

        _progress_update(
            progress_bar,
            status_placeholder,
            5,
            total_steps,
            "🔍 Creating Retriever...",
        )
        retriever = create_retriever(vector_store)

        _progress_update(
            progress_bar,
            status_placeholder,
            6,
            total_steps,
            "🤖 Initializing LLM...",
        )
        llm = get_cached_llm()

        _set_kb_session_values(retriever, llm, len(documents), len(chunks), uploaded_files)

        progress_bar.progress(1.0, text="✅ Ready")
        status_placeholder.success("✅ Knowledge Base Created Successfully!")

    except Exception as error:
        status_placeholder.error(f"❌ Failed to build knowledge base: {error}")


# ==========================================================
# Sidebar
# ==========================================================


def render_kb_status_card() -> None:
    """Render the knowledge base status card in the sidebar."""
    state = ss()
    ready = state[KEY_KB_READY]

    status_html = (
        '<span class="kb-status-ready">🟢 Ready</span>'
        if ready
        else '<span class="kb-status-pending">🔴 Not Built</span>'
    )

    kb_stats = state[KEY_KB_STATS]

    st.markdown(
        f"""
        <div class="kb-card">
            <p><strong>Knowledge Base</strong></p>
            <p>{status_html}</p>
            <p>Documents: <strong>{kb_stats['documents']}</strong></p>
            <p>Chunks: <strong>{kb_stats['chunks']}</strong></p>
            <p>Embedding Model: <strong>{EMBEDDING_MODEL}</strong></p>
            <p>LLM: <strong>{LLM_MODEL}</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_statistics() -> None:
    """Render simple chat statistics in the sidebar."""
    state = ss()
    chat = get_current_chat()

    col1, col2 = st.columns(2)
    col1.metric("Messages", len(chat[CHAT_KEY_MESSAGES]))
    col2.metric("Chats", len(state[KEY_CHATS]))


def _render_chat_list() -> None:
    """Render the list of chat sessions in the sidebar."""
    state = ss()

    st.markdown("**💬 Chats**")
    for chat_id, chat in state[KEY_CHATS].items():
        is_active = chat_id == state[KEY_CURRENT_CHAT_ID]
        label = f"{'🟢 ' if is_active else ''}{chat[CHAT_KEY_TITLE]}"
        if st.button(label, key=f"chat_btn_{chat_id}", use_container_width=True):
            state[KEY_CURRENT_CHAT_ID] = chat_id
            st.rerun()


def _render_uploaded_pdf_list() -> None:
    """Render uploaded PDF names in an expander."""
    pdf_names = ss()[KEY_UPLOADED_PDF_NAMES]
    if not pdf_names:
        return

    with st.expander("📎 Uploaded PDFs"):
        for name in pdf_names:
            st.markdown(f"- {name}")


def render_sidebar() -> Optional[List[Any]]:
    """Render the full sidebar and return selected uploaded files.

    Returns:
        Files currently selected in the uploader.
    """
    uploaded_files: Optional[List[Any]] = None

    with st.sidebar:
        st.markdown(f"## {APP_ICON} {APP_TITLE}")
        st.caption(APP_DESCRIPTION)

        st.divider()

        if st.button("➕ New Chat", use_container_width=True):
            start_new_chat()
            st.rerun()

        _render_chat_list()

        st.divider()

        st.markdown("**📤 Upload PDFs**")
        uploaded_files = st.file_uploader(
            "Upload PDF Files",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if st.button("🚀 Build Knowledge Base", use_container_width=True):
            build_knowledge_base(uploaded_files)

        st.divider()

        render_kb_status_card()
        _render_uploaded_pdf_list()

        st.divider()

        render_chat_statistics()

        if st.button("🗑️ Clear This Conversation", use_container_width=True):
            clear_current_conversation()
            st.rerun()

        render_sidebar_footer()

    return uploaded_files


def render_sidebar_footer() -> None:
    """Render the sidebar footer listing technologies used."""
    st.markdown(
        """
        <div class="app-footer">
            Powered by<br>
            Streamlit • LangChain • FAISS<br>
            HuggingFace • Groq
        </div>
        """,
        unsafe_allow_html=True,
    )


def clear_current_conversation() -> None:
    """Reset memory/logger and clear messages of the active chat."""
    state = ss()
    chat = get_current_chat()

    try:
        reset_session(memory=chat[CHAT_KEY_MEMORY], logger_obj=state[KEY_LOGGER])
    except TypeError:
        # Compatibility fallback for older signature.
        reset_session(chat[CHAT_KEY_MEMORY], state[KEY_LOGGER])

    chat[CHAT_KEY_MESSAGES] = []
    chat[CHAT_KEY_PIPELINE] = None


# ==========================================================
# Welcome Screen
# ==========================================================


def render_welcome_screen() -> None:
    """Render the centered welcome screen shown before KB is built."""
    st.markdown(
        f"""
        <div class="welcome-wrapper">
            <h1>{APP_ICON} {APP_TITLE}</h1>
            <p>Upload one or more PDF files to create your knowledge base.</p>
            <div class="feature-list">
                ✔ Multi-document QA<br>
                ✔ Conversation Memory<br>
                ✔ Source Citation<br>
                ✔ Semantic Search<br>
                ✔ FAISS Vector Database
            </div>
            <p style="margin-top:2rem;">⬅ Start by uploading PDFs from the sidebar.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# Chat Rendering
# ==========================================================


def render_sources(sources_text: Optional[str]) -> None:
    """Render a sources expander if source content is meaningful.

    Args:
        sources_text: Formatted sources string returned by the pipeline.
    """
    if not sources_text or sources_text.strip() == NO_SOURCES_TEXT:
        return

    with st.expander("📄 Sources"):
        st.markdown(sources_text)


def render_chat_history(chat: ChatEntry) -> None:
    """Render all previously exchanged messages for a chat.

    Args:
        chat: Active chat session dictionary.
    """
    for message in chat[CHAT_KEY_MESSAGES]:
        role = message["role"]
        avatar = "👤" if role == ROLE_USER else "🤖"

        with st.chat_message(role, avatar=avatar):
            st.markdown(message["content"])
            if role == ROLE_ASSISTANT:
                render_sources(message.get("sources"))


def stream_answer(placeholder: Any, answer: str) -> None:
    """Reveal the assistant answer word-by-word with a typing cursor.

    Args:
        placeholder: Streamlit placeholder to render into.
        answer: Full answer text to reveal progressively.
    """
    words = answer.split(" ")
    displayed_text = ""

    for index, word in enumerate(words):
        displayed_text += word if index == 0 else f" {word}"
        cursor = "" if index == len(words) - 1 else "▌"
        placeholder.markdown(displayed_text + cursor)
        time.sleep(TYPING_DELAY_SECONDS)

    placeholder.markdown(displayed_text)


def handle_user_question(question: str, chat: ChatEntry) -> None:
    """Process a user question: display, run pipeline, and stream reply.

    Args:
        question: Question typed by the user.
        chat: Active chat session dictionary.
    """
    chat[CHAT_KEY_MESSAGES].append({"role": ROLE_USER, "content": question})

    with st.chat_message(ROLE_USER, avatar="👤"):
        st.markdown(question)

    with st.chat_message(ROLE_ASSISTANT, avatar="🤖"):
        placeholder = st.empty()
        placeholder.markdown("▌")

        try:
            pipeline = ensure_pipeline_for_chat(chat)
            response = pipeline.ask(question)

            answer = response.get("answer", "")
            sources = response.get("sources", "")

            stream_answer(placeholder, answer)
            render_sources(sources)

            chat[CHAT_KEY_MESSAGES].append(
                {"role": ROLE_ASSISTANT, "content": answer, "sources": sources}
            )

        except Exception as error:
            placeholder.empty()
            error_message = f"❌ Something went wrong: {error}"
            st.error(error_message)
            chat[CHAT_KEY_MESSAGES].append(
                {"role": ROLE_ASSISTANT, "content": error_message, "sources": None}
            )


# ==========================================================
# Footer
# ==========================================================


def render_footer() -> None:
    """Render the bottom-of-page footer."""
    st.divider()
    st.markdown(
        """
        <div class="app-footer">
            Developed using Streamlit • LangChain • FAISS • HuggingFace Embeddings • Groq
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# Main Application
# ==========================================================


def main() -> None:
    """Application entry point."""
    configure_page()
    inject_custom_css()
    init_session_state()

    render_sidebar()

    state = ss()
    chat = get_current_chat()

    if not state[KEY_KB_READY]:
        render_welcome_screen()
    else:
        render_chat_history(chat)
        question = st.chat_input("Ask a question about your uploaded PDF...")
        if question:
            handle_user_question(question, chat)

    render_footer()


if __name__ == "__main__":
    main()
