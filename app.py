from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List, Optional, TypedDict
import storage_manager
from auth import login_user, register_user
import streamlit as st
from postgres_db import * 
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
    generate_chat_title,
    initialize_llm,
    load_documents,
    reset_session,
    save_vector_store,
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

# Authentication Page

def authentication_page():

    st.title("🔐 RAG PDF Chatbot")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    # ---------------- LOGIN ---------------- #

    with tab1:

        st.subheader("Login")

        email = st.text_input(
            "Email",
            key="login_email"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button("Login"):

            success, result = login_user(email, password)

            if success:

                st.session_state.logged_in = True
                st.session_state.user = result
                st.session_state.user_id = result["id"]
                st.session_state.view = "dashboard"

                st.success("Login Successful!")

                st.rerun()

            else:

                st.error(result)

    # ---------------- SIGNUP ---------------- #

    with tab2:

        st.subheader("Create Account")

        name = st.text_input(
            "Full Name",
            key="signup_name"
        )

        email = st.text_input(
            "Email",
            key="signup_email"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="signup_password"
        )

        confirm = st.text_input(
            "Confirm Password",
            type="password",
            key="signup_confirm"
        )

        if st.button("Create Account"):

            if password != confirm:
                st.error("Passwords do not match.")

            elif len(password) < 8:
                st.error("Password must be at least 8 characters.")

            else:

                success, msg = register_user(
                    name,
                    email,
                    password
                )

                if success:

                    st.success(msg)

                else:

                    st.error(msg)

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

        /* ---------- Dashboard & Setup ---------- */
        .info-card {
            background-color: #1f2130;
            border: 1px solid #2b2d3a;
            border-radius: 12px;
            padding: 1.5rem;
            height: 100%;
            transition: all 0.2s ease-in-out;
        }
        .info-card:hover {
            border-color: #6c63ff;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(108, 99, 255, 0.1);
        }
        .recent-conv-card {
            background-color: #1f2130;
            border: 1px solid #2b2d3a;
            border-radius: 12px;
            padding: 1.2rem;
            margin-bottom: 0.8rem;
            transition: all 0.2s ease-in-out;
        }
        .recent-conv-card:hover {
            border-color: #6c63ff;
            box-shadow: 0 4px 12px rgba(108, 99, 255, 0.08);
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
    """Initialize all required Streamlit session state variables in a single place."""
    state = ss()

    # 0. Authentication Session Keys
    if "logged_in" not in state:
        state.logged_in = False

    if "user" not in state:
        state.user = None

    if "user_id" not in state:
        state.user_id = None

    # 1. Services and Managers
    if "database" not in state:
        state.database = DatabaseManager()

    if "logger" not in state or KEY_LOGGER not in state:
        logger_inst = RAGLogger()
        state.logger = logger_inst
        state[KEY_LOGGER] = logger_inst

    if "memory" not in state:
        state.memory = ConversationMemory()

    # 2. Knowledge Base Keys
    if "retriever" not in state or KEY_RETRIEVER not in state:
        state.retriever = None
        state[KEY_RETRIEVER] = None

    if "vector_store" not in state:
        state.vector_store = None

    if "llm" not in state or KEY_LLM not in state:
        state.llm = None
        state[KEY_LLM] = None

    if "knowledge_base_ready" not in state or KEY_KB_READY not in state:
        state.knowledge_base_ready = False
        state[KEY_KB_READY] = False

    if KEY_KB_STATS not in state:
        state[KEY_KB_STATS] = {"documents": 0, "chunks": 0}

    if KEY_UPLOADED_PDF_NAMES not in state:
        state[KEY_UPLOADED_PDF_NAMES] = []

    # 3. RAG Pipeline
    if "pipeline" not in state:
        state.pipeline = None

    # 4. Active Chat & Conversation State
    if "view" not in state:
        state.view = "dashboard"

    if "setup_step" not in state:
        state.setup_step = "upload"

    if "conversation_id" not in state:
        state.conversation_id = None

    if "current_chat_title" not in state:
        state.current_chat_title = "New Chat"

    if "messages" not in state:
        state.messages = []

    if KEY_CHATS not in state:
        state[KEY_CHATS] = {}

    if KEY_CURRENT_CHAT_ID not in state:
        state[KEY_CURRENT_CHAT_ID] = None

    if KEY_CHAT_COUNTER not in state:
        state[KEY_CHAT_COUNTER] = 0

    # Auto-load active or latest conversation from SQLite if conversation_id is None
    if state.get("logged_in"):
        if state.conversation_id is None:
            conversations = state.database.get_all_conversations(state.user_id)
            if conversations:
                latest = conversations[0]
                state.conversation_id = latest[0]
                state.current_chat_title = latest[1]
                state.messages = state.database.load_chat_history(latest[0])
                state.memory.clear()
                for msg in state.messages:
                    state.memory.add_message(msg.get("role", ""), msg.get("content", ""))
                load_conversation_knowledge_base(state.user_id, state.conversation_id)
            else:
                new_id = state.database.create_conversation("New Chat", state.user_id)
                state.conversation_id = new_id
                state.current_chat_title = "New Chat"
                state.messages = []
                state.memory.clear()
                load_conversation_knowledge_base(state.user_id, state.conversation_id)

        # Sync KEY_CHATS dictionary structure
        if not state[KEY_CHATS] or state[KEY_CURRENT_CHAT_ID] is None:
            chat_id = str(uuid.uuid4())
            state[KEY_CURRENT_CHAT_ID] = chat_id
            state[KEY_CHATS][chat_id] = {
                CHAT_KEY_TITLE: state.current_chat_title,
                CHAT_KEY_MESSAGES: state.messages,
                CHAT_KEY_MEMORY: state.memory,
                CHAT_KEY_PIPELINE: state.pipeline,
            }


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
    retriever = state.get(KEY_RETRIEVER) or state.get("retriever")
    llm = state.get(KEY_LLM) or state.get("llm")

    if chat.get(CHAT_KEY_PIPELINE) is None or state.get("pipeline") is None:
        pipeline = RAGPipeline(
            retriever=retriever,
            llm=llm,
            memory=state.memory,
            logger=state.logger,
            database=state.database,
            conversation_id=state.conversation_id,
        )
        chat[CHAT_KEY_PIPELINE] = pipeline
        state.pipeline = pipeline
    else:
        pipeline = chat[CHAT_KEY_PIPELINE]
        pipeline.conversation_id = state.conversation_id
        pipeline.memory = state.memory
        pipeline.database = state.database

    return pipeline


def load_conversation_knowledge_base(user_id: int, conversation_id: int) -> None:
    """Load documents and vector store for the selected conversation if they exist."""
    state = ss()
    
    # 1. Fetch documents from the database
    docs = state.database.get_documents_by_conversation(conversation_id)
    
    if docs:
        try:
            # Recreate retriever and load vector store
            embedding_model = get_cached_embedding_model()
            from backend import load_vector_store, create_retriever
            vector_store = load_vector_store(embedding_model, user_id=user_id, conversation_id=conversation_id)
            retriever = create_retriever(vector_store)
            llm = get_cached_llm()
            
            # Set session values
            _set_kb_session_values(
                retriever=retriever,
                llm=llm,
                documents_count=len(docs),
                chunks_count=vector_store.index.ntotal,
                uploaded_files=[],  # Empty since files are loaded from disk
                vector_store=vector_store
            )
            # Override KEY_UPLOADED_PDF_NAMES since uploaded_files was empty list
            state[KEY_UPLOADED_PDF_NAMES] = [doc["original_filename"] for doc in docs]
            
            # Rebuild active chat pipeline
            if state.get(KEY_CURRENT_CHAT_ID) is not None:
                chat = get_current_chat()
                chat[CHAT_KEY_PIPELINE] = None
            state.pipeline = None
            return
        except Exception as e:
            # If load fails (e.g. file missing on disk but present in DB), log warning and fall back
            pass
            
    # If no documents or loading failed, mark knowledge base as not ready
    state[KEY_KB_READY] = False
    state.knowledge_base_ready = False
    state[KEY_KB_STATS] = {"documents": 0, "chunks": 0}
    state[KEY_UPLOADED_PDF_NAMES] = []
    state.vector_store = None
    state.retriever = None
    state[KEY_RETRIEVER] = None
    
    # Rebuild active chat pipeline
    if state.get(KEY_CURRENT_CHAT_ID) is not None:
        chat = get_current_chat()
        chat[CHAT_KEY_PIPELINE] = None
    state.pipeline = None


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
    vector_store: Any = None,
) -> None:
    """Store knowledge-base artifacts in session state.

    Args:
        retriever: Built retriever instance.
        llm: Initialized LLM instance.
        documents_count: Number of loaded documents.
        chunks_count: Number of generated chunks.
        uploaded_files: Uploaded Streamlit files.
        vector_store: Built FAISS vector store.
    """
    state = ss()
    state[KEY_RETRIEVER] = retriever
    state.retriever = retriever
    state[KEY_LLM] = llm
    state.llm = llm
    state.vector_store = vector_store
    state[KEY_KB_READY] = True
    state.knowledge_base_ready = True
    state[KEY_KB_STATS] = {"documents": documents_count, "chunks": chunks_count}
    state[KEY_UPLOADED_PDF_NAMES] = [file.name for file in uploaded_files]

    # Rebuild active chat pipeline lazily against the refreshed KB.
    chat = get_current_chat()
    chat[CHAT_KEY_PIPELINE] = None
    state.pipeline = None


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
        user_id = st.session_state.user_id
        conversation_id = st.session_state.conversation_id

        # 1. Clean up old documents/vectorstore for this conversation from disk and DB
        storage_manager.delete_uploaded_pdfs(user_id, conversation_id)
        storage_manager.delete_vectorstore(user_id, conversation_id)
        st.session_state.database.delete_conversation_documents(conversation_id)

        # 2. Save new PDFs using storage_manager
        saved_paths = storage_manager.save_uploaded_files(user_id, conversation_id, uploaded_files)
        # Convert path objects to strings for compatibility with backend.py
        pdf_paths = [str(p) for p in saved_paths]

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

        # 3. Save vector store using storage_manager
        save_vector_store(vector_store, user_id=user_id, conversation_id=conversation_id)
        vectorstore_path = storage_manager.get_vectorstore_dir(user_id, conversation_id)

        # 4. Save documents to database
        for saved_path, file in zip(saved_paths, uploaded_files):
            st.session_state.database.add_document(
                user_id=user_id,
                conversation_id=conversation_id,
                original_filename=file.name,
                stored_filename=file.name,
                file_path=str(saved_path),
                vectorstore_path=str(vectorstore_path)
            )

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

        _set_kb_session_values(retriever, llm, len(documents), len(chunks), uploaded_files, vector_store)
        st.session_state.setup_step = "ready"

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
    conversations = (
        st.session_state.database.get_all_conversations()
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Messages",
        int(len(st.session_state.messages)/2),
    )

    col2.metric(
        "Chats",
        len(conversations),
    )


def _render_chat_list() -> None:
    """Render all conversations stored in database."""
    conversations = st.session_state.database.get_all_conversations()

    st.markdown("**💬 Chats**")

    for conversation in conversations:
        conversation_id = conversation[0]
        title = conversation[1]

        is_active = (conversation_id == st.session_state.conversation_id)
        label = f"{'🟢 ' if is_active else ''}{title}"

        if st.button(
            label,
            key=f"chat_{conversation_id}",
            use_container_width=True,
        ):
            st.session_state.conversation_id = conversation_id
            st.session_state.current_chat_title = title

            # Load chat history from SQLite
            loaded_messages = st.session_state.database.load_chat_history(conversation_id)
            st.session_state.messages = loaded_messages

            # Rebuild ConversationMemory with loaded messages
            st.session_state.memory.clear()
            for message in loaded_messages:
                st.session_state.memory.add_message(
                    message.get("role", ""),
                    message.get("content", "")
                )

            # Load knowledge base for this conversation
            load_conversation_knowledge_base(st.session_state.user_id, conversation_id)

            # Update current chat entry & pipeline
            chat = get_current_chat()
            chat[CHAT_KEY_TITLE] = title
            chat[CHAT_KEY_MESSAGES] = loaded_messages
            chat[CHAT_KEY_MEMORY] = st.session_state.memory
            chat[CHAT_KEY_PIPELINE] = None

            if st.session_state.pipeline:
                st.session_state.pipeline.conversation_id = conversation_id
                st.session_state.pipeline.memory = st.session_state.memory

            st.session_state.view = "chat"
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
        st.write("")

        # 1. USER PROFILE SECTION
        if st.session_state.logged_in and st.session_state.user:
            st.markdown("### 👤 User Profile")
            st.markdown(
                f"""
                <div style="background-color: #1f2130; border: 1px solid #2b2d3a; border-radius: 10px; padding: 0.8rem; margin-bottom: 0.6rem;">
                    <div style="font-size: 0.95rem; font-weight: 600; color: #f5f5f7;">{st.session_state.user['name']}</div>
                    <div style="font-size: 0.8rem; color: #8a8d9a;">{st.session_state.user['email']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button("Logout", use_container_width=True, key="sidebar_logout"):
                st.session_state.logged_in = False
                st.session_state.user = None
                st.session_state.user_id = None
                st.session_state.conversation_id = None
                st.session_state.messages = []
                st.session_state.memory.clear()
                st.session_state[KEY_CHATS] = {}
                st.session_state[KEY_CURRENT_CHAT_ID] = None
                st.session_state.view = "dashboard"
                st.rerun()
            st.write("")

        # Navigation to Dashboard
        if st.button("🏛️ Workspace Dashboard", use_container_width=True, key="sidebar_dashboard"):
            st.session_state.view = "dashboard"
            st.rerun()
        st.write("")

        # 2. CONVERSATIONS SECTION
        st.markdown("### 💬 Conversations")
        if st.button("➕ New Conversation", use_container_width=True, key="sidebar_new_chat"):
            # 1. Create a new conversation in SQLite
            new_conv_id = st.session_state.database.create_conversation("New Chat")
            st.session_state.conversation_id = new_conv_id
            st.session_state.current_chat_title = "New Chat"

            # Load (reset) knowledge base for this empty conversation
            load_conversation_knowledge_base(st.session_state.user_id, new_conv_id)

            # 2. Reset session memory & logger
            reset_session(
                memory=st.session_state.memory,
                logger_obj=st.session_state.logger,
            )

            # 3. Clear current messages in session state
            st.session_state.messages = []

            # 4. Update current chat structure and pipeline
            chat = get_current_chat()
            chat[CHAT_KEY_TITLE] = "New Chat"
            chat[CHAT_KEY_MESSAGES] = []
            chat[CHAT_KEY_MEMORY] = st.session_state.memory
            chat[CHAT_KEY_PIPELINE] = None

            if st.session_state.pipeline:
                st.session_state.pipeline.conversation_id = new_conv_id
                st.session_state.pipeline.memory = st.session_state.memory

            st.session_state.view = "setup"
            st.session_state.setup_step = "upload"
            st.rerun()

        _render_chat_list()
        st.write("")

        # 3. KNOWLEDGE BASE SECTION
        st.markdown("### 🧠 Knowledge Base")
        
        pdf_names = st.session_state.get(KEY_UPLOADED_PDF_NAMES, [])
        pdf_count = len(pdf_names)
        ready = st.session_state.get(KEY_KB_READY, False)
        
        status_html = "🟢 Ready" if ready else "🔴 Not Built"
        
        st.markdown(
            f"""
            <div style="background-color: #1f2130; border: 1px solid #2b2d3a; border-radius: 10px; padding: 0.8rem; margin-bottom: 0.6rem;">
                <div style="font-size: 0.9rem; font-weight: 600; color: #f5f5f7;">Status: {status_html}</div>
                <div style="font-size: 0.85rem; color: #8a8d9a; margin-top: 4px;">Documents: <strong>{pdf_count}</strong></div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if pdf_count > 0:
            for name in pdf_names:
                st.markdown(f"<span style='font-size: 0.88rem; color: #d7d8e0; display: block; margin-left: 4px;'>📄 {name}</span>", unsafe_allow_html=True)
            st.write("")

        with st.expander("📥 Manage Documents", expanded=False):
            uploaded_files = st.file_uploader(
                "Upload PDF Files",
                type=["pdf"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                key="sidebar_pdf_uploader"
            )

            if st.button("🚀 Build Knowledge Base", use_container_width=True, key="sidebar_build_kb"):
                build_knowledge_base(uploaded_files)
                st.rerun()

            if st.button("🗑️ Clear This Conversation", use_container_width=True, key="sidebar_clear_conv"):
                clear_current_conversation()
                st.rerun()
                
            if st.button("🗑️ Delete this Chat", use_container_width=True, key="sidebar_delete_chat"):
                if st.session_state.conversation_id is not None:
                    storage_manager.delete_conversation_storage(st.session_state.user_id, st.session_state.conversation_id)
                st.session_state.database.delete_conversation(st.session_state.conversation_id)
                st.session_state.conversation_id = None
                st.session_state.current_chat_title = None
                st.session_state.messages = []
                st.session_state.memory.clear()
                st.session_state.view = "dashboard"
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
    reset_session(
        memory=st.session_state.memory,
        logger_obj=st.session_state.logger,
    )

    if st.session_state.conversation_id is not None:
        if hasattr(st.session_state.database, "clear_chat_history"):
            st.session_state.database.clear_chat_history(
                st.session_state.conversation_id
            )
        elif hasattr(st.session_state.database, "delete_messages"):
            st.session_state.database.delete_messages(
                st.session_state.conversation_id
            )

    st.session_state.messages = []
    chat = get_current_chat()
    chat[CHAT_KEY_MESSAGES] = []
    chat[CHAT_KEY_MEMORY].clear()

# ==========================================================
# Welcome Screen
# ==========================================================


def render_welcome_screen() -> None:
    """Render the centered welcome screen shown before KB is built."""
    st.markdown(
        f"""
        <div class="welcome-wrapper">
            <h1>{APP_ICON} {APP_TITLE}</h1>
            <h3 style="margin-top:0.5rem;">
                Chat intelligently with your PDF documents
            </h3>
            <p style="font-size:1.05rem; margin-bottom:2rem;">
                Upload one or more PDF files to build your personal AI knowledge base
                and ask questions in natural language.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
    #### ✨ Features

    - 📄 Multi-PDF Chat
    - 🧠 AI-Powered Answers
    - 💬 Conversation Memory
    - 📚 Source Citations
    - 🔍 Smart Document Retrieval
    - ⚡ Fast Responses
    - 🔒 Secure Local Processing
    """)
    with col2:
        st.markdown("""
    #### 💡 Try Asking

    - Summarize this document
    - Explain Chapter 3
    - Compare uploaded PDFs
    - What are the key findings?
    - List important dates
    - Extract conclusions
    """)
    st.markdown(
    """
<div class="welcome-wrapper">
<p>⬅ Upload your PDFs from the sidebar to get started.</p>
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
    messages = st.session_state.messages if st.session_state.messages else chat.get(CHAT_KEY_MESSAGES, [])
    for message in messages:
        role = message.get("role", "")
        avatar = "👤" if role == ROLE_USER else "🤖"

        with st.chat_message(role, avatar=avatar):
            st.markdown(message.get("content", ""))
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
    user_msg: ChatMessage = {"role": ROLE_USER, "content": question}
    if user_msg not in st.session_state.messages:
        st.session_state.messages.append(user_msg)
    if chat.get(CHAT_KEY_MESSAGES) is not st.session_state.messages and user_msg not in chat[CHAT_KEY_MESSAGES]:
        chat[CHAT_KEY_MESSAGES].append(user_msg)

    with st.chat_message(ROLE_USER, avatar="👤"):
        st.markdown(question)

    with st.chat_message(ROLE_ASSISTANT, avatar="🤖"):
        placeholder = st.empty()
        placeholder.markdown("▌")

        try:
            pipeline = ensure_pipeline_for_chat(chat)
            if st.session_state.conversation_id is None:
                conversation_id = st.session_state.database.create_conversation("New Chat")
                st.session_state.conversation_id = conversation_id
                st.session_state.current_chat_title = "New Chat"

            pipeline.conversation_id = st.session_state.conversation_id
            pipeline.memory = st.session_state.memory

            response = pipeline.ask(question)

            answer = response.get("answer", "")
            sources = response.get("sources", "")

            stream_answer(placeholder, answer)
            render_sources(sources)

            assistant_msg: ChatMessage = {"role": ROLE_ASSISTANT, "content": answer, "sources": sources}
            if assistant_msg not in st.session_state.messages:
                st.session_state.messages.append(assistant_msg)
            if chat.get(CHAT_KEY_MESSAGES) is not st.session_state.messages and assistant_msg not in chat[CHAT_KEY_MESSAGES]:
                chat[CHAT_KEY_MESSAGES].append(assistant_msg)

            # Generate chat title if it's the first message (current title is "New Chat")
            conversation = st.session_state.database.get_conversation(st.session_state.conversation_id)
            if conversation and conversation["title"] == "New Chat":
                try:
                    llm = st.session_state.get("llm") or get_cached_llm()
                    if llm:
                        new_title = generate_chat_title(llm, question)
                        if new_title:
                            st.session_state.database.update_conversation_title(
                                st.session_state.conversation_id,
                                new_title
                            )
                            st.session_state.current_chat_title = new_title
                            chat[CHAT_KEY_TITLE] = new_title
                            st.rerun()
                except Exception as title_err:
                    import logging
                    logging.getLogger(__name__).warning(f"Failed to generate chat title: {title_err}")

        except Exception as error:
            placeholder.empty()
            error_message = f"❌ Something went wrong: {error}"
            st.error(error_message)
            err_msg: ChatMessage = {"role": ROLE_ASSISTANT, "content": error_message, "sources": None}
            st.session_state.messages.append(err_msg)
            if chat.get(CHAT_KEY_MESSAGES) is not st.session_state.messages and err_msg not in chat[CHAT_KEY_MESSAGES]:
                chat[CHAT_KEY_MESSAGES].append(err_msg)


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
# Dashboard & Setup helpers
# ==========================================================

from datetime import datetime

def get_conversation_stats(db: DatabaseManager, conversation_id: int) -> Dict[str, Any]:
    """Retrieve message count and last updated timestamp for a conversation."""
    messages = db.get_messages(conversation_id)
    msg_count = len(messages)
    
    if msg_count > 0:
        last_updated = messages[-1]["timestamp"]
    else:
        conv = db.get_conversation(conversation_id)
        last_updated = conv["created_at"] if conv else "Unknown"
        
    return {
        "message_count": msg_count,
        "last_updated": last_updated
    }


def format_timestamp(ts_str: str) -> str:
    """Format a SQLite timestamp string into a friendly readable format."""
    try:
        dt = datetime.strptime(ts_str.split(".")[0], "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%b %d, %Y, %I:%M %p")
    except Exception:
        return ts_str


def render_dashboard() -> None:
    """Render the AI Workspace Dashboard landing page."""
    st.markdown(
        """
        <div style="text-align: center; padding: 2.5rem 1rem 1.5rem 1rem;">
            <h1 style="font-size: 3rem; font-weight: 700; margin-bottom: 0.5rem;">🤖 DocQuery AI</h1>
            <h3 style="font-size: 1.3rem; font-weight: 400; color: #8a8d9a; margin-bottom: 1.5rem;">
                Chat intelligently with your PDF documents.
            </h3>
            <p style="font-size: 1.05rem; color: #d7d8e0; max-width: 600px; margin: 0 auto 2.5rem auto; line-height: 1.6;">
                Upload documents, build an AI knowledge base, and ask natural language questions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ New Conversation", use_container_width=True, key="dash_btn_new_chat", type="primary"):
            new_conv_id = st.session_state.database.create_conversation("New Chat")
            st.session_state.conversation_id = new_conv_id
            st.session_state.current_chat_title = "New Chat"
            load_conversation_knowledge_base(st.session_state.user_id, new_conv_id)
            reset_session(
                memory=st.session_state.memory,
                logger_obj=st.session_state.logger,
            )
            st.session_state.messages = []
            
            chat = get_current_chat()
            chat[CHAT_KEY_TITLE] = "New Chat"
            chat[CHAT_KEY_MESSAGES] = []
            chat[CHAT_KEY_MEMORY] = st.session_state.memory
            chat[CHAT_KEY_PIPELINE] = None
            
            st.session_state.view = "setup"
            st.session_state.setup_step = "upload"
            st.rerun()
            
    with col2:
        if st.button("📂 Continue Previous Chats", use_container_width=True, key="dash_btn_continue"):
            st.info("💡 Scroll down to view and load your Recent Conversations.")

    st.markdown("<hr style='border-color: #2b2d3a; margin: 2.5rem 0 3rem 0;'>", unsafe_allow_html=True)
    
    # 3 modern cards
    card1, card2, card3 = st.columns(3)
    with card1:
        st.markdown(
            """
            <div class="info-card">
                <h4 style="color: #f5f5f7; margin: 0 0 0.8rem 0; font-size: 1.15rem;">📄 Upload Documents</h4>
                <p style="color: #8a8d9a; font-size: 0.92rem; line-height: 1.5; margin: 0;">
                    Upload one or multiple PDF documents directly into your workspace.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with card2:
        st.markdown(
            """
            <div class="info-card">
                <h4 style="color: #f5f5f7; margin: 0 0 0.8rem 0; font-size: 1.15rem;">🧠 AI Knowledge Base</h4>
                <p style="color: #8a8d9a; font-size: 0.92rem; line-height: 1.5; margin: 0;">
                    Each conversation has its own isolated knowledge base. Documents are never shared between conversations.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with card3:
        st.markdown(
            """
            <div class="info-card">
                <h4 style="color: #f5f5f7; margin: 0 0 0.8rem 0; font-size: 1.15rem;">💬 Chat Naturally</h4>
                <p style="color: #8a8d9a; font-size: 0.92rem; line-height: 1.5; margin: 0;">
                    Ask questions, summarize, compare, extract data, and analyze key information easily.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### 📂 Recent Conversations")
    
    conversations = st.session_state.database.get_all_conversations(st.session_state.user_id)
    if not conversations:
        st.info("No conversations found. Click 'New Conversation' to get started!")
    else:
        for i in range(0, len(conversations), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(conversations):
                    conv = conversations[i + j]
                    conv_id = conv[0]
                    title = conv[1]
                    
                    stats = get_conversation_stats(st.session_state.database, conv_id)
                    formatted_time = format_timestamp(stats["last_updated"])
                    
                    with cols[j]:
                        st.markdown(
                            f"""
                            <div class="recent-conv-card">
                                <h4 style="margin: 0; color: #f5f5f7; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 1.1rem;">💬 {title}</h4>
                                <p style="font-size: 0.85rem; color: #8a8d9a; margin: 6px 0 10px 0; line-height: 1.4;">
                                    Last Active: <strong>{formatted_time}</strong><br>
                                    Messages: <strong>{stats['message_count']}</strong>
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        if st.button("Open Chat →", key=f"dash_chat_{conv_id}", use_container_width=True):
                            st.session_state.conversation_id = conv_id
                            st.session_state.current_chat_title = title
                            
                            loaded_messages = st.session_state.database.load_chat_history(conv_id)
                            st.session_state.messages = loaded_messages
                            st.session_state.memory.clear()
                            for msg in loaded_messages:
                                st.session_state.memory.add_message(msg.get("role", ""), msg.get("content", ""))
                                
                            load_conversation_knowledge_base(st.session_state.user_id, conv_id)
                            
                            chat = get_current_chat()
                            chat[CHAT_KEY_TITLE] = title
                            chat[CHAT_KEY_MESSAGES] = loaded_messages
                            chat[CHAT_KEY_MEMORY] = st.session_state.memory
                            chat[CHAT_KEY_PIPELINE] = None
                            
                            st.session_state.view = "chat"
                            st.rerun()


def render_setup_step_indicator(current_step: str) -> None:
    """Render a step status bar for the setup page."""
    cols = st.columns(3)
    steps = [
        ("Step 1", "Upload PDFs", "upload"),
        ("Step 2", "Build Knowledge Base", "build"),
        ("Step 3", "Start Chatting", "ready")
    ]
    for idx, (label, desc, step_id) in enumerate(steps):
        with cols[idx]:
            is_active = (current_step == step_id)
            is_done = (step_id == "upload" and current_step in ["build", "ready"]) or (step_id == "build" and current_step == "ready")
            
            border_style = "border-color: #6c63ff;" if is_active else ("border-color: #3ddc84;" if is_done else "border-color: #2b2d3a;")
            bg_style = "background-color: rgba(108, 99, 255, 0.08);" if is_active else ("background-color: rgba(61, 220, 132, 0.04);" if is_done else "background-color: transparent;")
            text_color = "#f5f5f7" if is_active or is_done else "#8a8d9a"
            badge = "🟢" if is_active else ("✅" if is_done else "⚪")
            
            st.markdown(
                f"""
                <div style="border: 1px solid; border-radius: 10px; padding: 0.8rem; text-align: center; {border_style} {bg_style}">
                    <div style="font-size: 0.75rem; color: #8a8d9a; text-transform: uppercase; font-weight: 600;">{label}</div>
                    <div style="font-size: 0.95rem; font-weight: 500; margin-top: 4px; color: {text_color};">{badge} {desc}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


def render_conversation_setup_page(chat: ChatEntry) -> None:
    """Render the setup page guiding the user in building a knowledge base."""
    st.markdown("## Create New Conversation")
    
    if st.session_state.setup_step in ["upload", "build"]:
        render_setup_step_indicator(st.session_state.setup_step)
        st.write("")
        
        uploaded_files = st.file_uploader(
            "Upload PDF files for this conversation",
            type=["pdf"],
            accept_multiple_files=True,
            key="setup_pdf_uploader"
        )
        
        if uploaded_files:
            st.markdown(f"#### ✓ {len(uploaded_files)} PDF{'s' if len(uploaded_files) > 1 else ''} Selected")
            for f in uploaded_files:
                st.markdown(f"<span style='color: #3ddc84; font-size: 0.95rem; margin-left: 5px;'>📄 {f.name}</span>", unsafe_allow_html=True)
            st.write("")
            
            if st.button("🚀 Build Knowledge Base", use_container_width=True, type="primary"):
                # Run the actual build
                with st.spinner("Processing documents and building Knowledge Base..."):
                    build_knowledge_base(uploaded_files)
                st.rerun()
                
    elif st.session_state.setup_step == "ready":
        render_setup_step_indicator("ready")
        st.write("")
        
        st.markdown(
            """
            <div style="text-align: center; padding: 2.5rem 1.5rem; background-color: #1f2130; border: 1px solid #2b2d3a; border-radius: 12px; margin-bottom: 2rem;">
                <h2 style="color: #3ddc84; font-size: 2rem; margin: 0 0 0.8rem 0;">✅ Knowledge Base Ready</h2>
                <p style="color: #d7d8e0; font-size: 1.1rem; line-height: 1.6; margin: 0;">
                    Your documents have been processed successfully. AI is ready.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if st.button("Start Chat →", use_container_width=True, type="primary"):
            st.session_state.view = "chat"
            st.rerun()


# ==========================================================
# Main Application
# ==========================================================


def main() -> None:
    """Application entry point."""
    configure_page()
    inject_custom_css()
    init_session_state()

    if not st.session_state.logged_in:
        authentication_page()
        st.stop()

    render_sidebar()

    state = ss()
    chat = get_current_chat()

    if state.view == "dashboard":
        render_dashboard()
    elif state.view == "setup":
        render_conversation_setup_page(chat)
    elif state.view == "chat":
        # Check if Knowledge Base exists
        if not state[KEY_KB_READY]:
            st.markdown(
                """
                <div style="text-align: center; padding: 4rem 1rem; background-color: #1f2130; border: 1px solid #2b2d3a; border-radius: 12px; max-width: 600px; margin: 2rem auto;">
                    <h2 style="font-size: 3rem; margin-bottom: 1rem; color: #ff6b6b;">📂</h2>
                    <h3 style="color: #f5f5f7;">No Knowledge Base Found</h3>
                    <p style="color: #8a8d9a; margin-top: 0.5rem; margin-bottom: 2rem; font-size: 1.05rem;">
                        Upload one or more PDF documents to start chatting.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button("Upload PDFs", use_container_width=True, type="primary", key="empty_state_upload"):
                st.session_state.view = "setup"
                st.session_state.setup_step = "upload"
                st.rerun()
        else:
            render_chat_history(chat)
            question = st.chat_input("Ask a question about your uploaded PDF...")
            if question:
                handle_user_question(question, chat)

    render_footer()


if __name__ == "__main__":
    main()
