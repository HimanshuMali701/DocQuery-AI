import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from langchain_community.vectorstores import FAISS

STORAGE_ROOT = Path("storage")

def get_conversation_dir(user_id: int, conversation_id: int) -> Path:
    """Return the root path for a specific user's conversation."""
    return STORAGE_ROOT / "users" / f"user_{user_id}" / "conversations" / f"conv_{conversation_id}"

def get_uploads_dir(user_id: int, conversation_id: int) -> Path:
    """Return the uploads path for a conversation."""
    return get_conversation_dir(user_id, conversation_id) / "uploads"

def get_vectorstore_dir(user_id: int, conversation_id: int) -> Path:
    """Return the vectorstore path for a conversation."""
    return get_conversation_dir(user_id, conversation_id) / "vectorstore"

def create_folders(user_id: int, conversation_id: int) -> None:
    """Create directory structure for a conversation."""
    get_uploads_dir(user_id, conversation_id).mkdir(parents=True, exist_ok=True)
    get_vectorstore_dir(user_id, conversation_id).mkdir(parents=True, exist_ok=True)

def save_uploaded_files(user_id: int, conversation_id: int, uploaded_files: List[Any]) -> List[Path]:
    """Save Streamlit UploadedFile objects to the conversation uploads folder."""
    uploads_dir = get_uploads_dir(user_id, conversation_id)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for file in uploaded_files:
        file_path = uploads_dir / file.name
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        saved_paths.append(file_path)
    return saved_paths

def delete_uploaded_pdfs(user_id: int, conversation_id: int) -> None:
    """Delete all uploaded PDFs for a conversation."""
    uploads_dir = get_uploads_dir(user_id, conversation_id)
    if uploads_dir.exists():
        shutil.rmtree(uploads_dir)

def save_faiss(vector_store: Any, user_id: int, conversation_id: int) -> Path:
    """Save the FAISS vector database to the conversation vectorstore folder."""
    vs_dir = get_vectorstore_dir(user_id, conversation_id)
    vs_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(vs_dir))
    return vs_dir

def load_faiss(user_id: int, conversation_id: int, embedding_model: Any) -> Any:
    """Load the FAISS vector database from the conversation vectorstore folder."""
    vs_dir = get_vectorstore_dir(user_id, conversation_id)
    index_file = vs_dir / "index.faiss"
    if not index_file.exists():
        raise FileNotFoundError(f"No FAISS index found at '{vs_dir}'.")
    return FAISS.load_local(
        str(vs_dir),
        embedding_model,
        allow_dangerous_deserialization=True
    )

def delete_vectorstore(user_id: int, conversation_id: int) -> None:
    """Delete the vector database for a conversation."""
    vs_dir = get_vectorstore_dir(user_id, conversation_id)
    if vs_dir.exists():
        shutil.rmtree(vs_dir)

def delete_conversation_storage(user_id: int, conversation_id: int) -> None:
    """Delete the entire storage folder for a conversation."""
    conv_dir = get_conversation_dir(user_id, conversation_id)
    if conv_dir.exists():
        shutil.rmtree(conv_dir)

def get_storage_paths(user_id: int, conversation_id: int) -> Dict[str, Path]:
    """Return all storage paths for a conversation."""
    return {
        "conversation_dir": get_conversation_dir(user_id, conversation_id),
        "uploads_dir": get_uploads_dir(user_id, conversation_id),
        "vectorstore_dir": get_vectorstore_dir(user_id, conversation_id),
    }
