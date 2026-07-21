# ==========================================================
# Configuration
# ==========================================================

# ---------- Chunking ----------
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 100

# ---------- Embedding Model ----------
EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------- Vector Search ----------
SEARCH_TYPE: str = "similarity"
TOP_K: int = 8

# ---------- LLM ----------
LLM_PROVIDER: str = "groq"
LLM_MODEL: str = "llama-3.3-70b-versatile"
TEMPERATURE: int = 0
MAX_HISTORY: int = 6

# ---------- Paths ----------
FAISS_INDEX_PATH: str = "faiss_index"


# ==========================================================
# Imports
# ==========================================================

import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentLike(Protocol):
    """Minimal document protocol used by this module."""

    page_content: str
    metadata: Dict[str, Any]


class VectorStoreLike(Protocol):
    """Minimal vector store protocol used by this module."""

    index: Any

    def as_retriever(self, search_type: str, search_kwargs: Dict[str, Any]) -> Any:
        """Return a retriever for the vector store."""


class LLMLike(Protocol):
    """Minimal language model protocol used by this module."""

    def invoke(self, prompt: str) -> Any:
        """Generate a response for the given prompt."""


class PipelineLike(Protocol):
    """Minimal pipeline protocol used by safe_ask."""

    def ask(self, question: str) -> Dict[str, Any]:
        """Ask the pipeline a question."""


# ==========================================================
# Validation
# ==========================================================


def validate_pdf_paths(pdf_paths: Sequence[str]) -> None:
    """Validate uploaded PDF file paths.

    Args:
        pdf_paths: PDF file paths to validate.

    Raises:
        ValueError: If no paths are provided or a path is not a PDF.
        FileNotFoundError: If a file does not exist.
    """
    if not pdf_paths:
        raise ValueError("Please upload at least one PDF.")

    for path in pdf_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        if not path.lower().endswith(".pdf"):
            raise ValueError(f"Not a PDF file: {path}")


def validate_documents(documents: Sequence[DocumentLike]) -> None:
    """Ensure documents contain readable text.

    Args:
        documents: Documents to validate.

    Raises:
        ValueError: If no readable text was found.
    """
    if len(documents) == 0:
        raise ValueError("No readable text found in the uploaded PDF.")


def validate_question(question: Optional[str]) -> None:
    """Validate user question.

    Args:
        question: User question text.

    Raises:
        ValueError: If the question is missing or blank.
    """
    if question is None:
        raise ValueError("Question cannot be None.")

    if not question.strip():
        raise ValueError("Please enter a valid question.")


def validate_retrieved_documents(documents: Sequence[DocumentLike]) -> None:
    """Ensure retrieval returned documents.

    Args:
        documents: Retrieved documents to validate.

    Raises:
        ValueError: If no documents were retrieved.
    """
    if len(documents) == 0:
        raise ValueError("No relevant information found in the uploaded PDF.")


def validate_vector_store(vector_store: Optional[VectorStoreLike]) -> None:
    """Validate FAISS vector store.

    Args:
        vector_store: Vector store instance to validate.

    Raises:
        ValueError: If the vector store is not initialized.
    """
    if vector_store is None:
        raise ValueError("Vector store has not been initialized.")


def safe_ask(pipeline: PipelineLike, question: str) -> Dict[str, Any]:
    """Safely execute the RAG pipeline.

    Args:
        pipeline: Pipeline instance to execute.
        question: User question.

    Returns:
        A response dictionary matching the pipeline output shape.
    """
    try:
        return pipeline.ask(question)
    except Exception as error:
        logger.exception("Pipeline execution failed")
        return {
            "answer": str(error),
            "sources": "No sources available."
        }


# ==========================================================
# Document Loading
# ==========================================================


def load_documents(pdf_paths: List[str]):
    """Load one or more PDF files and return LangChain document objects.

    Args:
        pdf_paths: List of PDF file paths.

    Returns:
        Combined list of documents from all PDFs.
    """
    validate_pdf_paths(pdf_paths)
    documents = []

    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            logger.warning("File not found: %s", pdf_path)
            continue

        logger.info("Loading PDF: %s", os.path.basename(pdf_path))

        loader = PyPDFLoader(pdf_path)
        pdf_docs = loader.load()

        pdf_docs = [doc for doc in pdf_docs if doc.page_content.strip()]
        documents.extend(pdf_docs)

    logger.info("========== Document Statistics ==========")
    logger.info("Total PDFs Loaded : %s", len(pdf_paths))
    logger.info("Total Pages       : %s", len(documents))

    validate_documents(documents)
    return documents


# ==========================================================
# Chunking
# ==========================================================


def chunk_documents(documents: Sequence[DocumentLike]):
    """Split LangChain documents into smaller chunks.

    Args:
        documents: Documents returned by load_documents().

    Returns:
        Chunked documents preserving metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(list(documents))

    logger.info("========== Chunk Statistics ==========")
    logger.info("Original Documents : %s", len(documents))
    logger.info("Total Chunks       : %s", len(chunks))
    logger.info("Chunk Size         : %s", CHUNK_SIZE)
    logger.info("Chunk Overlap      : %s", CHUNK_OVERLAP)

    return chunks


# ==========================================================
# Embedding
# ==========================================================


def create_embedding_model():
    """Initialize the HuggingFace embedding model.

    Returns:
        Configured embedding model.
    """
    logger.info("Loading Embedding Model...")
    logger.info("Model : %s", EMBEDDING_MODEL)

    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")

    if hf_token is None:
        raise ValueError("HF_TOKEN not found in .env file.")

    login(token=hf_token)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

    logger.info("Using device: %s", device)
    logger.info("Embedding model loaded successfully.")

    return embedding_model


# ==========================================================
# Vector Store
# ==========================================================


def build_vector_store(chunks: Sequence[DocumentLike], embedding_model: Any):
    """Create a FAISS vector store from document chunks.

    Args:
        chunks: Chunked documents.
        embedding_model: Embedding model used to generate vector representations.

    Returns:
        Initialized FAISS vector store.
    """
    logger.info("Creating FAISS Vector Store...")

    vector_store = FAISS.from_documents(
        documents=list(chunks),
        embedding=embedding_model
    )

    logger.info("FAISS Vector Store created successfully.")

    return vector_store


def save_vector_store(vector_store: Any) -> None:
    """Save the FAISS vector store to disk.

    Args:
        vector_store: FAISS vector store instance.
    """
    vector_store.save_local(FAISS_INDEX_PATH)

    logger.info("Vector Store saved to '%s'", FAISS_INDEX_PATH)


def load_vector_store(embedding_model: Any):
    """Load an existing FAISS vector store.

    Args:
        embedding_model: Embedding model used for loading.

    Returns:
        Loaded FAISS vector store.
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"No FAISS index found at '{FAISS_INDEX_PATH}'.")

    logger.info("Loading FAISS Vector Store...")

    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    logger.info("Vector Store loaded successfully.")

    return vector_store


def vector_store_info(vector_store: Any) -> None:
    """Display information about the FAISS vector store.

    Args:
        vector_store: FAISS vector store instance.
    """
    logger.info("========== Vector Store ==========")
    logger.info("Indexed Chunks : %s", vector_store.index.ntotal)
    logger.info("Index Path     : %s", FAISS_INDEX_PATH)


# ==========================================================
# Retriever
# ==========================================================


def create_retriever(vector_store: Any):
    """Create a document retriever from the FAISS vector store.

    Args:
        vector_store: Initialized FAISS vector store.

    Returns:
        Configured retriever.
    """
    logger.info("Initializing Retriever...")

    retriever = vector_store.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={"k": TOP_K}
    )

    logger.info("Retriever initialized successfully.")
    validate_vector_store(vector_store)
    return retriever


def retrieve_context(retriever: Any, query: str):
    """Retrieve the most relevant document chunks.

    Args:
        retriever: VectorStoreRetriever instance.
        query: User question.

    Returns:
        Retrieved LangChain Document objects.
    """
    documents = retriever.invoke(query)

    return documents


# ==========================================================
# Context Formatting
# ==========================================================


def format_context(documents: Sequence[DocumentLike]) -> str:
    """Convert retrieved documents into a formatted context string.

    Args:
        documents: Retrieved LangChain documents.

    Returns:
        Formatted context string.
    """
    context = []

    for i, doc in enumerate(documents, start=1):
        metadata = doc.metadata
        source = metadata.get("source", "Unknown")
        page = metadata.get("page", 0) + 1

        context.append(
            f"""
========== Document {i} ==========
Source : {source}
Page   : {page}

{doc.page_content}
"""
        )

    return "\n".join(context)


# ==========================================================
# Conversation Memory
# ==========================================================


class ConversationMemory:
    """Stores conversation history for the current chat session."""

    def __init__(self) -> None:
        """Initialize an empty conversation history."""
        self.history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history.

        Args:
            role: Message role, such as user or assistant.
            content: Message text.
        """
        self.history.append({"role": role, "content": content})

    def get_history(self, max_messages: Optional[int] = None) -> List[Dict[str, str]]:
        """Return the conversation history, optionally truncated.

        Args:
            max_messages: Optional maximum number of trailing messages to return.

        Returns:
            Conversation history entries.
        """
        if max_messages is None:
            return self.history
        return self.history[-max_messages:]

    def clear(self) -> None:
        """Clear the conversation history."""
        self.history = []

    def display(self) -> None:
        """Print the conversation history."""
        logger.info("========== Conversation History ==========")

        for message in self.history:
            logger.info("%s:", message['role'].upper())
            logger.info(message["content"])


# ==========================================================
# LLM
# ==========================================================


def initialize_llm():
    """Initialize the Groq LLM.

    Returns:
        Configured LLM instance.
    """
    logger.info("Initializing LLM...")

    load_dotenv()

    groq_api_key = os.getenv("GROQ_API_KEY")

    if groq_api_key is None:
        raise ValueError("GROQ_API_KEY not found in .env file.")

    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=groq_api_key
    )

    logger.info("Model : %s", LLM_MODEL)
    logger.info("LLM initialized successfully.")

    return llm


# ==========================================================
# Prompt Builder
# ==========================================================


def build_prompt(context: str, history: Sequence[Dict[str, str]], question: str) -> str:
    """Build the prompt using conversation history and retrieved context.

    Args:
        context: Retrieved document context.
        history: Conversation history entries.
        question: User question.

    Returns:
        Complete prompt string.
    """
    history_text = ""

    for message in history:
        history_text += f"{message['role'].capitalize()}: {message['content']}\n"

    prompt = f"""
You are an intelligent PDF Question Answering Assistant.

Your task is to answer ONLY from the supplied context.

Rules:
1. Read the complete context carefully.
2. Answer ONLY using the provided context.
3. Do NOT use your own knowledge.
4. If the answer is not available in the context, reply exactly:

I couldn't find relevant information in the uploaded PDF.

5. If multiple context documents contain relevant information, combine them into one complete answer.
6. Write the answer in clear bullet points whenever appropriate.
7. Do not mention "according to the context" or "based on the provided context".

==================================================
Conversation History
==================================================

{history_text}

==================================================
CONTEXT
==================================================

{context}

==================================================
QUESTION
==================================================

{question}

==================================================
ANSWER
==================================================
"""

    return prompt


def generate_answer(llm: LLMLike, prompt: str) -> str:
    """Generate an answer using the LLM.

    Args:
        llm: LLM instance.
        prompt: Prompt text.

    Returns:
        Generated answer text.
    """
    response = llm.invoke(prompt)

    return response.content


# ==========================================================
# Source Citation
# ==========================================================


def collect_sources(documents: Sequence[DocumentLike]) -> Dict[str, set]:
    """Collect unique source files and page numbers from retrieved documents.

    Args:
        documents: Retrieved LangChain documents.

    Returns:
        Mapping of source filenames to sets of page numbers.
    """
    sources: Dict[str, set] = defaultdict(set)

    for doc in documents:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 0) + 1
        sources[source].add(page)

    return sources


def format_sources(documents: Sequence[DocumentLike]) -> str:
    """Format retrieved document sources into a readable string.

    Args:
        documents: Retrieved LangChain documents.

    Returns:
        Readable source listing.
    """
    sources = collect_sources(documents)

    if not sources:
        return "No source information available."

    formatted = []

    for source, pages in sources.items():
        pages = sorted(pages)
        page_string = ", ".join(str(page) for page in pages)
        formatted.append(f"• {source} (Pages: {page_string})")

    return "\n".join(formatted)


# ==========================================================
# RAG Pipeline
# ==========================================================


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline."""

    def __init__(self, retriever: Any, llm: LLMLike, memory: ConversationMemory, logger: Any, database=None,
    conversation_id=None) -> None:
        """Initialize the pipeline.

        Args:
            retriever: Configured retriever.
            llm: Initialized language model.
            memory: Conversation memory instance.
            logger: Logger instance.
        """
        self.retriever = retriever
        self.llm = llm
        self.memory = memory
        self.logger = logger
        self.database = database
        self.conversation_id = conversation_id

    def ask(self, question: str) -> Dict[str, Any]:
        """Answer a question using retrieved context and the LLM.

        Args:
            question: User question.

        Returns:
            Response payload containing answer, sources, and retrieved documents.
        """
        validate_question(question)
        if self.database and self.conversation_id:
            self.database.save_message(self.conversation_id,"user",question)
        self.memory.add_message("user", question)

        documents = retrieve_context(self.retriever, question)
        validate_retrieved_documents(documents)

        context = format_context(documents)
        history = self.memory.get_history(MAX_HISTORY)

        prompt = build_prompt(
            context=context,
            history=history,
            question=question
        )

        answer = generate_answer(self.llm, prompt)

        if self.database and self.conversation_id:
            self.database.save_message(self.conversation_id,"assistant",answer)
        self.memory.add_message("assistant", answer)

        formatted_sources = format_sources(documents)

        return {
            "question": question,
            "answer": answer,
            "sources": formatted_sources,
            "retrieved_documents": documents
        }


# ==========================================================
# Logger
# ==========================================================


class RAGLogger:
    """Logger for storing RAG chatbot interactions and performance metrics."""

    def __init__(self) -> None:
        """Initialize an empty log list."""
        self.logs: List[Dict[str, Any]] = []

    def log(
        self,
        question: str,
        answer: str,
        response_time: float,
        retrieved_documents: Sequence[DocumentLike],
        llm_model: str,
        embedding_model: str
    ) -> None:
        """Store one chatbot interaction.

        Args:
            question: Asked question.
            answer: Generated answer.
            response_time: Time spent generating the answer.
            retrieved_documents: Documents used for generation.
            llm_model: LLM model name.
            embedding_model: Embedding model name.
        """
        sources = []

        for doc in retrieved_documents:
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", 0) + 1
            })

        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "answer": answer,
            "response_time_seconds": round(response_time, 3),
            "retrieved_chunks": len(retrieved_documents),
            "sources": sources,
            "llm_model": llm_model,
            "embedding_model": embedding_model
        }

        self.logs.append(log_entry)

    def show_logs(self) -> None:
        """Display all logged interactions."""
        logger.info("========== RAG Logs ==========")

        for i, log in enumerate(self.logs, start=1):
            logger.info("=" * 80)
            logger.info("Interaction %s", i)
            logger.info("=" * 80)
            logger.info("Time          : %s", log['timestamp'])
            logger.info("Question      : %s", log['question'])
            logger.info("Response Time : %s sec", log['response_time_seconds'])
            logger.info("Chunks Used   : %s", log['retrieved_chunks'])
            logger.info("LLM           : %s", log['llm_model'])
            logger.info("Embedding     : %s", log['embedding_model'])
            logger.info("Sources       : %s", log['sources'])

    def export_json(self, filename: str = "rag_logs.json") -> None:
        """Export logs to a JSON file.

        Args:
            filename: Output JSON filename.
        """
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(self.logs, file, indent=4)

        logger.info("Logs exported to %s", filename)

    def export_csv(self, filename: str = "rag_logs.csv") -> None:
        """Export logs to CSV.

        Args:
            filename: Output CSV filename.
        """
        rows = []

        for log in self.logs:
            rows.append({
                "Timestamp": log["timestamp"],
                "Question": log["question"],
                "Response Time": log["response_time_seconds"],
                "Chunks": log["retrieved_chunks"],
                "LLM": log["llm_model"],
                "Embedding": log["embedding_model"],
                "Soyce": log["sources"]
            })

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)

        logger.info("Logs exported to %s", filename)

    def clear(self) -> None:
        """Clear stored logs."""
        self.logs.clear()


# ==========================================================
# Reset
# ==========================================================


def reset_memory(memory: ConversationMemory) -> None:
    """Reset conversation memory.

    Args:
        memory: Conversation memory instance.
    """
    memory.clear()

    logger.info("Conversation memory cleared.")


def reset_logger(logger_obj: RAGLogger) -> None:
    """Reset the logger by clearing all stored logs.

    Args:
        logger_obj: Logger instance.
    """
    logger_obj.clear()

    logger.info("Logger cleared.")


def reset_pipeline(memory: ConversationMemory, logger_obj: RAGLogger) -> None:
    """Reset both conversation memory and logger.

    Args:
        memory: Conversation memory instance.
        logger_obj: Logger instance.
    """
    reset_memory(memory)
    reset_logger(logger_obj)

    logger.info("Pipeline reset completed.")


def reset_session(
    memory: ConversationMemory,
    logger_obj: Optional[RAGLogger] = None,
    logger: Optional[RAGLogger] = None,
) -> None:
    """Reset the current chatbot session.

    Args:
        memory: Conversation memory instance.
        logger_obj: Logger instance.
        logger: Logger instance alias.
    """
    log_inst = logger if logger is not None else logger_obj
    reset_memory(memory)
    if log_inst is not None:
        reset_logger(log_inst)

    logging.getLogger(__name__).info("Session reset successfully.")
