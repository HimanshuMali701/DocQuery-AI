# 🤖 RAG PDF Chatbot

A Retrieval-Augmented Generation (RAG) based PDF Chatbot built using Streamlit, LangChain, FAISS, Hugging Face Embeddings, and Groq LLM.

## Features

- 📄 Chat with one or more PDF documents
- 🔍 Semantic document retrieval using FAISS
- 🧠 Conversation memory
- 📚 Source citations
- 📈 Query logging
- 💬 ChatGPT-inspired Streamlit UI

## Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- Hugging Face Embeddings
- Groq LLM

## Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

Run the application:

```bash
streamlit run app.py
```

## Project Structure

```
app.py
backend.py
requirements.txt
uploaded_pdfs/
faiss_index/
```
