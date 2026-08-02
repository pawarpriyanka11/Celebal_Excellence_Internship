# Document Question Answering System (RAG)

This project implements a simple Retrieval-Augmented Generation (RAG) system for answering questions from your own documents.

## Features

- Upload PDF or text files
- Split documents into chunks
- Create embeddings with a sentence-transformer model
- Retrieve the most relevant chunks using FAISS
- Generate answers with a local language model
- Provide a simple Streamlit web app for interaction

## Setup

1. Create a virtual environment
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app
   ```bash
   streamlit run app.py
   ```

## Notes

- The first run will download the embedding and generation models. This can take a few minutes depending on your internet connection.
- If the local LLM download is too slow, the app falls back to a simple keyword-based answer generator so the pipeline still works.
- You can also place your own PDFs or text files in the `data/` folder for testing.
