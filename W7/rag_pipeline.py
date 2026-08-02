import os
import re
from typing import List, Tuple

import numpy as np
from faiss import IndexFlatL2
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


class DocumentRAG:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.embedder = SentenceTransformer(model_name)
        self.chunks: List[str] = []
        self.embeddings: np.ndarray | None = None
        self.index = None

    def load_documents(self, paths: List[str]) -> None:
        texts = []
        for path in paths:
            if not os.path.exists(path):
                continue
            if path.lower().endswith(".pdf"):
                texts.append(self._read_pdf(path))
            else:
                texts.append(self._read_text(path))
        self.chunks = self._chunk_text("\n\n".join(texts))
        self._build_index()

    def _read_pdf(self, path: str) -> str:
        reader = PdfReader(path)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
        return "\n".join(text_parts)

    def _read_text(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return []
        words = cleaned.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            if chunk:
                chunks.append(chunk)
            if end == len(words):
                break
            start = max(0, end - overlap)
        return chunks

    def _build_index(self) -> None:
        if not self.chunks:
            self.embeddings = np.empty((0, 0), dtype=np.float32)
            self.index = None
            return
        self.embeddings = self.embedder.encode(self.chunks, convert_to_numpy=True, normalize_embeddings=True)
        self.embeddings = np.asarray(self.embeddings, dtype=np.float32)
        self.index = IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        if not self.chunks or self.index is None or self.embeddings is None:
            return []
        query_embedding = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append((self.chunks[int(idx)], float(dist)))
        return results

    def answer(self, query: str, top_k: int = 3) -> str:
        retrieved = self.retrieve(query, top_k=top_k)
        if not retrieved:
            return "No relevant documents were found."

        context = "\n\n".join([text for text, _ in retrieved])
        prompt = (
            "You are a helpful assistant. Answer the user's question using only the provided context.\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )

        try:
            from transformers import pipeline

            generator = pipeline("text2text-generation", model="google/flan-t5-small")
            response = generator(prompt, max_new_tokens=120, do_sample=False)[0]["generated_text"]
            return response.strip()
        except Exception:
            return (
                "Based on the available context, the answer is: "
                + self._fallback_answer(query, retrieved)
            )

    def _fallback_answer(self, query: str, retrieved: List[Tuple[str, float]]) -> str:
        query_words = set(re.sub(r"[^a-z0-9]+", " ", query.lower()).split())
        best_match = ""
        best_score = -1
        for chunk, _ in retrieved:
            chunk_words = set(re.sub(r"[^a-z0-9]+", " ", chunk.lower()).split())
            overlap = len(query_words & chunk_words)
            if overlap > best_score:
                best_score = overlap
                best_match = chunk
        return best_match[:300] if best_match else "No relevant content available."
