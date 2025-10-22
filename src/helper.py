"""Utility helpers for the Streamlit information retrieval app."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import faiss
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader

load_dotenv()

_EMBED_MODEL = "models/text-embedding-004"
_CHAT_MODEL = "gemini-1.5-flash"


def _configure_client() -> None:
    """Ensure the Generative AI client is configured with an API key."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Please add it to your environment or .env file."
        )
    if not getattr(_configure_client, "_configured", False):
        genai.configure(api_key=api_key)
        _configure_client._configured = True  # type: ignore[attr-defined]


def get_pdf_text(pdf_docs: Iterable) -> str:
    """Read multiple uploaded PDF files and return their concatenated text."""
    if not pdf_docs:
        return ""

    collected: list[str] = []
    for pdf in pdf_docs:
        pdf.seek(0)
        reader = PdfReader(pdf)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                collected.append(text)
    return "\n".join(collected)


def get_text_chunks(
    raw_text: str,
    *,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[str]:
    """Split raw text into overlapping chunks suitable for retrieval."""
    if not raw_text:
        return []
    if overlap >= chunk_size:
        raise ValueError("`overlap` must be smaller than `chunk_size`.")

    chunks: list[str] = []
    step = chunk_size - overlap
    length = len(raw_text)
    for start in range(0, length, step):
        end = min(start + chunk_size, length)
        chunk = raw_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
    return chunks


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if not norm:
        return vector
    return vector / norm


class VectorStore:
    """Thin FAISS-based wrapper to retrieve the most relevant text chunks."""

    def __init__(self, texts: Sequence[str], index: faiss.Index):
        self._texts = list(texts)
        self._index = index

    def similarity_search(self, query: str, *, k: int = 3) -> list[str]:
        if not self._texts:
            return []
        _configure_client()
        query_vector = np.asarray(
            genai.embed_content(model=_EMBED_MODEL, content=query)["embedding"],
            dtype="float32",
        )
        query_vector = _normalize(query_vector)
        _, indices = self._index.search(query_vector.reshape(1, -1), min(k, len(self._texts)))
        return [
            self._texts[i]
            for i in indices[0]
            if 0 <= i < len(self._texts)
        ]


def get_vector_store(text_chunks: Sequence[str]) -> VectorStore:
    """Create a FAISS vector store using Gemini embeddings."""
    filtered_chunks = [chunk for chunk in text_chunks if chunk.strip()]
    if not filtered_chunks:
        raise ValueError("No textual content found to build the vector store.")

    _configure_client()

    vectors: list[np.ndarray] = []
    for chunk in filtered_chunks:
        embedding = np.asarray(
            genai.embed_content(model=_EMBED_MODEL, content=chunk)["embedding"],
            dtype="float32",
        )
        vectors.append(_normalize(embedding))

    matrix = np.vstack(vectors).astype("float32")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return VectorStore(filtered_chunks, index)


@dataclass
class Message:
    role: str
    content: str


class ConversationalRetrieval:
    """Callable conversation helper compatible with the Streamlit app."""

    def __init__(self, vector_store: VectorStore, *, k: int = 3):
        _configure_client()
        self._vector_store = vector_store
        self._model = genai.GenerativeModel(_CHAT_MODEL)
        self._history: list[Message] = []
        self._k = k

    def __call__(self, payload: dict[str, str]) -> dict[str, list[Message] | str]:
        question = (payload or {}).get("question", "").strip()
        if not question:
            return {"answer": "", "chat_history": list(self._history)}

        context_chunks = self._vector_store.similarity_search(question, k=self._k)
        prompt = self._build_prompt(question, context_chunks)
        response = self._model.generate_content(prompt)
        answer = getattr(response, "text", str(response))

        self._history.append(Message(role="user", content=question))
        self._history.append(Message(role="assistant", content=answer))

        return {"answer": answer, "chat_history": list(self._history)}

    def _build_prompt(self, question: str, contexts: Sequence[str]) -> str:
        history = "\n".join(
            f"{msg.role.title()}: {msg.content}" for msg in self._history[-6:]
        )
        context_section = "\n\n".join(f"- {ctx}" for ctx in contexts) or "No relevant context available."
        prompt_parts = [
            "You are a helpful assistant that answers questions based strictly on the provided context.",
            "If the answer is not present in the context, reply with \"I'm not sure\".",
        ]
        if history:
            prompt_parts.append(f"Conversation so far:\n{history}")
        prompt_parts.append(f"Context:\n{context_section}")
        prompt_parts.append(f"Question: {question}")
        prompt_parts.append("Answer in a concise paragraph:")
        return "\n\n".join(prompt_parts)


def get_conversational_chain(vector_store: VectorStore) -> ConversationalRetrieval:
    """Return a conversational retrieval helper compatible with the Streamlit app."""
    return ConversationalRetrieval(vector_store)


__all__ = [
    "get_pdf_text",
    "get_text_chunks",
    "get_vector_store",
    "get_conversational_chain",
]
