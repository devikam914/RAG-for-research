from typing import Any
import numpy as np
from google import genai
from src.config import TOP_K
from src.document_processor import extract_chunks_from_pdf
from src.embeddings import embed_texts


def build_vector_index(
    library: dict[str, dict[str, Any]],
    client: genai.Client,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    chunks: list[dict[str, Any]] = []

    for item in library.values():
        chunks.extend(extract_chunks_from_pdf(item["bytes"], item["name"]))

    if not chunks:
        raise ValueError(
            "No text could be extracted from the uploaded files. "
            "The PDFs may be scanned images or empty."
        )

    embedding_inputs = [
        f"title: {item['source']} | text: {item['text']}"
        for item in chunks
    ]

    embeddings = embed_texts(client, embedding_inputs, task_type="RETRIEVAL_DOCUMENT")

    return chunks, embeddings


def search_similar_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
    client: genai.Client,
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    if not chunks or embeddings.size == 0:
        return []

    query_vector = embed_texts(client, [query], task_type="RETRIEVAL_QUERY")[0]
    scores = embeddings @ query_vector
    top_indices = np.argsort(scores)[::-1][:min(top_k, len(chunks))]

    return [
        {**chunks[i], "score": float(scores[i])}
        for i in top_indices
    ]
