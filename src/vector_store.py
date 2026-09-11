import os
from typing import Any, Optional
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from google import genai

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    TOP_K,
)
from src.document_processor import extract_chunks_from_pdf
from src.embeddings import embed_texts


_chroma_client: Optional[ClientAPI] = None


def get_chroma_client(persist_dir: str = CHROMA_PERSIST_DIR) -> ClientAPI:
    """Returns a persistent ChromaDB client instance."""
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(persist_dir, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=persist_dir)
    return _chroma_client


def get_or_create_collection(
    client: Optional[ClientAPI] = None,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> Collection:
    """Returns or creates a ChromaDB collection configured for cosine similarity."""
    chroma = client or get_chroma_client()
    return chroma.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def reset_vector_collection(
    client: Optional[ClientAPI] = None,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> Collection:
    """Deletes existing collection if present and creates a fresh collection."""
    chroma = client or get_chroma_client()
    try:
        chroma.delete_collection(name=collection_name)
    except Exception:
        pass
    return chroma.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def build_vector_index(
    library: dict[str, dict[str, Any]],
    client: genai.Client,
    chroma_client: Optional[ClientAPI] = None,
    collection_name: str = CHROMA_COLLECTION_NAME,
    batch_size: int = 500,
) -> int:
    """
    Extracts text chunks from PDFs in the library, generates embeddings,
    and indexes them into ChromaDB.
    Returns the total number of chunks indexed.
    """
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

    # Reset collection to ensure clean state matching the library
    collection = reset_vector_collection(client=chroma_client, collection_name=collection_name)

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [
        {"source": str(chunk["source"]), "page": int(chunk["page"])}
        for chunk in chunks
    ]
    embeddings_list = embeddings.tolist() if hasattr(embeddings, "tolist") else list(embeddings)

    total_chunks = len(chunks)
    for i in range(0, total_chunks, batch_size):
        end = min(i + batch_size, total_chunks)
        collection.add(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            embeddings=embeddings_list[i:end],
        )

    return total_chunks


def search_similar_chunks(
    query: str,
    client: genai.Client,
    top_k: int = TOP_K,
    collection: Optional[Collection] = None,
    chroma_client: Optional[ClientAPI] = None,
    collection_name: str = CHROMA_COLLECTION_NAME,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """
    Searches ChromaDB for chunks similar to the query using cosine distance.
    Returns a list of chunk dicts containing 'text', 'source', 'page', and 'score'.
    """
    col = collection or get_or_create_collection(client=chroma_client, collection_name=collection_name)
    total_count = col.count()
    if total_count == 0:
        return []

    query_vectors = embed_texts(client, [query], task_type="RETRIEVAL_QUERY")
    query_vector = query_vectors[0].tolist() if hasattr(query_vectors[0], "tolist") else list(query_vectors[0])

    n_results = min(top_k, total_count)
    results = col.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    matched_chunks: list[dict[str, Any]] = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        # In cosine distance space, similarity score = 1.0 - distance
        score = max(0.0, float(1.0 - dist))
        metadata_dict = meta or {}
        matched_chunks.append({
            "text": doc,
            "source": metadata_dict.get("source", "Unknown"),
            "page": metadata_dict.get("page", 1),
            "score": score,
        })

    return matched_chunks
