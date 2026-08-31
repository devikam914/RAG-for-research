from typing import Literal
import numpy as np
from google import genai
from google.genai import types
from src.config import EMBEDDING_DIMENSION, EMBEDDING_MODEL

TaskType = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY", "SEMANTIC_SIMILARITY", "CLASSIFICATION", "CLUSTERING"]


def embed_texts(
    client: genai.Client,
    texts: list[str],
    task_type: TaskType = "RETRIEVAL_DOCUMENT",
    model: str = EMBEDDING_MODEL,
    dimension: int = EMBEDDING_DIMENSION,
    batch_size: int = 32,
) -> np.ndarray:
    if not texts:
        return np.empty((0, dimension), dtype=np.float32)

    vectors: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = client.models.embed_content(
            model=model,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=dimension,
            ),
        )
        vectors.extend([item.values for item in result.embeddings])

    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)
