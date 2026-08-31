import os

GENERATION_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
EMBEDDING_MODEL: str = "gemini-embedding-001"
EMBEDDING_DIMENSION: int = 768
CHUNK_WORDS: int = 420
CHUNK_OVERLAP: int = 70
TOP_K: int = 6
DEFAULT_TEMPERATURE: float = 0.2
