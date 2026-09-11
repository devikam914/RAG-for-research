import os
from dotenv import load_dotenv

load_dotenv()

GENERATION_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
EMBEDDING_MODEL: str = "gemini-embedding-001"
EMBEDDING_DIMENSION: int = 768
CHUNK_WORDS: int = 420
CHUNK_OVERLAP: int = 70
TOP_K: int = 6
DEFAULT_TEMPERATURE: float = 0.2

CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "research_papers")
