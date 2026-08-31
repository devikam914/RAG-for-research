import hashlib
from io import BytesIO
from typing import Any
from pypdf import PdfReader
from src.config import CHUNK_OVERLAP, CHUNK_WORDS


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def extract_chunks_from_pdf(
    pdf_bytes: bytes,
    filename: str,
    chunk_words: int = CHUNK_WORDS,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    reader = PdfReader(BytesIO(pdf_bytes))
    chunks: list[dict[str, Any]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").replace("\x00", " ")
        words = text.split()
        if not words:
            continue

        start = 0
        while start < len(words):
            end = min(start + chunk_words, len(words))
            chunk_text = " ".join(words[start:end]).strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "source": filename,
                    "page": page_number,
                })

            if end == len(words):
                break

            start = end - chunk_overlap

    return chunks
