"""Text processing utilities."""

import hashlib
from typing import List

from src.config.constants import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


def chunk_text(text: str, size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    text = text.strip()
    if not text:
        return []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def sanitize_collection_name(topic: str) -> str:
    """Create valid ChromaDB collection name from topic."""
    h = hashlib.md5(topic.encode()).hexdigest()[:8]
    clean = "".join(c if c.isalnum() else "_" for c in topic[:20])
    return f"topic_{clean}_{h}"