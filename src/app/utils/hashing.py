"""
Content hashing utilities for document deduplication.
Provides deterministic hashing based on document content and metadata.
"""

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document


def content_hash(doc: "Document") -> str:
    """
    Generate a deterministic SHA256 hash from document content and metadata.

    The hash is based on:
    - page_content
    - source (from metadata)
    - page (from metadata)
    - chunk (from metadata)

    Args:
        doc: LangChain Document instance

    Returns:
        64-character hexadecimal SHA256 hash string
    """
    payload = {
        "content": doc.page_content,
        "source": doc.metadata.get("source"),
        "page": doc.metadata.get("page"),
        "chunk": doc.metadata.get("chunk"),
    }

    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
