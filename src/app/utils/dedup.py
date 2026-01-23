"""
Document deduplication utilities.
Provides functions to deduplicate documents by content or hash.
"""

from typing import List, TYPE_CHECKING

from src.app.utils.hashing import content_hash

if TYPE_CHECKING:
    from langchain_core.documents import Document


def dedup_by_content(docs: List["Document"]) -> List["Document"]:
    """
    Remove duplicate documents based on page_content.

    Keeps the first occurrence of each unique content.

    Args:
        docs: List of Document objects

    Returns:
        List of unique documents
    """
    seen = {}
    for doc in docs:
        if doc.page_content not in seen:
            seen[doc.page_content] = doc
    return list(seen.values())


def dedup_by_hash(docs: List["Document"]) -> List["Document"]:
    """
    Remove duplicate documents based on content_hash metadata.

    Documents must have 'content_hash' in metadata (use add_content_hash_to_docs first).
    Keeps the first occurrence of each unique hash.

    Args:
        docs: List of Document objects with content_hash in metadata

    Returns:
        List of unique documents
    """
    seen = {}
    for doc in docs:
        doc_hash = doc.metadata.get("content_hash")
        if doc_hash and doc_hash not in seen:
            seen[doc_hash] = doc
    return list(seen.values())


def add_content_hash_to_docs(docs: List["Document"]) -> List["Document"]:
    """
    Add content_hash to each document's metadata.

    Uses the deterministic content_hash function based on
    page_content, source, page, and chunk.

    Args:
        docs: List of Document objects

    Returns:
        Same list of documents with content_hash added to metadata
    """
    for doc in docs:
        doc.metadata["content_hash"] = content_hash(doc)
    return docs


def get_content_hash_ids(docs: List["Document"]) -> List[str]:
    """
    Extract content_hash from each document to use as IDs (e.g., for Pinecone).

    Documents must have 'content_hash' in metadata.

    Args:
        docs: List of Document objects with content_hash in metadata

    Returns:
        List of content_hash strings

    Raises:
        ValueError: If any document is missing content_hash
    """
    ids = []
    for i, doc in enumerate(docs):
        hash_val = doc.metadata.get("content_hash")
        if not hash_val:
            raise ValueError(f"Document at index {i} missing content_hash in metadata")
        ids.append(hash_val)
    return ids
