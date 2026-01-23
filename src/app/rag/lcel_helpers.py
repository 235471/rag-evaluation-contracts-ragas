"""
LCEL helper functions for RAG chains.
Extracted from notebook2.md for use with RunnableLambda.
"""

from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document


def docs_chunks_queried(x: Dict[str, Any]) -> List[str]:
    """
    Extract page_content from docs in a RunnableMap result.

    Expected input: {"docs": [Document, ...], "query": "..."}

    Args:
        x: Dict containing 'docs' key with list of Documents

    Returns:
        List of page_content strings
    """
    return [d.page_content for d in x["docs"]]


def join_contexts(contexts: List[str]) -> str:
    """
    Join a list of context strings with double newlines.

    Args:
        contexts: List of context strings

    Returns:
        Single joined string
    """
    return "\n\n".join(contexts)


def docs_to_text(docs: List["Document"]) -> str:
    """
    Convert a list of Documents to a single text string.

    Args:
        docs: List of Document objects

    Returns:
        Concatenated page_content of all documents
    """
    return "\n\n".join([doc.page_content for doc in docs])


def get_unique_documents(queries: List[str], retriever) -> List["Document"]:
    """
    Retrieve documents for multiple queries and deduplicate by content.

    Uses retriever.batch() to run all queries in parallel, then flattens
    and deduplicates by page_content.

    Args:
        queries: List of query strings
        retriever: Retriever instance with batch() method

    Returns:
        List of unique Document objects
    """
    # Run all queries at once
    list_of_doc_lists = retriever.batch(queries)

    # Flatten
    flattened_docs = [doc for sublist in list_of_doc_lists for doc in sublist]

    # Deduplicate by page_content to save LLM tokens
    unique_docs = {doc.page_content: doc for doc in flattened_docs}.values()
    return list(unique_docs)


def extract_query(x: Dict[str, Any]) -> str:
    """
    Extract query from a RunnableMap result.

    Args:
        x: Dict containing 'query' key

    Returns:
        Query string
    """
    return x["query"]


def format_docs_for_context(docs: List["Document"]) -> str:
    """
    Format documents as a context string for the LLM prompt.

    Same as docs_to_text but with explicit naming for clarity.

    Args:
        docs: List of Document objects

    Returns:
        Formatted context string
    """
    return "\n\n".join([doc.page_content for doc in docs])
