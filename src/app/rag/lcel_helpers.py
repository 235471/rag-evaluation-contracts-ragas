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


def llm_rerank_docs(
    docs: List["Document"],
    query: str,
    llm=None,
    top_k: int = 3,
) -> List["Document"]:
    """
    Rerank documents using an LLM as judge.

    Evaluates each document's relevance to the query on a 1-10 scale,
    then returns the top_k highest-scored documents.

    Args:
        docs: List of Document objects to rerank
        query: The user's query
        llm: LLM to use for scoring (defaults to gemma-3-27b-it via Gemini)
        top_k: Number of top documents to return

    Returns:
        List of top_k most relevant Document objects
    """
    from langchain_core.output_parsers import StrOutputParser
    from src.app.rag.prompts import get_reranker_prompt
    from src.app.logging_conf import get_logger

    logger = get_logger(__name__)

    if llm is None:
        from src.app.config import get_llm

        llm = get_llm(provider="gemini", model="gemma-3-27b-it", temperature=0.1)

    reranker_prompt = get_reranker_prompt()
    reranker_chain = reranker_prompt | llm | StrOutputParser()

    # Prepare batch inputs for parallel processing
    batch_inputs = [{"query": query, "document": doc.page_content} for doc in docs]

    # Execute all scoring requests in parallel
    logger.debug(f"Scoring {len(docs)} documents in parallel using batch()...")
    try:
        score_strings = reranker_chain.batch(batch_inputs)
    except Exception as e:
        logger.error(f"Batch processing failed: {e}, falling back to sequential")
        # Fallback to sequential processing if batch fails
        score_strings = []
        for inp in batch_inputs:
            try:
                score_strings.append(reranker_chain.invoke(inp))
            except Exception as doc_error:
                logger.warning(f"Failed to score document: {doc_error}")
                score_strings.append("5")  # Default score

    # Parse scores and pair with documents
    scored_docs = []
    for i, (score_str, doc) in enumerate(zip(score_strings, docs)):
        try:
            # Parse score (handle potential text around the number)
            score = int("".join(filter(str.isdigit, score_str[:5])) or "5")
            score = max(1, min(10, score))  # Clamp to 1-10
            scored_docs.append((score, doc))
            logger.debug(f"Doc {i+1}/{len(docs)}: score={score}")
        except Exception as e:
            logger.warning(
                f"Failed to parse score for doc {i+1}: {e}, assigning default score 5"
            )
            scored_docs.append((5, doc))

    # Sort by score descending and take top_k
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    reranked = [doc for _, doc in scored_docs[:top_k]]

    logger.info(
        f"Reranked {len(docs)} docs -> top {len(reranked)} (scores: {[s for s, _ in scored_docs[:top_k]]})"
    )
    return reranked


def create_reranker_lambda(llm=None, top_k: int = 3):
    """
    Create a RunnableLambda-compatible reranker function.

    This is used in LCEL chains where the input is a dict with 'query' and 'docs_raw'.

    Args:
        llm: LLM for reranking (defaults to gemma-3-27b-it)
        top_k: Number of top documents to keep

    Returns:
        A function compatible with RunnableLambda
    """

    def _rerank(x: Dict[str, Any]) -> List["Document"]:
        return llm_rerank_docs(
            docs=x["docs_raw"],
            query=x["query"],
            llm=llm,
            top_k=top_k,
        )

    return _rerank
