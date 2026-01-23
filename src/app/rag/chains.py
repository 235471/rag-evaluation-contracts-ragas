"""
LCEL chain builders for RAG.
All chains preserve the structure from notebook2.md.
"""

from operator import itemgetter
from typing import Optional

from langchain_core.output_parsers import (
    StrOutputParser,
    CommaSeparatedListOutputParser,
)
from langchain_core.runnables import (
    RunnableMap,
    RunnableAssign,
    RunnableLambda,
    RunnablePassthrough,
)

from src.app.rag.prompts import (
    get_rag_prompt,
    get_rewriter_prompt,
    get_multi_query_prompt,
    get_hyde_prompt,
)
from src.app.rag.lcel_helpers import (
    docs_chunks_queried,
    join_contexts,
    docs_to_text,
    get_unique_documents,
)
from src.app.logging_conf import get_logger

logger = get_logger(__name__)


def build_answer_chain(llm):
    """
    Build the base answer chain (prompt | llm | parser).

    Args:
        llm: Language model instance

    Returns:
        LCEL chain that takes {context, query} and returns answer string
    """
    prompt = get_rag_prompt()
    return prompt | llm | StrOutputParser()


def build_base_rag_chain(retriever, llm, k: int = 10):
    """
    Build a basic RAG chain.

    Structure: retriever -> format context -> prompt -> llm -> parser

    Args:
        retriever: Vector store retriever
        llm: Language model instance
        k: Number of documents to retrieve (updates retriever search_kwargs)

    Returns:
        LCEL chain that takes query string and returns answer string
    """
    # Update retriever k if needed
    if hasattr(retriever, "search_kwargs"):
        retriever.search_kwargs["k"] = k

    answer_chain = build_answer_chain(llm)

    def format_context(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = {
        "context": retriever | RunnableLambda(format_context),
        "query": RunnablePassthrough(),
    } | answer_chain

    logger.debug("Built base RAG chain")
    return rag_chain


def build_rewriter_rag_chain(retriever, llm, rewriter_llm=None):
    """
    Build a RAG chain with query rewriting.

    Structure:
    1. Rewrite query using rewriter_llm
    2. Use rewritten query for retrieval
    3. Answer with retrieved context

    Args:
        retriever: Vector store retriever
        llm: Language model for answering
        rewriter_llm: Language model for rewriting (defaults to llm)

    Returns:
        LCEL chain that takes query string and returns answer string
    """
    rewriter_llm = rewriter_llm or llm

    rewriter_prompt = get_rewriter_prompt()
    answer_chain = build_answer_chain(llm)

    # Rewriter chain: prompt -> llm -> parser
    rewriter_chain = rewriter_prompt | rewriter_llm | StrOutputParser()

    def format_context(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain_rewriter = {
        "context": rewriter_chain | retriever | RunnableLambda(format_context),
        "query": RunnablePassthrough(),
    } | answer_chain

    logger.debug("Built rewriter RAG chain")
    return rag_chain_rewriter


def build_multi_query_chain(retriever, llm, multi_query_llm=None):
    """
    Build a RAG chain with multi-query augmented retrieval.

    Structure:
    1. Generate 5 versions of the query
    2. Batch retrieve for all queries
    3. Flatten and deduplicate documents
    4. Answer with unique context

    Args:
        retriever: Vector store retriever
        llm: Language model for answering
        multi_query_llm: Language model for generating queries (defaults to llm)

    Returns:
        LCEL chain that takes query string and returns answer string
    """
    multi_query_llm = multi_query_llm or llm

    multi_query_prompt = get_multi_query_prompt()
    answer_chain = build_answer_chain(llm)

    # Multi-query generation chain
    multi_query_chain = (
        multi_query_prompt | multi_query_llm | CommaSeparatedListOutputParser()
    )

    # Function to get unique docs from multiple queries
    def get_unique_docs_from_queries(queries):
        return get_unique_documents(queries, retriever)

    modern_multi_query_chain = {
        "context": (
            multi_query_chain
            | RunnableLambda(get_unique_docs_from_queries)
            | RunnableLambda(docs_to_text)
        ),
        "query": RunnablePassthrough(),
    } | answer_chain

    logger.debug("Built multi-query RAG chain")
    return modern_multi_query_chain


def build_hyde_chain(retriever, llm, hyde_llm=None):
    """
    Build a RAG chain with HyDE (Hypothetical Document Embeddings).

    Structure:
    1. Generate a hypothetical answer paragraph
    2. Use hypothetical paragraph for retrieval
    3. Answer with real retrieved context

    Args:
        retriever: Vector store retriever
        llm: Language model for answering
        hyde_llm: Language model for generating hypothetical doc (defaults to llm)

    Returns:
        LCEL chain that takes query string and returns answer string
    """
    hyde_llm = hyde_llm or llm

    hyde_prompt = get_hyde_prompt()
    answer_chain = build_answer_chain(llm)

    # HyDE generation chain
    hyde_generation = hyde_prompt | hyde_llm | StrOutputParser()

    hyde_chain = {
        "context": (hyde_generation | retriever | RunnableLambda(docs_to_text)),
        "query": RunnablePassthrough(),
    } | answer_chain

    logger.debug("Built HyDE RAG chain")
    return hyde_chain


def build_rag_chain_with_context_full(retriever, llm):
    """
    Build the full RAG chain that returns query, answer, contexts, and docs.

    This is the CHOSEN version per requirements - uses RunnableAssign to
    derive contexts from docs.

    Structure:
    1. Input: query string
    2. Retrieve docs
    3. Extract contexts from docs (page_content list)
    4. Generate answer using joined contexts
    5. Return: {"query": ..., "answer": ..., "contexts": [...], "docs": [...]}

    Args:
        retriever: Vector store retriever
        llm: Language model instance

    Returns:
        LCEL chain that takes query string and returns dict with query, answer, contexts, docs
    """
    answer_chain = build_answer_chain(llm)

    rag_chain_with_context_full = (
        # 1️⃣ Input: query string -> {"query": query, "docs": [docs]}
        RunnableMap(
            {
                "query": RunnablePassthrough(),
                "docs": retriever,
            }
        )
        # 2️⃣ Derive contexts from docs
        | RunnableAssign({"contexts": RunnableLambda(docs_chunks_queried)})
        # 3️⃣ Generate answer using contexts
        | RunnableAssign(
            {
                "answer": (
                    {
                        "context": itemgetter("contexts")
                        | RunnableLambda(join_contexts),
                        "query": itemgetter("query"),
                    }
                    | answer_chain
                )
            }
        )
    )

    logger.debug("Built rag_chain_with_context_full")
    return rag_chain_with_context_full


# Convenience function to get a default chain
def get_default_rag_chain(
    vectorstore_backend: str = "postgres",
    llm_provider: Optional[str] = None,
    k: int = 10,
    with_context: bool = True,
):
    """
    Get a default RAG chain using configured vectorstore and LLM.

    Args:
        vectorstore_backend: "postgres" or "pinecone"
        llm_provider: LLM provider override
        k: Number of documents to retrieve
        with_context: If True, returns rag_chain_with_context_full; otherwise base chain

    Returns:
        Configured LCEL chain
    """
    from src.app.config import get_llm, get_embeddings

    # Get vectorstore
    if vectorstore_backend == "postgres":
        from src.app.vectorstores.postgres_pgvector import get_pgvector_store

        embeddings = get_embeddings()
        vector_store = get_pgvector_store(embeddings=embeddings)
    else:
        from src.app.vectorstores.pinecone_store import get_pinecone_store

        embeddings = get_embeddings()
        vector_store = get_pinecone_store(embeddings=embeddings)

    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    llm = get_llm(provider=llm_provider)

    if with_context:
        return build_rag_chain_with_context_full(retriever, llm)
    else:
        return build_base_rag_chain(retriever, llm, k=k)
