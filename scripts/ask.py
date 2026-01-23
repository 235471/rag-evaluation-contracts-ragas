#!/usr/bin/env python3
"""
CLI script to ask questions using the RAG system.
Uses existing vectorstore - does NOT run ingestão or bootstrap.

Usage:
    python scripts/ask.py "Como fazer um seguro viagem?"
    python scripts/ask.py "Qual o limite do cartão?" --backend postgres
    python scripts/ask.py "..." --chain-type multiquery
"""

import argparse
import sys
from pathlib import Path

# Fix for psycopg3 on Windows
if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app.logging_conf import setup_logging, get_logger
from src.app.config import get_settings, get_llm, get_embeddings


def main():
    parser = argparse.ArgumentParser(description="Ask a question using the RAG system")
    parser.add_argument(
        "question",
        type=str,
        help="The question to ask",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["postgres", "pinecone"],
        default=None,
        help="Vectorstore backend (default: from VECTORSTORE_BACKEND env)",
    )
    parser.add_argument(
        "--chain-type",
        type=str,
        choices=["base", "rewriter", "multiquery", "hyde", "full"],
        default="full",
        help="Type of RAG chain to use (default: full for rag_chain_with_context_full)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="Number of documents to retrieve (default: 2)",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        choices=["gemini", "groq", "ollama", "perplexity"],
        default=None,
        help="LLM provider (default: from LLM_PROVIDER env)",
    )
    parser.add_argument(
        "--show-contexts",
        action="store_true",
        help="Show retrieved contexts in output",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    logger = get_logger(__name__)

    settings = get_settings()
    backend = args.backend or settings.VECTORSTORE_BACKEND

    logger.info(f"Using backend: {backend}")
    logger.info(f"Chain type: {args.chain_type}")

    try:
        # Get vectorstore
        embeddings = get_embeddings()

        if backend == "postgres":
            from src.app.vectorstores.postgres_pgvector import get_pgvector_store

            vector_store = get_pgvector_store(embeddings=embeddings)
        else:
            from src.app.vectorstores.pinecone_store import get_pinecone_store

            vector_store = get_pinecone_store(embeddings=embeddings)

        retriever = vector_store.as_retriever(search_kwargs={"k": args.k})
        llm = get_llm(provider=args.llm_provider)

        # Build appropriate chain
        from src.app.rag.chains import (
            build_base_rag_chain,
            build_rewriter_rag_chain,
            build_multi_query_chain,
            build_hyde_chain,
            build_rag_chain_with_context_full,
        )

        if args.chain_type == "base":
            chain = build_base_rag_chain(retriever, llm, k=args.k)
            # Base chain returns string, wrap for consistency
            result = chain.invoke(args.question)
            output = {"query": args.question, "answer": result, "contexts": []}

        elif args.chain_type == "rewriter":
            chain = build_rewriter_rag_chain(retriever, llm)
            result = chain.invoke(args.question)
            output = {"query": args.question, "answer": result, "contexts": []}

        elif args.chain_type == "multiquery":
            # Multi-query needs a model that handles list parsing
            multi_llm = get_llm(provider="groq")  # Better for structured output
            chain = build_multi_query_chain(retriever, llm, multi_query_llm=multi_llm)
            result = chain.invoke(args.question)
            output = {"query": args.question, "answer": result, "contexts": []}

        elif args.chain_type == "hyde":
            chain = build_hyde_chain(retriever, llm)
            result = chain.invoke(args.question)
            output = {"query": args.question, "answer": result, "contexts": []}

        else:  # full
            chain = build_rag_chain_with_context_full(retriever, llm)
            output = chain.invoke(args.question)

        # Display results
        print("\n" + "=" * 60)
        print("📝 PERGUNTA:")
        print(output.get("query", args.question))
        print("\n" + "-" * 60)
        print("💬 RESPOSTA:")
        print(output.get("answer", ""))

        if args.show_contexts and output.get("contexts"):
            print("\n" + "-" * 60)
            print("📚 CONTEXTOS RECUPERADOS:")
            for i, ctx in enumerate(output["contexts"], 1):
                print(f"\n[{i}] {ctx[:500]}{'...' if len(ctx) > 500 else ''}")

        print("\n" + "=" * 60)

    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"\n❌ Erro: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
