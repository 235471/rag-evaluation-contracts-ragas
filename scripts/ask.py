#!/usr/bin/env python3
"""
CLI script to ask questions using the RAG system.
Uses existing vectorstore - does NOT run ingestão or bootstrap.

Features:
    - Prompt injection guard: 3-layer defense (keywords, patterns, LLM)
    - PII guardrails: sanitizes sensitive data before processing
    - Semantic cache: avoids repeated LLM calls for similar questions
    - BM25 fallback: FAQ-based fallback on chain errors

Usage:
    python scripts/ask.py "Como fazer um seguro viagem?"
    python scripts/ask.py "Qual o limite do cartão?" --backend postgres
    python scripts/ask.py "..." --chain-type multiquery
    python scripts/ask.py "..." --no-cache --no-guard
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
        choices=["base", "rewriter", "multiquery", "hyde", "full", "rerank"],
        default="full",
        help="Type of RAG chain to use (default: full, rerank uses LLM judge)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="Number of documents to retrieve (default: 2, ignored for rerank)",
    )
    parser.add_argument(
        "--initial-k",
        type=int,
        default=20,
        help="Initial candidates for rerank chain (default: 20)",
    )
    parser.add_argument(
        "--rerank-k",
        type=int,
        default=3,
        help="Top docs to keep after reranking (default: 3)",
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
        "--no-cache",
        action="store_true",
        help="Skip semantic cache lookup and storage",
    )
    parser.add_argument(
        "--no-pii",
        action="store_true",
        help="Skip PII detection and sanitization",
    )
    parser.add_argument(
        "--no-guard",
        action="store_true",
        help="Skip prompt injection detection",
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
    question = args.question

    logger.info(f"Using backend: {backend}")
    logger.info(f"Chain type: {args.chain_type}")

    # --- Prompt Injection Guard ---
    if not args.no_guard:
        try:
            from src.app.guardrails.prompt_guard import get_prompt_guard

            guard = get_prompt_guard()
            guard_result = guard.classify(question)

            if not guard_result.is_safe:
                reason = guard_result.blocked_reason
                print(f"\n🚫 Solicitação bloqueada (Camada {reason.layer})")
                print(f"   Motivo: {reason.reason}")
                print(f"   Detalhe: {reason.detail}")
                logger.warning(
                    f"Prompt blocked: layer={reason.layer}, "
                    f"reason={reason.reason}, detail={reason.detail}"
                )
                sys.exit(0)
        except Exception as e:
            logger.warning(f"Prompt guard check failed (continuing): {e}")

    # --- PII Guardrail ---
    if not args.no_pii:
        try:
            from src.app.guardrails.pii_filter import get_pii_guardrail

            guardrail = get_pii_guardrail()
            detected_types = guardrail.get_detected_types(question)

            if detected_types:
                print(f"\n⚠️  PII detectado: {', '.join(detected_types)}")
                question = guardrail.sanitize(question)
                print(f"🔒 Input sanitizado antes do processamento")
                logger.warning(f"PII sanitized: {detected_types}")
        except ImportError:
            logger.debug(
                "Presidio not installed, skipping PII check. "
                "Install with: pip install presidio-analyzer presidio-anonymizer"
            )
        except Exception as e:
            logger.warning(f"PII check failed (continuing without): {e}")

    # --- Semantic Cache Lookup ---
    cache = None
    if not args.no_cache:
        try:
            from src.app.cache.semantic_cache import get_semantic_cache

            cache = get_semantic_cache()
            result = cache.lookup(question)

            if result:
                verified_badge = " ✅ Validada" if result.is_verified else ""
                print("\n" + "=" * 60)
                print(f"📦 RESPOSTA DO CACHE{verified_badge}")
                print(f"   Similaridade: {result.similarity_score:.2%}")
                print(f"   Chain original: {result.chain_type}")
                print(f"   Cached em: {result.created_at}")
                print("-" * 60)
                print("📝 PERGUNTA:")
                print(question)
                print("\n" + "-" * 60)
                print("💬 RESPOSTA:")
                print(result.answer)
                print("\n" + "=" * 60)
                return

        except Exception as e:
            logger.warning(f"Cache lookup failed (continuing without): {e}")

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
            build_rag_chain_with_rerank,
        )

        if args.chain_type == "base":
            chain = build_base_rag_chain(retriever, llm, k=args.k)
            # Base chain returns string, wrap for consistency
            result = chain.invoke(question)
            output = {"query": question, "answer": result, "contexts": []}

        elif args.chain_type == "rewriter":
            chain = build_rewriter_rag_chain(retriever, llm)
            result = chain.invoke(question)
            output = {"query": question, "answer": result, "contexts": []}

        elif args.chain_type == "multiquery":
            # Multi-query needs a model that handles list parsing
            multi_llm = get_llm(provider="groq")  # Better for structured output
            chain = build_multi_query_chain(retriever, llm, multi_query_llm=multi_llm)
            result = chain.invoke(question)
            output = {"query": question, "answer": result, "contexts": []}

        elif args.chain_type == "hyde":
            chain = build_hyde_chain(retriever, llm)
            result = chain.invoke(question)
            output = {"query": question, "answer": result, "contexts": []}

        elif args.chain_type == "rerank":
            chain = build_rag_chain_with_rerank(
                retriever,
                llm,
                initial_k=args.initial_k,
                rerank_top_k=args.rerank_k,
            )
            output = chain.invoke(question)

        else:  # full
            chain = build_rag_chain_with_context_full(retriever, llm)
            output = chain.invoke(question)

        answer = output.get("answer", "")

        # --- Store in Cache ---
        if cache and answer:
            try:
                cache.store(
                    question=args.question,  # Store original question, not sanitized
                    answer=answer,
                    chain_type=args.chain_type,
                )
            except Exception as e:
                logger.warning(f"Cache store failed: {e}")

        # Display results
        print("\n" + "=" * 60)
        print("📝 PERGUNTA:")
        print(output.get("query", question))
        print("\n" + "-" * 60)
        print("💬 RESPOSTA:")
        print(answer)

        if args.show_contexts and output.get("contexts"):
            print("\n" + "-" * 60)
            print("📚 CONTEXTOS RECUPERADOS:")
            for i, ctx in enumerate(output["contexts"], 1):
                print(f"\n[{i}] {ctx[:500]}{'...' if len(ctx) > 500 else ''}")

        print("\n" + "=" * 60)

    except Exception as e:
        logger.error(f"Chain error: {e}")
        print(f"\n⚠️ Sistema principal indisponível: {e}")

        # --- BM25 Fallback ---
        try:
            from src.app.rag.bm25_fallback import get_bm25_fallback

            fallback = get_bm25_fallback()
            fb_result = fallback.fallback_answer(question)

            if fb_result:
                print("\n" + "=" * 60)
                print("⚠️ FALLBACK (FAQ)")
                print("   Sistema principal indisponível — resposta aproximada do FAQ")
                print("-" * 60)
                print("📝 PERGUNTA ORIGINAL:")
                print(args.question)
                print(f"\n📋 FAQ MAIS PRÓXIMO:")
                print(f"   {fb_result.matched_question}")
                print("\n" + "-" * 60)
                print("💬 RESPOSTA:")
                print(fb_result.answer)
                print("\n" + "=" * 60)
            else:
                print("\n❌ Fallback FAQ também não encontrou resultados.")
                sys.exit(1)

        except Exception as fb_err:
            logger.error(f"BM25 fallback also failed: {fb_err}")
            print(f"\n❌ Erro: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
