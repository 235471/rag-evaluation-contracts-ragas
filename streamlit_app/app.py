#!/usr/bin/env python3
"""
Streamlit Chat RAG Application - Main Entry Point.
Interactive UI for testing RAG chains with different strategies.

Features:
    - Semantic cache: avoids repeated LLM calls for similar questions
    - PII guardrails: sanitizes sensitive data before processing
    - Multiple RAG chain strategies (base, rewriter, multiquery, hyde, rerank)

Run with: streamlit run streamlit_app/app.py
"""

import sys
from pathlib import Path

# Fix for psycopg3 on Windows
if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.app.logging_conf import setup_logging, get_logger
from src.app.config import get_settings, get_llm, get_embeddings
from streamlit_app.shared.ui import inject_custom_css, render_footer

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)

# Page config
st.set_page_config(
    page_title="RAG Chat - LangChain Advanced",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom styles
inject_custom_css()

# --- Lazy singletons (initialized once per session) ---

HAS_PRESIDIO = True
try:
    from src.app.guardrails.pii_filter import get_pii_guardrail
except ImportError:
    HAS_PRESIDIO = False

HAS_CACHE = True
try:
    from src.app.cache.semantic_cache import get_semantic_cache
except ImportError:
    HAS_CACHE = False

HAS_GUARD = True
try:
    from src.app.guardrails.prompt_guard import get_prompt_guard
except ImportError:
    HAS_GUARD = False

HAS_FALLBACK = True
try:
    from src.app.rag.bm25_fallback import get_bm25_fallback
except ImportError:
    HAS_FALLBACK = False


@st.cache_resource
def get_vectorstore(backend: str = "postgres"):
    """Get cached vectorstore connection."""
    embeddings = get_embeddings()

    if backend == "postgres":
        from src.app.vectorstores.postgres_pgvector import get_pgvector_store

        return get_pgvector_store(embeddings=embeddings)
    else:
        from src.app.vectorstores.pinecone_store import get_pinecone_store

        return get_pinecone_store(embeddings=embeddings)


def build_chain(
    chain_type: str, retriever, llm, initial_k: int = 20, rerank_k: int = 3
):
    """Build the selected RAG chain."""
    from src.app.rag.chains import (
        build_base_rag_chain,
        build_rewriter_rag_chain,
        build_multi_query_chain,
        build_hyde_chain,
        build_rag_chain_with_context_full,
        build_rag_chain_with_rerank,
    )

    if chain_type == "base":
        return build_base_rag_chain(retriever, llm)
    elif chain_type == "rewriter":
        return build_rewriter_rag_chain(retriever, llm)
    elif chain_type == "multiquery":
        multi_llm = get_llm(provider="groq")
        return build_multi_query_chain(retriever, llm, multi_query_llm=multi_llm)
    elif chain_type == "hyde":
        return build_hyde_chain(retriever, llm)
    elif chain_type == "rerank":
        return build_rag_chain_with_rerank(
            retriever,
            llm,
            initial_k=initial_k,
            rerank_top_k=rerank_k,
        )
    else:  # full
        return build_rag_chain_with_context_full(retriever, llm)


def main():
    st.title("🔍 RAG Chat")
    st.markdown("Interface interativa para testar diferentes estratégias de RAG.")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configurações")

        settings = get_settings()

        # Backend selection
        backend = st.selectbox(
            "Vectorstore Backend",
            options=["postgres", "pinecone"],
            index=0 if settings.VECTORSTORE_BACKEND == "postgres" else 1,
        )

        # Chain type selection
        chain_type = st.selectbox(
            "Tipo de Chain",
            options=["full", "rerank", "base", "rewriter", "multiquery", "hyde"],
            index=0,
            help="full: contexto completo | rerank: LLM judge reranking | base: simples",
        )

        # LLM provider
        llm_provider = st.selectbox(
            "LLM Provider",
            options=["gemini", "groq"],
            index=0,
        )

        # Retrieval settings
        st.subheader("🎯 Parâmetros de Retrieval")

        if chain_type == "rerank":
            initial_k = st.slider("Initial K (candidatos)", 5, 50, 20)
            rerank_k = st.slider("Rerank K (final)", 1, 10, 3)
            k = initial_k
        else:
            k = st.slider("K (documentos)", 1, 20, 5)
            initial_k = 20
            rerank_k = 3

        # Show contexts toggle
        show_contexts = st.checkbox("Mostrar Contextos", value=True)

        st.divider()
        st.subheader("🛡️ Features")
        use_cache = st.checkbox("📦 Usar Cache Semântico", value=HAS_CACHE)
        use_pii = st.checkbox("🔒 Guardrail PII", value=HAS_PRESIDIO)
        use_guard = st.checkbox("🛡️ Proteção Prompt Injection", value=HAS_GUARD)

        st.divider()
        st.caption(f"Backend: {backend} | Chain: {chain_type}")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Show cache badge if applicable
            if message.get("from_cache"):
                st.caption(
                    f"📦 Cache (similaridade: {message['cache_similarity']:.0%})"
                    + (" ✅ Validada" if message.get("is_verified") else "")
                )
            if message.get("contexts") and show_contexts:
                with st.expander("📚 Contextos Recuperados"):
                    for i, ctx in enumerate(message["contexts"], 1):
                        st.markdown(
                            f"**[{i}]** {ctx[:500]}{'...' if len(ctx) > 500 else ''}"
                        )

    # Chat input
    if prompt := st.chat_input("Faça sua pergunta..."):
        original_prompt = prompt

        # --- Prompt Injection Guard ---
        injection_blocked = False
        if use_guard and HAS_GUARD:
            try:
                guard = get_prompt_guard()
                guard_result = guard.classify(prompt)

                if not guard_result.is_safe:
                    reason = guard_result.blocked_reason
                    st.error(
                        f"🚫 Solicitação bloqueada (Camada {reason.layer})\n\n"
                        f"**Motivo:** {reason.reason}\n\n"
                        f"**Detalhe:** {reason.detail}"
                    )
                    logger.warning(
                        f"Prompt blocked: layer={reason.layer}, "
                        f"reason={reason.reason}, detail={reason.detail}"
                    )
                    injection_blocked = True
            except Exception as e:
                logger.warning(f"Prompt guard check failed: {e}")

        if injection_blocked:
            st.stop()

        # --- PII Guardrail ---
        pii_warning = None
        if use_pii and HAS_PRESIDIO:
            try:
                guardrail = get_pii_guardrail()
                detected_types = guardrail.get_detected_types(prompt)

                if detected_types:
                    pii_warning = f"⚠️ Dados sensíveis detectados e removidos: {', '.join(detected_types)}"
                    prompt = guardrail.sanitize(prompt)
                    logger.warning(f"PII sanitized: {detected_types}")
            except Exception as e:
                logger.warning(f"PII check failed: {e}")

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if pii_warning:
            st.warning(pii_warning)

        # Generate response
        with st.chat_message("assistant"):
            # --- Semantic Cache Lookup ---
            cache_hit = None
            if use_cache and HAS_CACHE:
                try:
                    cache = get_semantic_cache()
                    cache_hit = cache.lookup(prompt)
                except Exception as e:
                    logger.warning(f"Cache lookup failed: {e}")

            if cache_hit:
                # Display cached answer
                answer = cache_hit.answer
                st.markdown(answer)

                verified_text = " ✅ Validada" if cache_hit.is_verified else ""
                st.info(
                    f"📦 Resposta do cache "
                    f"(similaridade: {cache_hit.similarity_score:.0%})"
                    f"{verified_text}"
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "contexts": [],
                        "from_cache": True,
                        "cache_similarity": cache_hit.similarity_score,
                        "is_verified": cache_hit.is_verified,
                    }
                )

            else:
                # Normal chain execution
                with st.spinner("🔄 Processando..."):
                    try:
                        # Get vectorstore and build chain
                        vector_store = get_vectorstore(backend)
                        retriever = vector_store.as_retriever(search_kwargs={"k": k})
                        llm = get_llm(provider=llm_provider)

                        chain = build_chain(
                            chain_type,
                            retriever,
                            llm,
                            initial_k=initial_k,
                            rerank_k=rerank_k,
                        )

                        # Invoke chain
                        result = chain.invoke(prompt)

                        # Handle different output formats
                        if isinstance(result, dict):
                            answer = result.get("answer", str(result))
                            contexts = result.get("contexts", [])
                        else:
                            answer = str(result)
                            contexts = []

                        # Display answer
                        st.markdown(answer)

                        # Display contexts if available
                        if contexts and show_contexts:
                            with st.expander(
                                "📚 Contextos Recuperados", expanded=False
                            ):
                                for i, ctx in enumerate(contexts, 1):
                                    st.markdown(
                                        f"**[{i}]** {ctx[:500]}{'...' if len(ctx) > 500 else ''}"
                                    )

                        # Store in cache
                        if use_cache and HAS_CACHE and answer:
                            try:
                                cache = get_semantic_cache()
                                cache.store(
                                    question=original_prompt,
                                    answer=answer,
                                    chain_type=chain_type,
                                )
                            except Exception as e:
                                logger.warning(f"Cache store failed: {e}")

                        # Add to history
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "contexts": contexts,
                            }
                        )

                    except Exception as e:
                        logger.error(f"Chain error: {e}")

                        # --- BM25 Fallback ---
                        fallback_used = False
                        if HAS_FALLBACK:
                            try:
                                fallback = get_bm25_fallback()
                                fb_result = fallback.fallback_answer(prompt)

                                if fb_result:
                                    answer = fb_result.answer
                                    st.markdown(answer)
                                    st.warning(
                                        f"⚠️ **Fallback FAQ** — Sistema principal indisponível.\n\n"
                                        f"Resposta aproximada baseada no FAQ.\n\n"
                                        f"_FAQ: {fb_result.matched_question}_"
                                    )
                                    st.session_state.messages.append(
                                        {
                                            "role": "assistant",
                                            "content": answer,
                                            "contexts": [],
                                            "is_fallback": True,
                                        }
                                    )
                                    fallback_used = True
                            except Exception as fb_err:
                                logger.error(f"BM25 fallback also failed: {fb_err}")

                        if not fallback_used:
                            error_msg = f"❌ Erro: {str(e)}"
                            st.error(error_msg)

    # Footer
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Limpar Histórico"):
            st.session_state.messages = []
            st.rerun()

    render_footer()


if __name__ == "__main__":
    main()
