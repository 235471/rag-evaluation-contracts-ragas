#!/usr/bin/env python3
"""
Streamlit Chat RAG Application - Main Entry Point.
Interactive UI for testing RAG chains with different strategies.

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
            options=["gemini", "groq", "ollama", "perplexity"],
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
        st.caption(f"Backend: {backend} | Chain: {chain_type}")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("contexts") and show_contexts:
                with st.expander("📚 Contextos Recuperados"):
                    for i, ctx in enumerate(message["contexts"], 1):
                        st.markdown(
                            f"**[{i}]** {ctx[:500]}{'...' if len(ctx) > 500 else ''}"
                        )

    # Chat input
    if prompt := st.chat_input("Faça sua pergunta..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
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
                        with st.expander("📚 Contextos Recuperados", expanded=False):
                            for i, ctx in enumerate(contexts, 1):
                                st.markdown(
                                    f"**[{i}]** {ctx[:500]}{'...' if len(ctx) > 500 else ''}"
                                )

                    # Add to history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "contexts": contexts,
                        }
                    )

                except Exception as e:
                    error_msg = f"❌ Erro: {str(e)}"
                    st.error(error_msg)
                    logger.error(f"Error in chat: {e}")

    # Footer
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Limpar Histórico"):
            st.session_state.messages = []
            st.rerun()

    render_footer()


if __name__ == "__main__":
    main()
