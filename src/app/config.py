"""
Application configuration module.
Loads environment variables and provides factory functions for LLM/Embeddings.
"""

import os
from dataclasses import dataclass, field
from typing import Literal, Optional

from dotenv import load_dotenv

# Load .env file
load_dotenv()


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    # Database
    POSTGRES_URL: str = field(default_factory=lambda: os.getenv("POSTGRES_URL", ""))

    # Pinecone
    PINECONE_API_KEY: str = field(
        default_factory=lambda: os.getenv("PINECONE_API_KEY", "")
    )
    PINECONE_INDEX: str = field(default_factory=lambda: os.getenv("PINECONE_INDEX", ""))
    PINECONE_CLOUD: str = field(
        default_factory=lambda: os.getenv("PINECONE_CLOUD", "aws")
    )
    PINECONE_REGION: str = field(
        default_factory=lambda: os.getenv("PINECONE_REGION", "us-east-1")
    )
    PINECONE_METRIC: str = field(
        default_factory=lambda: os.getenv("PINECONE_METRIC", "cosine")
    )
    PINECONE_DIMENSION: int = field(
        default_factory=lambda: int(os.getenv("PINECONE_DIMENSION", "3072"))
    )

    # API Keys
    GOOGLE_API_KEY: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    GROQ_API_KEY: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    PPLX_API_KEY: str = field(default_factory=lambda: os.getenv("PPLX_API_KEY", ""))
    LANGSMITH_API_KEY: str = field(
        default_factory=lambda: os.getenv("LANGSMITH_API_KEY", "")
    )
    LANGSMITH_TRACING: bool = field(
        default_factory=lambda: os.getenv("LANGSMITH_TRACING", "false").lower()
        == "true"
    )

    # Backend selection
    VECTORSTORE_BACKEND: Literal["postgres", "pinecone"] = field(
        default_factory=lambda: os.getenv("VECTORSTORE_BACKEND", "postgres")  # type: ignore
    )
    LLM_PROVIDER: Literal["gemini", "groq", "ollama", "perplexity"] = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini")  # type: ignore
    )
    EMBEDDING_PROVIDER: Literal["gemini", "ollama"] = field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "gemini")  # type: ignore
    )

    # Table names (defaults from notebooks)
    POSTGRES_TABLE_NAME: str = field(
        default_factory=lambda: os.getenv(
            "POSTGRES_TABLE_NAME", "documents_embeddings_gemini"
        )
    )

    # Semantic Cache
    CACHE_TABLE_NAME: str = field(
        default_factory=lambda: os.getenv("CACHE_TABLE_NAME", "semantic_cache")
    )
    CACHE_SIMILARITY_THRESHOLD: float = field(
        default_factory=lambda: float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.92"))
    )
    CACHE_EMBEDDING_DIMENSION: int = field(
        default_factory=lambda: int(os.getenv("CACHE_EMBEDDING_DIMENSION", "768"))
    )

    def validate(self) -> list[str]:
        """Validate required settings are present. Returns list of missing keys."""
        missing = []
        if not self.POSTGRES_URL and self.VECTORSTORE_BACKEND == "postgres":
            missing.append("POSTGRES_URL")
        if not self.PINECONE_API_KEY and self.VECTORSTORE_BACKEND == "pinecone":
            missing.append("PINECONE_API_KEY")
        if not self.GOOGLE_API_KEY and self.EMBEDDING_PROVIDER == "gemini":
            missing.append("GOOGLE_API_KEY")
        return missing


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings (singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def get_embeddings(provider: Optional[str] = None):
    """
    Factory function to get embeddings based on provider.

    Args:
        provider: Override for EMBEDDING_PROVIDER setting

    Returns:
        Embeddings instance (GoogleGenerativeAIEmbeddings or OllamaEmbeddings)
    """
    settings = get_settings()
    provider = provider or settings.EMBEDDING_PROVIDER

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=settings.GOOGLE_API_KEY,
        )
    elif provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model="qllama/bge-small-en-v1.5")
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


def get_cache_embeddings():
    """
    Factory function to get embeddings for semantic cache.

    Uses reduced dimensionality (default: 768) via Matryoshka truncation,
    since cache only compares questions to questions, not to document chunks.
    This avoids the HNSW 2000-dimension limit in pgvector.

    Returns:
        Embeddings instance with reduced output_dimensionality
    """
    settings = get_settings()
    dim = settings.CACHE_EMBEDDING_DIMENSION

    if settings.EMBEDDING_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=settings.GOOGLE_API_KEY,
            task_type="semantic_similarity",
            output_dimensionality=dim,
        )
    else:
        # For non-Gemini providers, use the standard embeddings
        return get_embeddings()


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.3,
):
    """
    Factory function to get LLM based on provider.

    Args:
        provider: Override for LLM_PROVIDER setting
        model: Specific model name (uses defaults if not provided)
        temperature: LLM temperature

    Returns:
        LLM instance
    """
    settings = get_settings()
    provider = provider or settings.LLM_PROVIDER

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model or "gemini-2.5-flash-lite",
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=temperature,
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model or "llama-3.3-70b-versatile",
            api_key=settings.GROQ_API_KEY,
            temperature=temperature,
        )
    elif provider == "ollama":
        from langchain_ollama.llms import OllamaLLM

        return OllamaLLM(model=model or "gemma3:4b")
    elif provider == "perplexity":
        from langchain_community.chat_models import ChatPerplexity

        return ChatPerplexity(
            model=model or "sonar",
            pplx_api_key=settings.PPLX_API_KEY,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def get_eval_llm():
    """Get LLM specifically configured for RAGAS evaluation."""
    settings = get_settings()
    from langchain_perplexity import ChatPerplexity

    return ChatPerplexity(
        model="sonar-pro",
        pplx_api_key=settings.PPLX_API_KEY,
        temperature=0.1,
    )


def get_eval_embeddings():
    """Get embeddings specifically configured for RAGAS evaluation."""
    settings = get_settings()
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=settings.GOOGLE_API_KEY,
        transport="rest",
    )
