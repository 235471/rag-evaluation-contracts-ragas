"""
Chunking configuration module.
Maps embedding models to appropriate tokenizers and chunk sizes based on their context windows.
"""

from dataclasses import dataclass
from typing import Optional, Literal

from src.app.logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""

    tokenizer_name: str  # HuggingFace model name or "tiktoken:encoding_name"
    chunk_size: int
    chunk_overlap: int
    max_tokens: int  # Model's max token window
    embedding_dims: int  # Embedding dimensions


# Registry of popular embedding models with their chunking configs
# Based on MTEB leaderboard data and model documentation
EMBEDDING_MODEL_REGISTRY = {
    # Gemini models
    "gemini-embedding-001": ChunkingConfig(
        tokenizer_name="tiktoken:cl100k_base",
        chunk_size=1500,
        chunk_overlap=150,
        max_tokens=2048,
        embedding_dims=3072,
    ),
    "text-embedding-004": ChunkingConfig(
        tokenizer_name="tiktoken:cl100k_base",
        chunk_size=1500,
        chunk_overlap=150,
        max_tokens=2048,
        embedding_dims=768,
    ),
    # OpenAI models
    "text-embedding-3-large": ChunkingConfig(
        tokenizer_name="tiktoken:cl100k_base",
        chunk_size=6000,
        chunk_overlap=600,
        max_tokens=8191,
        embedding_dims=3072,
    ),
    "text-embedding-3-small": ChunkingConfig(
        tokenizer_name="tiktoken:cl100k_base",
        chunk_size=6000,
        chunk_overlap=600,
        max_tokens=8191,
        embedding_dims=1536,
    ),
    # HuggingFace/Ollama models (commonly used)
    "BAAI/bge-small-en-v1.5": ChunkingConfig(
        tokenizer_name="BAAI/bge-small-en-v1.5",
        chunk_size=380,
        chunk_overlap=50,
        max_tokens=512,
        embedding_dims=384,
    ),
    "BAAI/bge-m3": ChunkingConfig(
        tokenizer_name="BAAI/bge-m3",
        chunk_size=6000,
        chunk_overlap=600,
        max_tokens=8194,
        embedding_dims=1024,
    ),
    "intfloat/multilingual-e5-large": ChunkingConfig(
        tokenizer_name="intfloat/multilingual-e5-large",
        chunk_size=380,
        chunk_overlap=50,
        max_tokens=514,
        embedding_dims=1024,
    ),
    "intfloat/multilingual-e5-small": ChunkingConfig(
        tokenizer_name="intfloat/multilingual-e5-small",
        chunk_size=380,
        chunk_overlap=50,
        max_tokens=512,
        embedding_dims=384,
    ),
    "jinaai/jina-embeddings-v3": ChunkingConfig(
        tokenizer_name="jinaai/jina-embeddings-v3",
        chunk_size=6000,
        chunk_overlap=600,
        max_tokens=8194,
        embedding_dims=1024,
    ),
    "Qwen/Qwen3-Embedding-8B": ChunkingConfig(
        tokenizer_name="Qwen/Qwen2-7B",  # Use Qwen2 tokenizer
        chunk_size=24000,
        chunk_overlap=2400,
        max_tokens=32768,
        embedding_dims=4096,
    ),
    "Qwen/Qwen3-Embedding-4B": ChunkingConfig(
        tokenizer_name="Qwen/Qwen2-7B",
        chunk_size=24000,
        chunk_overlap=2400,
        max_tokens=32768,
        embedding_dims=2560,
    ),
    "nvidia/llama-embed-nemotron-8b": ChunkingConfig(
        tokenizer_name="nvidia/llama-embed-nemotron-8b",
        chunk_size=24000,
        chunk_overlap=2400,
        max_tokens=32768,
        embedding_dims=4096,
    ),
}


# Provider defaults (when specific model not in registry)
PROVIDER_DEFAULTS = {
    "gemini": ChunkingConfig(
        tokenizer_name="tiktoken:cl100k_base",
        chunk_size=1500,
        chunk_overlap=150,
        max_tokens=2048,
        embedding_dims=3072,
    ),
    "openai": ChunkingConfig(
        tokenizer_name="tiktoken:cl100k_base",
        chunk_size=6000,
        chunk_overlap=600,
        max_tokens=8191,
        embedding_dims=1536,
    ),
    "ollama": ChunkingConfig(
        tokenizer_name="BAAI/bge-small-en-v1.5",
        chunk_size=380,
        chunk_overlap=50,
        max_tokens=512,
        embedding_dims=384,
    ),
}


def get_chunking_config(
    embedding_model: Optional[str] = None,
    embedding_provider: Optional[Literal["gemini", "openai", "ollama"]] = None,
) -> ChunkingConfig:
    """
    Get chunking configuration for a specific embedding model.

    Args:
        embedding_model: Full model name (e.g., "gemini-embedding-001", "BAAI/bge-m3")
        embedding_provider: Provider shorthand if model not specified

    Returns:
        ChunkingConfig with tokenizer and chunk sizes
    """
    # Try exact model match first
    if embedding_model and embedding_model in EMBEDDING_MODEL_REGISTRY:
        config = EMBEDDING_MODEL_REGISTRY[embedding_model]
        logger.info(
            f"Using chunking config for model '{embedding_model}': {config.chunk_size} tokens"
        )
        return config

    # Fall back to provider default
    if embedding_provider and embedding_provider in PROVIDER_DEFAULTS:
        config = PROVIDER_DEFAULTS[embedding_provider]
        logger.info(
            f"Using default chunking config for provider '{embedding_provider}': {config.chunk_size} tokens"
        )
        return config

    # Ultimate fallback (conservative for multilingual models)
    logger.warning(
        f"No chunking config found for model={embedding_model}, provider={embedding_provider}. Using fallback."
    )
    return ChunkingConfig(
        tokenizer_name="BAAI/bge-small-en-v1.5",
        chunk_size=380,
        chunk_overlap=50,
        max_tokens=512,
        embedding_dims=384,
    )


def create_text_splitter(config: ChunkingConfig):
    """
    Create a text splitter configured for the given chunking config.

    Args:
        config: ChunkingConfig instance

    Returns:
        RecursiveCharacterTextSplitter with appropriate length function
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Handle tiktoken encoders (for OpenAI/Gemini)
    if config.tokenizer_name.startswith("tiktoken:"):
        encoding_name = config.tokenizer_name.split(":", 1)[1]
        try:
            import tiktoken

            encoder = tiktoken.get_encoding(encoding_name)
            length_fn = lambda text: len(encoder.encode(text))
            logger.info(f"Using tiktoken encoder: {encoding_name}")
        except ImportError:
            logger.warning("tiktoken not installed, falling back to character length")
            length_fn = len
    else:
        # Use HuggingFace tokenizer
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
            length_fn = lambda text: len(
                tokenizer.encode(text, add_special_tokens=False)
            )
            logger.info(f"Using HuggingFace tokenizer: {config.tokenizer_name}")
        except Exception as e:
            logger.warning(
                f"Failed to load tokenizer '{config.tokenizer_name}': {e}. Using character length."
            )
            length_fn = len

    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=length_fn,
    )


def get_chunking_config_from_env():
    """
    Get chunking configuration based on environment settings.

    Respects EMBEDDING_MODEL or EMBEDDING_PROVIDER from .env,
    with optional manual override via CHUNK_SIZE/CHUNK_OVERLAP.

    Returns:
        ChunkingConfig instance
    """
    import os
    from src.app.config import get_settings

    settings = get_settings()

    # Check for manual override
    manual_chunk_size = os.getenv("CHUNK_SIZE")
    manual_chunk_overlap = os.getenv("CHUNK_OVERLAP")
    manual_tokenizer = os.getenv("CHUNKING_TOKENIZER")

    if manual_chunk_size and manual_tokenizer:
        logger.info(
            f"Using manual chunking override: tokenizer={manual_tokenizer}, size={manual_chunk_size}"
        )
        return ChunkingConfig(
            tokenizer_name=manual_tokenizer,
            chunk_size=int(manual_chunk_size),
            chunk_overlap=(
                int(manual_chunk_overlap)
                if manual_chunk_overlap
                else int(manual_chunk_size) // 10
            ),
            max_tokens=999999,  # Unknown
            embedding_dims=999999,  # Unknown
        )

    # Try to detect from embedding settings
    embedding_model = os.getenv("EMBEDDING_MODEL")  # Full model name if specified
    if embedding_model:
        return get_chunking_config(embedding_model=embedding_model)

    # Fall back to provider
    return get_chunking_config(embedding_provider=settings.EMBEDDING_PROVIDER)
