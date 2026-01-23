"""
Pinecone vectorstore connector.
Provides connection and bootstrap functions with NO side effects on import.
"""

import time
from typing import Optional

from src.app.config import get_settings
from src.app.logging_conf import get_logger

logger = get_logger(__name__)


class PineconeIndexNotFoundError(Exception):
    """Raised when the required Pinecone index does not exist."""

    pass


def index_exists(index_name: Optional[str] = None) -> bool:
    """
    Check if a Pinecone index exists.

    Args:
        index_name: Index name (defaults to PINECONE_INDEX from settings)

    Returns:
        True if index exists, False otherwise
    """
    from pinecone import Pinecone

    settings = get_settings()
    index_name = index_name or settings.PINECONE_INDEX

    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    return index_name in pc.list_indexes().names()


def get_pinecone_store(
    embeddings=None,
    index_name: Optional[str] = None,
    check_exists: bool = True,
):
    """
    Get a PineconeVectorStore instance connected to an existing index.

    This function ONLY connects to an existing index. It does NOT create
    indexes. Use bootstrap_pinecone_index for infrastructure setup.

    Args:
        embeddings: Embeddings instance (defaults to get_embeddings())
        index_name: Index name (defaults to PINECONE_INDEX from settings)
        check_exists: If True, verify index exists before connecting

    Returns:
        PineconeVectorStore instance

    Raises:
        PineconeIndexNotFoundError: If index doesn't exist and check_exists is True
    """
    from langchain_pinecone import PineconeVectorStore

    settings = get_settings()
    index_name = index_name or settings.PINECONE_INDEX

    if check_exists and not index_exists(index_name):
        raise PineconeIndexNotFoundError(
            f"Pinecone index '{index_name}' not found. "
            f"Run 'python scripts/bootstrap_pinecone.py' to create it."
        )

    if embeddings is None:
        from src.app.config import get_embeddings

        embeddings = get_embeddings()

    logger.info(f"Connecting to Pinecone index: {index_name}")

    try:
        vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
        )
        logger.info(f"Connected to Pinecone index: {index_name}")
        return vector_store

    except Exception as e:
        logger.error(f"Failed to connect to Pinecone: {e}")
        raise


def bootstrap_pinecone_index(
    index_name: Optional[str] = None,
    dimension: Optional[int] = None,
    metric: Optional[str] = None,
    cloud: Optional[str] = None,
    region: Optional[str] = None,
    wait_seconds: int = 15,
) -> None:
    """
    Create Pinecone index if it doesn't exist.

    This function is intended to be called ONLY via CLI script, not during
    normal application runtime.

    Args:
        index_name: Index name (defaults to PINECONE_INDEX from settings)
        dimension: Vector dimension (defaults to PINECONE_DIMENSION from settings)
        metric: Similarity metric (defaults to PINECONE_METRIC from settings)
        cloud: Cloud provider (defaults to PINECONE_CLOUD from settings)
        region: Region (defaults to PINECONE_REGION from settings)
        wait_seconds: Seconds to wait after creating index
    """
    from pinecone import Pinecone, ServerlessSpec

    settings = get_settings()

    index_name = index_name or settings.PINECONE_INDEX
    dimension = dimension or settings.PINECONE_DIMENSION
    metric = metric or settings.PINECONE_METRIC
    cloud = cloud or settings.PINECONE_CLOUD
    region = region or settings.PINECONE_REGION

    pc = Pinecone(api_key=settings.PINECONE_API_KEY)

    if index_name in pc.list_indexes().names():
        logger.info(f"Pinecone index '{index_name}' already exists, skipping creation")
        return

    logger.info(
        f"Creating Pinecone index: {index_name} (dimension={dimension}, metric={metric})"
    )

    try:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        logger.info(f"Index created, waiting {wait_seconds}s for initialization...")
        time.sleep(wait_seconds)
        logger.info(f"Pinecone index '{index_name}' is ready")

    except Exception as e:
        logger.error(f"Failed to create Pinecone index: {e}")
        raise
