"""
PostgreSQL/PGVector vectorstore connector.
Provides connection and bootstrap functions with NO side effects on import.
"""

from typing import Optional

from src.app.config import get_settings
from src.app.logging_conf import get_logger

logger = get_logger(__name__)


class PGVectorTableNotFoundError(Exception):
    """Raised when the required PGVector table does not exist."""

    pass


def table_exists(table_name: str, schema: str = "public") -> bool:
    """
    Check if a table exists in the PostgreSQL database.

    Args:
        table_name: Name of the table to check
        schema: Schema name (default: public)

    Returns:
        True if table exists, False otherwise
    """
    from sqlalchemy import create_engine, inspect

    settings = get_settings()
    engine = create_engine(settings.POSTGRES_URL)

    try:
        inspector = inspect(engine)
        return table_name in inspector.get_table_names(schema=schema)
    finally:
        engine.dispose()


def get_pgvector_store(
    table_name: Optional[str] = None,
    embeddings=None,
    check_exists: bool = True,
):
    """
    Get a PGVectorStore instance connected to an existing table.

    This function ONLY connects to an existing table. It does NOT create
    tables or indexes. Use bootstrap_pgvector_table for infrastructure setup.

    Args:
        table_name: Table name (defaults to POSTGRES_TABLE_NAME from settings)
        embeddings: Embeddings instance (defaults to get_embeddings())
        check_exists: If True, verify table exists before connecting

    Returns:
        PGVectorStore instance

    Raises:
        PGVectorTableNotFoundError: If table doesn't exist and check_exists is True
    """
    from langchain_postgres import PGEngine, PGVectorStore

    settings = get_settings()
    table_name = table_name or settings.POSTGRES_TABLE_NAME

    if check_exists and not table_exists(table_name):
        raise PGVectorTableNotFoundError(
            f"Table '{table_name}' not found in database. "
            f"Run 'python scripts/bootstrap_postgres.py --table {table_name}' to create it."
        )

    if embeddings is None:
        from src.app.config import get_embeddings

        embeddings = get_embeddings()

    logger.info(f"Connecting to PGVector table: {table_name}")

    try:
        pg_engine = PGEngine.from_connection_string(url=settings.POSTGRES_URL)
        vector_store = PGVectorStore.create_sync(
            engine=pg_engine,
            table_name=table_name,
            embedding_service=embeddings,
        )
        logger.info(f"Connected to PGVector table: {table_name}")
        return vector_store

    except Exception as e:
        logger.error(f"Failed to connect to PGVector: {e}")
        raise


def bootstrap_pgvector_table(
    table_name: str,
    vector_size: int = 3072,
    schema: str = "public",
    create_unique_index: bool = True,
) -> None:
    """
    Create PGVector table and optional unique index for content_hash.

    This function is intended to be called ONLY via CLI script, not during
    normal application runtime.

    Args:
        table_name: Name of the table to create
        vector_size: Dimension of the embedding vectors
        schema: Schema name (default: public)
        create_unique_index: If True, create unique index on content_hash
    """
    import psycopg
    from langchain_postgres import PGEngine
    from sqlalchemy import create_engine, inspect

    settings = get_settings()

    # Check if table already exists
    engine = create_engine(settings.POSTGRES_URL)
    try:
        inspector = inspect(engine)
        if table_name in inspector.get_table_names(schema=schema):
            logger.info(f"Table '{table_name}' already exists, skipping creation")
        else:
            # Create table using PGEngine
            pg_engine = PGEngine.from_connection_string(url=settings.POSTGRES_URL)
            pg_engine.init_vectorstore_table(
                table_name=table_name,
                vector_size=vector_size,
                schema_name=schema,
            )
            logger.info(
                f"Created PGVector table: {schema}.{table_name} (vector_size={vector_size})"
            )
    finally:
        engine.dispose()

    # Create unique index on content_hash
    if create_unique_index:
        from sqlalchemy.engine import make_url

        url_obj = make_url(settings.POSTGRES_URL)

        # psycopg3 supports connection URIs directly.
        # We just need to ensure it's in a format it understands (strip +psycopg if present)
        pg_uri = settings.POSTGRES_URL.replace("postgresql+psycopg://", "postgresql://")

        try:
            # psycopg.connect supports URIs
            conn = psycopg.connect(pg_uri)
            with conn.cursor() as cur:
                index_name = f"idx_unique_content_hash_{table_name}"
                cur.execute(
                    f"""
                    CREATE UNIQUE INDEX IF NOT EXISTS {index_name}
                    ON {schema}.{table_name}
                    ((langchain_metadata->>'content_hash'));
                    """
                )
                conn.commit()
            logger.info(f"Created unique index on content_hash: {index_name}")
        except Exception as e:
            logger.error(f"Failed to create unique index: {e}")
            raise
        finally:
            if "conn" in locals() and conn:
                conn.close()


def bootstrap_cache_table(
    table_name: str = "semantic_cache",
    vector_size: int = 768,
    schema: str = "public",
) -> None:
    """
    Create the semantic cache table for storing question-answer pairs.

    This table has a different schema than the LangChain vectorstore tables,
    so we use raw SQL instead of PGEngine.

    Args:
        table_name: Name of the cache table to create
        vector_size: Dimension of the embedding vectors
        schema: Schema name (default: public)
    """
    import psycopg

    settings = get_settings()

    # Check if table already exists
    if table_exists(table_name, schema=schema):
        logger.info(f"Cache table '{table_name}' already exists, skipping creation")
        return

    pg_uri = settings.POSTGRES_URL.replace("postgresql+psycopg://", "postgresql://")

    try:
        conn = psycopg.connect(pg_uri)
        with conn.cursor() as cur:
            # Ensure pgvector extension exists
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create cache table
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {schema}.{table_name} (
                    id SERIAL PRIMARY KEY,
                    question_text TEXT NOT NULL,
                    question_embedding VECTOR({vector_size}) NOT NULL,
                    answer_text TEXT NOT NULL,
                    chain_type VARCHAR(50) NOT NULL DEFAULT 'full',
                    is_verified BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

            # Create HNSW index for fast cosine similarity search
            index_name = f"idx_hnsw_{table_name}"
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON {schema}.{table_name}
                USING hnsw (question_embedding vector_cosine_ops)
                """
            )

            conn.commit()

        logger.info(
            f"Created cache table: {schema}.{table_name} "
            f"(vector_size={vector_size}) with HNSW index"
        )

    except Exception as e:
        logger.error(f"Failed to create cache table: {e}")
        raise
    finally:
        if "conn" in locals() and conn:
            conn.close()
