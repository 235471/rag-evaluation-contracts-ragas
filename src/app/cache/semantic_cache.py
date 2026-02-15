"""
Semantic Cache module using PostgreSQL + pgvector.

Stores question embeddings and their corresponding answers to avoid
repeated LLM calls for semantically similar questions. Uses cosine
similarity search to find cached answers.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.app.config import get_settings, get_cache_embeddings
from src.app.logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class CacheResult:
    """Represents a cache hit result."""

    answer: str
    similarity_score: float
    is_verified: bool
    chain_type: str
    created_at: datetime
    original_question: str


class SemanticCache:
    """
    Semantic cache backed by PostgreSQL + pgvector.

    Stores question embeddings and answers. On lookup, finds the most similar
    cached question using cosine similarity and returns the answer if above
    the configured threshold.

    Usage:
        cache = SemanticCache()
        result = cache.lookup("What is the coverage limit?")
        if result:
            print(f"Cache hit: {result.answer}")
        else:
            answer = chain.invoke(question)
            cache.store(question, answer, chain_type="full")
    """

    def __init__(
        self,
        table_name: Optional[str] = None,
        threshold: Optional[float] = None,
    ):
        """
        Initialize the semantic cache.

        Args:
            table_name: Cache table name (default: from settings).
            threshold: Cosine similarity threshold for cache hits (default: from settings).
        """
        settings = get_settings()
        self.table_name = table_name or settings.CACHE_TABLE_NAME
        self.threshold = threshold or settings.CACHE_SIMILARITY_THRESHOLD
        self.postgres_url = settings.POSTGRES_URL
        self._embeddings = None

        logger.info(
            f"SemanticCache initialized: table={self.table_name}, "
            f"threshold={self.threshold}"
        )

    @property
    def embeddings(self):
        """Lazy-load embeddings to avoid import-time side effects."""
        if self._embeddings is None:
            self._embeddings = get_cache_embeddings()
        return self._embeddings

    def _get_pg_uri(self) -> str:
        """Convert SQLAlchemy URL to psycopg-compatible URI."""
        return self.postgres_url.replace("postgresql+psycopg://", "postgresql://")

    def lookup(self, question: str) -> Optional[CacheResult]:
        """
        Look up a semantically similar question in the cache.

        Generates an embedding for the question, then queries the cache table
        for the most similar stored question using cosine distance.

        Args:
            question: The user's question.

        Returns:
            CacheResult if a sufficiently similar question is found, None otherwise.
        """
        import psycopg

        try:
            # Generate embedding for the question
            question_embedding = self.embeddings.embed_query(question)

            pg_uri = self._get_pg_uri()
            conn = psycopg.connect(pg_uri)

            try:
                with conn.cursor() as cur:
                    # Use cosine distance operator (<=>), convert to similarity
                    cur.execute(
                        f"""
                        SELECT
                            question_text,
                            answer_text,
                            chain_type,
                            is_verified,
                            created_at,
                            1 - (question_embedding <=> %s::vector) AS similarity
                        FROM {self.table_name}
                        ORDER BY question_embedding <=> %s::vector
                        LIMIT 1
                        """,
                        (str(question_embedding), str(question_embedding)),
                    )

                    row = cur.fetchone()

                    if row is None:
                        logger.debug("Cache miss: no entries in cache")
                        return None

                    (
                        cached_question,
                        answer_text,
                        chain_type,
                        is_verified,
                        created_at,
                        similarity,
                    ) = row

                    if similarity >= self.threshold:
                        logger.info(
                            f"Cache HIT (similarity={similarity:.4f}): "
                            f"'{question[:50]}...' -> '{cached_question[:50]}...'"
                        )
                        return CacheResult(
                            answer=answer_text,
                            similarity_score=similarity,
                            is_verified=is_verified,
                            chain_type=chain_type,
                            created_at=created_at,
                            original_question=cached_question,
                        )
                    else:
                        logger.debug(
                            f"Cache miss (similarity={similarity:.4f} < "
                            f"threshold={self.threshold})"
                        )
                        return None

            finally:
                conn.close()

        except Exception as e:
            logger.error(f"Cache lookup failed: {e}")
            return None

    def store(
        self,
        question: str,
        answer: str,
        chain_type: str = "full",
        is_verified: bool = False,
    ) -> bool:
        """
        Store a question-answer pair in the cache.

        Args:
            question: The user's question.
            answer: The generated answer.
            chain_type: Which RAG chain produced this answer.
            is_verified: Whether this answer has been verified by a specialist.

        Returns:
            True if stored successfully, False otherwise.
        """
        import psycopg

        try:
            question_embedding = self.embeddings.embed_query(question)

            pg_uri = self._get_pg_uri()
            conn = psycopg.connect(pg_uri)

            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        INSERT INTO {self.table_name}
                            (question_text, question_embedding, answer_text, chain_type, is_verified)
                        VALUES (%s, %s::vector, %s, %s, %s)
                        """,
                        (
                            question,
                            str(question_embedding),
                            answer,
                            chain_type,
                            is_verified,
                        ),
                    )
                    conn.commit()

                logger.info(f"Cache STORE: '{question[:60]}...' (chain={chain_type})")
                return True

            finally:
                conn.close()

        except Exception as e:
            logger.error(f"Cache store failed: {e}")
            return False

    def clear(self) -> bool:
        """
        Clear all entries from the cache table.

        Returns:
            True if cleared successfully, False otherwise.
        """
        import psycopg

        try:
            pg_uri = self._get_pg_uri()
            conn = psycopg.connect(pg_uri)

            try:
                with conn.cursor() as cur:
                    cur.execute(f"DELETE FROM {self.table_name}")
                    deleted = cur.rowcount
                    conn.commit()

                logger.info(f"Cache CLEARED: {deleted} entries removed")
                return True

            finally:
                conn.close()

        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return False

    def count(self) -> int:
        """
        Get the number of entries in the cache.

        Returns:
            Number of cached entries, or -1 on error.
        """
        import psycopg

        try:
            pg_uri = self._get_pg_uri()
            conn = psycopg.connect(pg_uri)

            try:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                    result = cur.fetchone()
                    return result[0] if result else 0

            finally:
                conn.close()

        except Exception as e:
            logger.error(f"Cache count failed: {e}")
            return -1


# Singleton instance
_cache: Optional[SemanticCache] = None


def get_semantic_cache() -> SemanticCache:
    """Get semantic cache singleton instance."""
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache
