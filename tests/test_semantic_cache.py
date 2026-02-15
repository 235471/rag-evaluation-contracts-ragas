"""
Unit tests for Semantic Cache module.

Tests cover:
- CacheResult dataclass
- SemanticCache initialization with custom settings
"""

import pytest
from datetime import datetime


class TestCacheResult:
    """Test the CacheResult dataclass."""

    def test_cache_result_creation(self):
        from src.app.cache.semantic_cache import CacheResult

        result = CacheResult(
            answer="Os portadores de cartão Mastercard Gold.",
            similarity_score=0.95,
            is_verified=True,
            chain_type="full",
            created_at=datetime(2026, 1, 15, 10, 30),
            original_question="Quem está coberto?",
        )

        assert result.answer == "Os portadores de cartão Mastercard Gold."
        assert result.similarity_score == 0.95
        assert result.is_verified is True
        assert result.chain_type == "full"
        assert result.original_question == "Quem está coberto?"

    def test_cache_result_not_verified(self):
        from src.app.cache.semantic_cache import CacheResult

        result = CacheResult(
            answer="Resposta teste",
            similarity_score=0.93,
            is_verified=False,
            chain_type="rerank",
            created_at=datetime.now(),
            original_question="Pergunta teste",
        )

        assert result.is_verified is False
        assert result.chain_type == "rerank"


class TestSemanticCacheInit:
    """Test SemanticCache initialization."""

    def test_default_settings(self):
        """Cache should use Settings defaults when no args provided."""
        from src.app.cache.semantic_cache import SemanticCache
        from src.app.config import get_settings

        settings = get_settings()
        cache = SemanticCache()

        assert cache.table_name == settings.CACHE_TABLE_NAME
        assert cache.threshold == settings.CACHE_SIMILARITY_THRESHOLD

    def test_custom_settings(self):
        """Cache should accept custom table_name and threshold."""
        from src.app.cache.semantic_cache import SemanticCache

        cache = SemanticCache(
            table_name="my_custom_cache",
            threshold=0.85,
        )

        assert cache.table_name == "my_custom_cache"
        assert cache.threshold == 0.85
