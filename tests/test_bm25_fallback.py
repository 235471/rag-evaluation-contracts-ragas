"""
Unit tests for BM25 Fallback module.
"""

import pytest
from pathlib import Path


class TestFAQLoading:
    """Test FAQ data loading."""

    def test_faq_loads_from_json(self):
        from src.app.rag.bm25_fallback import BM25FallbackRetriever

        retriever = BM25FallbackRetriever()
        assert len(retriever.faq_data) == 13

    def test_faq_entries_have_required_fields(self):
        from src.app.rag.bm25_fallback import BM25FallbackRetriever

        retriever = BM25FallbackRetriever()
        for entry in retriever.faq_data:
            assert "question" in entry
            assert "answer" in entry


class TestFallbackAnswer:
    """Test BM25 fallback answers."""

    def test_returns_fallback_result(self):
        from src.app.rag.bm25_fallback import BM25FallbackRetriever, FallbackResult

        retriever = BM25FallbackRetriever()
        result = retriever.fallback_answer("Qual é o prazo da garantia estendida?")

        assert result is not None
        assert isinstance(result, FallbackResult)
        assert result.is_fallback is True
        assert result.source == "faq"

    def test_answer_is_relevant(self):
        from src.app.rag.bm25_fallback import BM25FallbackRetriever

        retriever = BM25FallbackRetriever()
        result = retriever.fallback_answer("Qual o valor máximo de indenização?")

        assert result is not None
        assert "USD" in result.answer or "200" in result.answer

    def test_matched_question_is_populated(self):
        from src.app.rag.bm25_fallback import BM25FallbackRetriever

        retriever = BM25FallbackRetriever()
        result = retriever.fallback_answer("proteção de preço")

        assert result is not None
        assert len(result.matched_question) > 0


class TestFallbackResultDataclass:
    """Test FallbackResult dataclass."""

    def test_defaults(self):
        from src.app.rag.bm25_fallback import FallbackResult

        result = FallbackResult(
            answer="Test answer",
            matched_question="Test question",
        )
        assert result.source == "faq"
        assert result.is_fallback is True


class TestSingleton:
    """Test singleton pattern."""

    def test_get_bm25_fallback_returns_same_instance(self):
        from src.app.rag.bm25_fallback import get_bm25_fallback

        a = get_bm25_fallback()
        b = get_bm25_fallback()
        assert a is b
