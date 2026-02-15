"""
BM25 Fallback module for RAG Insurance.

Provides a lightweight fallback mechanism using BM25 retrieval over
a curated FAQ dataset (documents/faq.json). Triggered when the main
RAG chain fails (timeout, rate limit, any exception).

Dependencies:
    - langchain-community (BM25Retriever)
    - rank_bm25
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.app.logging_conf import get_logger

logger = get_logger(__name__)

# Path to FAQ file relative to project root
FAQ_PATH = Path(__file__).parent.parent.parent.parent / "documents" / "faq.json"


@dataclass
class FallbackResult:
    """Result from BM25 FAQ fallback."""

    answer: str
    matched_question: str
    source: str = "faq"
    is_fallback: bool = True


class BM25FallbackRetriever:
    """
    BM25-based fallback retriever over curated FAQ data.

    Loads question-answer pairs from faq.json and creates a BM25 retriever
    that matches user questions to FAQ questions by keyword overlap.
    """

    def __init__(self, faq_path: Optional[str] = None, k: int = 1):
        """
        Initialize the BM25 fallback retriever.

        Args:
            faq_path: Path to FAQ JSON file (default: documents/faq.json)
            k: Number of top results to consider
        """
        self.faq_path = Path(faq_path) if faq_path else FAQ_PATH
        self.k = k
        self._retriever = None
        self._faq_data = None

    @property
    def faq_data(self) -> list[dict]:
        """Lazy-load FAQ data from JSON."""
        if self._faq_data is None:
            self._faq_data = self._load_faq()
        return self._faq_data

    @property
    def retriever(self):
        """Lazy-load BM25 retriever."""
        if self._retriever is None:
            self._retriever = self._build_retriever()
        return self._retriever

    def _load_faq(self) -> list[dict]:
        """Load FAQ entries from JSON file."""
        if not self.faq_path.exists():
            logger.error(f"FAQ file not found: {self.faq_path}")
            return []

        with open(self.faq_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Loaded {len(data)} FAQ entries from {self.faq_path.name}")
        return data

    def _build_retriever(self):
        """Build BM25Retriever from FAQ documents."""
        from langchain_community.retrievers import BM25Retriever
        from langchain_core.documents import Document

        # Create documents: page_content is the question (for matching),
        # answer is stored in metadata
        documents = [
            Document(
                page_content=entry["question"],
                metadata={
                    "answer": entry["answer"],
                    "question": entry["question"],
                },
            )
            for entry in self.faq_data
        ]

        if not documents:
            logger.warning("No FAQ documents loaded, fallback will be unavailable")
            return None

        retriever = BM25Retriever.from_documents(documents, k=self.k)
        logger.info(f"BM25 fallback retriever ready with {len(documents)} FAQ entries")
        return retriever

    def fallback_answer(self, question: str) -> Optional[FallbackResult]:
        """
        Find the best FAQ match for a question using BM25.

        Args:
            question: User question to match against FAQ

        Returns:
            FallbackResult with the best matching FAQ answer,
            or None if no FAQ data is available
        """
        if self.retriever is None:
            logger.warning("BM25 retriever not available")
            return None

        try:
            results = self.retriever.invoke(question)

            if not results:
                logger.info("No BM25 results found for fallback")
                return None

            best = results[0]
            result = FallbackResult(
                answer=best.metadata["answer"],
                matched_question=best.metadata["question"],
            )

            logger.info(f"BM25 fallback match: '{result.matched_question[:60]}...'")
            return result

        except Exception as e:
            logger.error(f"BM25 fallback failed: {e}")
            return None


# ── Singleton ──────────────────────────────────────────────
_instance: Optional[BM25FallbackRetriever] = None


def get_bm25_fallback() -> BM25FallbackRetriever:
    """Get or create singleton BM25FallbackRetriever instance."""
    global _instance
    if _instance is None:
        _instance = BM25FallbackRetriever()
    return _instance
