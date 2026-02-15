"""
PII (Personally Identifiable Information) Guardrail module.

Uses Microsoft Presidio Analyzer + Anonymizer as the primary detection engine,
with custom PatternRecognizers for BR-specific entities (CPF, CNPJ) and API Keys.

Requires:
    pip install presidio-analyzer presidio-anonymizer
    python -m spacy download pt_core_news_lg
"""

from dataclasses import dataclass
from typing import List, Optional

from src.app.logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class PIIMatch:
    """Represents a detected PII entity."""

    entity_type: str
    start: int
    end: int
    score: float
    text: str


def _build_cpf_recognizer():
    """Build a custom PatternRecognizer for Brazilian CPF."""
    from presidio_analyzer import Pattern, PatternRecognizer

    cpf_pattern = Pattern(
        name="cpf_pattern",
        regex=r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b",
        score=0.85,
    )
    return PatternRecognizer(
        supported_entity="BR_CPF",
        patterns=[cpf_pattern],
        context=["cpf", "cadastro", "pessoa física", "documento"],
        supported_language="pt",
    )


def _build_cnpj_recognizer():
    """Build a custom PatternRecognizer for Brazilian CNPJ."""
    from presidio_analyzer import Pattern, PatternRecognizer

    cnpj_pattern = Pattern(
        name="cnpj_pattern",
        regex=r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b",
        score=0.85,
    )
    return PatternRecognizer(
        supported_entity="BR_CNPJ",
        patterns=[cnpj_pattern],
        context=["cnpj", "cadastro", "pessoa jurídica", "empresa"],
        supported_language="pt",
    )


def _build_api_key_recognizer():
    """Build a custom PatternRecognizer for common API keys."""
    from presidio_analyzer import Pattern, PatternRecognizer

    api_key_patterns = [
        Pattern(
            name="gemini_key",
            regex=r"\bAIzaSy[A-Za-z0-9_-]{33}\b",
            score=0.95,
        ),
        Pattern(
            name="groq_key",
            regex=r"\bgsk_[A-Za-z0-9_-]{20,}\b",
            score=0.95,
        ),
        Pattern(
            name="openai_key",
            regex=r"\bsk-[A-Za-z0-9_-]{20,}\b",
            score=0.95,
        ),
        Pattern(
            name="pinecone_key",
            regex=r"\bpcsk_[A-Za-z0-9_-]{20,}\b",
            score=0.95,
        ),
    ]
    return PatternRecognizer(
        supported_entity="API_KEY",
        patterns=api_key_patterns,
        supported_language="pt",
    )


class PIIGuardrail:
    """
    PII detection and anonymization guardrail using Microsoft Presidio.

    Combines Presidio's built-in recognizers (EMAIL, PHONE, CREDIT_CARD, etc.)
    with custom recognizers for BR-specific entities (CPF, CNPJ) and API Keys.

    Usage:
        guardrail = PIIGuardrail()
        if guardrail.contains_pii("Meu CPF é 123.456.789-00"):
            clean_text = guardrail.sanitize("Meu CPF é 123.456.789-00")
    """

    def __init__(self, language: str = "pt"):
        """
        Initialize the PII guardrail.

        Args:
            language: Language code for NLP processing (default: "pt" for Portuguese).
        """
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        self.language = language

        # Configure spaCy NLP engine for Portuguese
        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "pt", "model_name": "pt_core_news_lg"}],
        }

        try:
            nlp_engine = NlpEngineProvider(nlp_configuration=nlp_config).create_engine()
            self.analyzer = AnalyzerEngine(
                nlp_engine=nlp_engine,
                supported_languages=["pt"],
            )
        except OSError:
            logger.warning(
                "spaCy model 'pt_core_news_lg' not found. "
                "Falling back to default engine. "
                "Run: python -m spacy download pt_core_news_lg"
            )
            self.analyzer = AnalyzerEngine()
            self.language = "en"

        # Register custom recognizers
        self.analyzer.registry.add_recognizer(_build_cpf_recognizer())
        self.analyzer.registry.add_recognizer(_build_cnpj_recognizer())
        self.analyzer.registry.add_recognizer(_build_api_key_recognizer())

        logger.info("PIIGuardrail initialized with Presidio + custom recognizers")

    def scan(self, text: str) -> List[PIIMatch]:
        """
        Scan text for PII entities.

        Args:
            text: Input text to scan.

        Returns:
            List of PIIMatch objects with detected entities.
        """
        results = self.analyzer.analyze(
            text=text,
            language=self.language,
        )

        matches = []
        for result in results:
            matches.append(
                PIIMatch(
                    entity_type=result.entity_type,
                    start=result.start,
                    end=result.end,
                    score=result.score,
                    text=text[result.start : result.end],
                )
            )

        if matches:
            entity_types = list(set(m.entity_type for m in matches))
            logger.warning(f"PII detected: {entity_types} ({len(matches)} matches)")

        return matches

    def sanitize(self, text: str) -> str:
        """
        Sanitize text by replacing PII with anonymized placeholders.

        Example: "Meu CPF é 123.456.789-00" -> "Meu CPF é <BR_CPF>"

        Args:
            text: Input text to sanitize.

        Returns:
            Sanitized text with PII replaced by entity type placeholders.
        """
        from presidio_anonymizer import AnonymizerEngine
        from presidio_anonymizer.entities import OperatorConfig

        results = self.analyzer.analyze(
            text=text,
            language=self.language,
        )

        if not results:
            return text

        anonymizer = AnonymizerEngine()
        anonymized = anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators={"DEFAULT": OperatorConfig("replace", {"new_value": None})},
        )

        entity_types = list(set(r.entity_type for r in results))
        logger.info(f"Sanitized PII: {entity_types}")

        return anonymized.text

    def contains_pii(self, text: str) -> bool:
        """
        Quick check if text contains any PII.

        Args:
            text: Input text to check.

        Returns:
            True if PII is detected, False otherwise.
        """
        results = self.analyzer.analyze(
            text=text,
            language=self.language,
            score_threshold=0.5,
        )
        return len(results) > 0

    def get_detected_types(self, text: str) -> List[str]:
        """
        Get list of PII entity types detected in text.

        Args:
            text: Input text to check.

        Returns:
            List of unique entity type strings (e.g., ["BR_CPF", "EMAIL"]).
        """
        results = self.analyzer.analyze(
            text=text,
            language=self.language,
        )
        return list(set(r.entity_type for r in results))


# Singleton instance for reuse
_guardrail: Optional[PIIGuardrail] = None


def get_pii_guardrail() -> PIIGuardrail:
    """Get PII guardrail singleton instance."""
    global _guardrail
    if _guardrail is None:
        _guardrail = PIIGuardrail()
    return _guardrail
