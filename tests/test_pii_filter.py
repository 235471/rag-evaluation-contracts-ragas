"""
Unit tests for PII Guardrail module.

Tests cover:
- Custom recognizers: BR_CPF, BR_CNPJ, API_KEY
- Presidio native recognizers: EMAIL, PHONE_NUMBER
- Sanitization with AnonymizerEngine
- Clean text passthrough
"""

import pytest


@pytest.fixture(scope="module")
def guardrail():
    """Create a PIIGuardrail instance for all tests."""
    from src.app.guardrails.pii_filter import PIIGuardrail

    return PIIGuardrail()


class TestCPFDetection:
    """Test CPF detection via custom PatternRecognizer."""

    def test_cpf_with_punctuation(self, guardrail):
        text = "Meu CPF é 123.456.789-00"
        assert guardrail.contains_pii(text)
        types = guardrail.get_detected_types(text)
        assert "BR_CPF" in types

    def test_cpf_without_punctuation(self, guardrail):
        text = "CPF: 12345678900"
        assert guardrail.contains_pii(text)
        types = guardrail.get_detected_types(text)
        assert "BR_CPF" in types

    def test_cpf_sanitized(self, guardrail):
        text = "Meu CPF é 123.456.789-00"
        sanitized = guardrail.sanitize(text)
        assert "123.456.789-00" not in sanitized
        assert "BR_CPF" in sanitized or "CPF" in sanitized.upper()


class TestCNPJDetection:
    """Test CNPJ detection via custom PatternRecognizer."""

    def test_cnpj_with_punctuation(self, guardrail):
        text = "CNPJ da empresa: 12.345.678/0001-90"
        assert guardrail.contains_pii(text)
        types = guardrail.get_detected_types(text)
        assert "BR_CNPJ" in types

    def test_cnpj_without_punctuation(self, guardrail):
        text = "CNPJ: 12345678000190"
        assert guardrail.contains_pii(text)
        types = guardrail.get_detected_types(text)
        assert "BR_CNPJ" in types

    def test_cnpj_sanitized(self, guardrail):
        text = "CNPJ da empresa: 12.345.678/0001-90"
        sanitized = guardrail.sanitize(text)
        assert "12.345.678/0001-90" not in sanitized


class TestAPIKeyDetection:
    """Test API key detection via custom PatternRecognizer."""

    def test_gemini_api_key(self, guardrail):
        text = "Minha chave é AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q"
        assert guardrail.contains_pii(text)
        types = guardrail.get_detected_types(text)
        assert "API_KEY" in types

    def test_groq_api_key(self, guardrail):
        text = "Use gsk_abc123def456ghi789jkl012mno345pqr678"
        assert guardrail.contains_pii(text)
        types = guardrail.get_detected_types(text)
        assert "API_KEY" in types

    def test_pinecone_api_key(self, guardrail):
        text = "Pinecone key: pcsk_abc123def456ghi789jkl012mno345"
        assert guardrail.contains_pii(text)
        types = guardrail.get_detected_types(text)
        assert "API_KEY" in types

    def test_openai_api_key(self, guardrail):
        text = "OpenAI key: sk-abc123def456ghi789jkl012mno345pqr678"
        assert guardrail.contains_pii(text)
        types = guardrail.get_detected_types(text)
        assert "API_KEY" in types

    def test_api_key_sanitized(self, guardrail):
        text = "Minha chave é AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q"
        sanitized = guardrail.sanitize(text)
        assert "AIzaSy" not in sanitized


class TestNativePresidioEntities:
    """Test Presidio's built-in recognizers."""

    def test_email_detection(self, guardrail):
        text = "Meu email é joao@empresa.com.br"
        assert guardrail.contains_pii(text)
        types = guardrail.get_detected_types(text)
        assert "EMAIL_ADDRESS" in types

    def test_email_sanitized(self, guardrail):
        text = "Contato: maria@gmail.com"
        sanitized = guardrail.sanitize(text)
        assert "maria@gmail.com" not in sanitized


class TestCleanText:
    """Test that clean text passes through without changes."""

    def test_clean_insurance_question(self, guardrail):
        text = "Qual é o limite de cobertura do seguro viagem?"
        assert not guardrail.contains_pii(text)

    def test_clean_text_sanitize_passthrough(self, guardrail):
        text = "Qual é o prazo máximo de cobertura em meses?"
        sanitized = guardrail.sanitize(text)
        assert sanitized == text

    def test_clean_text_no_matches(self, guardrail):
        text = "Preciso de informações sobre a apólice."
        matches = guardrail.scan(text)
        assert len(matches) == 0


class TestMultiplePII:
    """Test detection and sanitization of multiple PII types."""

    def test_multiple_pii_types(self, guardrail):
        text = "CPF: 123.456.789-00, email: user@test.com"
        types = guardrail.get_detected_types(text)
        assert "BR_CPF" in types
        assert "EMAIL_ADDRESS" in types

    def test_multiple_pii_sanitized(self, guardrail):
        text = "CPF: 123.456.789-00, email: user@test.com"
        sanitized = guardrail.sanitize(text)
        assert "123.456.789-00" not in sanitized
        assert "user@test.com" not in sanitized


class TestScanResults:
    """Test the scan method returns proper PIIMatch objects."""

    def test_scan_returns_pii_matches(self, guardrail):
        text = "CPF: 123.456.789-00"
        matches = guardrail.scan(text)
        assert len(matches) > 0

        match = matches[0]
        assert match.entity_type == "BR_CPF"
        assert match.score > 0
        assert match.start >= 0
        assert match.end > match.start
        assert match.text == "123.456.789-00"
