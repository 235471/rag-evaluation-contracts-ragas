"""
Prompt Injection Guard for RAG Insurance.

Multi-layered defense against prompt injection attacks:
    - Layer 1: Keyword blocklist (instant block for sensitive terms)
    - Layer 2: Pattern matching (regex for known injection patterns)
    - Layer 3: LLM classifier via Groq (Llama Prompt Guard 2 86M)

Graceful degradation: if Groq API is unavailable, layers 1 & 2 remain active.
"""

import re
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.app.logging_conf import get_logger

logger = get_logger(__name__)


class Intent(Enum):
    """Classification result for prompt injection detection."""

    SAFE = "safe"
    BLOCKED = "blocked"


@dataclass
class BlockedReason:
    """Details about why a prompt was blocked."""

    reason: str
    layer: int  # 1 = keyword, 2 = pattern, 3 = LLM
    detail: str = ""


@dataclass
class GuardResult:
    """Result from prompt injection classification."""

    intent: Intent
    blocked_reason: Optional[BlockedReason] = None

    @property
    def is_safe(self) -> bool:
        return self.intent == Intent.SAFE


# ── Layer 1: Keyword Blocklist ──────────────────────────────
KEYWORD_BLOCKLIST = [
    # API keys and credentials (EN)
    "api key",
    "api_key",
    "apikey",
    "secret key",
    "secret_key",
    "access token",
    "access_token",
    "password",
    "bearer",
    "authorization",
    # API keys and credentials (PT)
    "chave de api",
    "chave da api",
    "chave api",
    "token de acesso",
    "senha",
    "credencial",
    "credenciais",
    # Known API key prefixes
    "sk-",
    "aizasy",
    "gsk_",
    "pcsk_",
    # DB credentials
    "connection string",
    "postgres://",
    "postgresql://",
    "mysql://",
]


# ── Layer 2: Pattern Matching ───────────────────────────────
INJECTION_PATTERNS = [
    # English injection patterns
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?)",
    r"forget\s+(all\s+)?(previous|above|prior|your)\s+(instructions?|prompts?|rules?)",
    r"disregard\s+(all\s+)?(previous|above|prior|your)\s+(instructions?|prompts?|rules?)",
    r"you\s+are\s+now\s+",
    r"act\s+as\s+(if\s+you\s+were|a|an)\s+",
    r"pretend\s+(to\s+be|you\s+are)",
    r"from\s+now\s+on\s+you\s+(are|will|must|should)",
    r"new\s+instructions?\s*:",
    r"override\s+(previous\s+)?(instructions?|rules?|prompt)",
    r"(reveal|show|display|print|output)\s+(your|the)\s+(system\s+)?(prompt|instructions?|rules?)",
    r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions?|rules?)",
    # Portuguese injection patterns
    r"ignore\s+(todas?\s+)?(as\s+)?(instruções?|regras?|prompt)\s*(anteriores?)?",
    r"esqueça\s+(todas?\s+)?(as\s+)?(instruções?|regras?|prompt)",
    r"desconsidere\s+(todas?\s+)?(as\s+)?(instruções?|regras?|prompt)",
    r"agora\s+você\s+é\s+",
    r"finja\s+(que\s+)?(você\s+)?(é|ser)",
    r"a\s+partir\s+de\s+agora\s+",
    r"novas?\s+instruções?\s*:",
    r"mostre\s+(seu|o)\s+(system\s+)?(prompt|instruções?)",
    r"quais?\s+(são?\s+)?(suas?\s+)?(instruções?|regras?|prompt)",
    # Code/markup injection
    r"```\s*system",
    r"<\s*script",
    r"<\|",
    r"\{\{.*\}\}",
    r"<\s*system\s*>",
    # Credential exfiltration attempts (PT)
    r"(me\s+)?(diga|fale|passe|mande|envie|mostre)\s+(sua|seu|a|o)\s+(chave|key|senha|password|token|credencial)",
    r"qual\s+[eé]\s+(sua|seu|a|o)\s+(chave|key|senha|password|token|credencial)",
    # Credential exfiltration attempts (EN)
    r"(tell|give|show|send)\s+me\s+(your|the)\s+(key|password|token|credential|secret)",
    r"what\s+is\s+your\s+(api\s+)?(key|password|token|credential|secret)",
]

# Pre-compile patterns for performance
_COMPILED_PATTERNS = [
    re.compile(p, re.IGNORECASE | re.UNICODE) for p in INJECTION_PATTERNS
]


class PromptInjectionGuard:
    """
    Multi-layered prompt injection detection.

    Layer 1: Keyword blocklist — instant block for sensitive terms
    Layer 2: Pattern matching — regex for known injection patterns
    Layer 3: LLM classifier — Llama Prompt Guard 2 via Groq API
    """

    def __init__(self, use_llm: bool = True):
        """
        Initialize the guard.

        Args:
            use_llm: Whether to enable Layer 3 (LLM classification)
        """
        self.use_llm = use_llm
        self._groq_client = None
        self._groq_available = None

    def classify(self, text: str) -> GuardResult:
        """
        Classify a prompt through all defense layers.

        Args:
            text: User input text to classify

        Returns:
            GuardResult with intent (SAFE or BLOCKED) and reason if blocked
        """
        text_lower = text.lower().strip()

        # Layer 1: Keyword blocklist
        result = self._check_keywords(text_lower)
        if result:
            logger.warning(f"🚫 BLOCKED (Layer 1 - Keyword): {result.detail}")
            return GuardResult(intent=Intent.BLOCKED, blocked_reason=result)

        # Layer 2: Pattern matching
        result = self._check_patterns(text)
        if result:
            logger.warning(f"🚫 BLOCKED (Layer 2 - Pattern): {result.detail}")
            return GuardResult(intent=Intent.BLOCKED, blocked_reason=result)

        # Layer 3: LLM classification (Groq Prompt Guard)
        if self.use_llm:
            result = self._check_llm(text)
            if result:
                logger.warning(f"🚫 BLOCKED (Layer 3 - LLM): {result.detail}")
                return GuardResult(intent=Intent.BLOCKED, blocked_reason=result)

        logger.debug(f"✅ Prompt classified as SAFE: '{text[:60]}...'")
        return GuardResult(intent=Intent.SAFE)

    def _check_keywords(self, text_lower: str) -> Optional[BlockedReason]:
        """Layer 1: Check for blocked keywords."""
        for keyword in KEYWORD_BLOCKLIST:
            if keyword in text_lower:
                return BlockedReason(
                    reason="Termo sensível detectado",
                    layer=1,
                    detail=f"Keyword: '{keyword}'",
                )
        return None

    def _check_patterns(self, text: str) -> Optional[BlockedReason]:
        """Layer 2: Check for injection patterns via regex."""
        for pattern in _COMPILED_PATTERNS:
            match = pattern.search(text)
            if match:
                return BlockedReason(
                    reason="Padrão de prompt injection detectado",
                    layer=2,
                    detail=f"Pattern match: '{match.group()}'",
                )
        return None

    def _check_llm(self, text: str) -> Optional[BlockedReason]:
        """
        Layer 3: Classify using Llama Prompt Guard 2 via Groq API.

        Returns None if the prompt is safe or if Groq is unavailable.
        """
        if not self._is_groq_available():
            return None

        try:
            completion = self._groq_client.chat.completions.create(
                model="meta-llama/llama-prompt-guard-2-86m",
                messages=[{"role": "user", "content": text}],
            )

            label = completion.choices[0].message.content.strip().lower()
            logger.debug(f"Llama Prompt Guard result: {label}")

            if "malicious" in label:
                return BlockedReason(
                    reason="Classificado como malicioso pelo LLM",
                    layer=3,
                    detail=f"Llama Prompt Guard: {label}",
                )

            return None

        except Exception as e:
            logger.debug(f"Groq LLM check failed (continuing): {e}")
            return None

    def _is_groq_available(self) -> bool:
        """Check if Groq API is available (cached)."""
        if self._groq_available is not None:
            return self._groq_available

        groq_key = os.getenv("GROQ_API_KEY", "")
        if not groq_key:
            logger.debug("GROQ_API_KEY not set, Layer 3 (LLM) disabled")
            self._groq_available = False
            return False

        try:
            from groq import Groq

            self._groq_client = Groq(api_key=groq_key)
            self._groq_available = True
            logger.info("Groq Prompt Guard (Layer 3) enabled")
            return True
        except ImportError:
            logger.debug(
                "groq package not installed, Layer 3 (LLM) disabled. "
                "Install with: pip install groq"
            )
            self._groq_available = False
            return False


# ── Singleton ──────────────────────────────────────────────
_instance: Optional[PromptInjectionGuard] = None


def get_prompt_guard() -> PromptInjectionGuard:
    """Get or create singleton PromptInjectionGuard instance."""
    global _instance
    if _instance is None:
        _instance = PromptInjectionGuard()
    return _instance
