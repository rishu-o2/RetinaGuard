"""Voice synthesis provider abstraction.

Supports:
- SarvamVoiceProvider: Calls the Sarvam AI TTS API (Hindi & English-IN).
- BrowserFallbackProvider: Sentinel that tells the client to use Web Speech API.

PHI Note: This module does NOT log the text at INFO level — only at DEBUG.
Callers MUST pass only generic clinical advice text, never patient-identifying data.
"""

from __future__ import annotations

import base64
import logging
import os
from abc import ABC, abstractmethod

import httpx

logger = logging.getLogger("retinaguard.services.voice_provider")


# ---------------------------------------------------------------------------
# Text chunking helper
# ---------------------------------------------------------------------------


def _split_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks <= max_chars, preferring sentence boundaries.

    Handles both English full stops and Hindi Devanagari danda (।).
    """
    # Normalise Devanagari danda to full stop for splitting
    normalised = text.replace("।", ".")
    sentences = normalised.split(".")
    chunks: list[str] = []
    current = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        candidate = (current + ". " + s) if current else s
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current + ".")
            # If single sentence exceeds max_chars, hard-truncate
            if len(s) > max_chars:
                chunks.append(s[:max_chars])
                current = ""
            else:
                current = s
    if current:
        chunks.append(current + ".")
    return chunks or [text[:max_chars]]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class VoiceProvider(ABC):
    """Abstract voice synthesis provider."""

    @abstractmethod
    async def synthesize(self, text: str, language: str) -> bytes:
        """Return raw audio bytes (WAV or MP3).

        Args:
            text:     Plain text to synthesize.  Must NOT contain PHI.
            language: "en" or "hi".

        Returns:
            Raw audio bytes.
        """
        ...


# ---------------------------------------------------------------------------
# Sarvam provider
# ---------------------------------------------------------------------------


class SarvamVoiceProvider(VoiceProvider):
    """Calls the Sarvam AI Text-to-Speech API.

    API reference: https://docs.sarvam.ai/api-reference/text-to-speech

    Text is NOT logged at INFO level to avoid inadvertent PHI exposure.
    """

    API_URL = "https://api.sarvam.ai/text-to-speech"
    MAX_CHARS = 500  # Sarvam per-request character limit

    async def synthesize(self, text: str, language: str) -> bytes:
        """Synthesize speech via Sarvam API.

        Splits long text into sentence-bounded chunks and concatenates the
        resulting audio parts.
        """
        api_key = os.environ.get("SARVAM_API_KEY", "")
        if not api_key:
            raise RuntimeError("SARVAM_API_KEY not set")

        # Map language to Sarvam codes
        model = "bulbul:v1"
        target_lang = "hi-IN" if language == "hi" else "en-IN"
        speaker = "meera"

        chunks = _split_text(text, self.MAX_CHARS)
        logger.debug("Synthesising %d chunk(s) in language=%s", len(chunks), language)

        audio_parts: list[bytes] = []
        async with httpx.AsyncClient(timeout=30) as client:
            for chunk in chunks:
                resp = await client.post(
                    self.API_URL,
                    headers={"api-subscription-key": api_key},
                    json={
                        "inputs": [chunk],
                        "target_language_code": target_lang,
                        "model": model,
                        "speaker": speaker,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                # Sarvam returns base64-encoded audio per input
                audio_b64: str = data["audios"][0]
                audio_parts.append(base64.b64decode(audio_b64))

        return b"".join(audio_parts)


# ---------------------------------------------------------------------------
# Browser fallback sentinel
# ---------------------------------------------------------------------------


class BrowserFallbackProvider(VoiceProvider):
    """Sentinel provider — instructs the client to use the Web Speech API.

    Raises NotImplementedError; the API layer catches this and returns 503
    with {"fallback": "browser_speech"}.
    """

    async def synthesize(self, text: str, language: str) -> bytes:
        raise NotImplementedError("Use browser Web Speech API")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_voice_provider() -> VoiceProvider:
    """Return the appropriate provider based on environment configuration."""
    if os.environ.get("SARVAM_API_KEY"):
        return SarvamVoiceProvider()
    return BrowserFallbackProvider()
