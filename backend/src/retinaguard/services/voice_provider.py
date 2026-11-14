"""Voice synthesis provider abstraction with AgriNexus 11-language parity.

Implements a 3-tier synthesis chain:
  Tier 1: Sarvam AI Bulbul:v3 (model="bulbul:v3", speaker="shubh")
  Tier 2: Edge-TTS fallback tier
  Tier 3: Browser Web Speech API fallback (sentinel raising NotImplementedError -> HTTP 503)

Privacy Guard:
  - ZERO text logging. No clinical advice text or inputs are logged at any log level.
  - Callers must pass only generic clinical advice, never patient identifiers.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import wave
from abc import ABC, abstractmethod
from typing import Optional

import httpx

logger = logging.getLogger("retinaguard.services.voice_provider")

# ---------------------------------------------------------------------------
# 11-Language Mapping: hi/pa/te/ta/ml/kn/bn/mr/gu/od/en -> xx-IN
# ---------------------------------------------------------------------------

SARVAM_LANGUAGE_MAP: dict[str, str] = {
    "hi": "hi-IN",
    "pa": "pa-IN",
    "te": "te-IN",
    "ta": "ta-IN",
    "ml": "ml-IN",
    "kn": "kn-IN",
    "bn": "bn-IN",
    "mr": "mr-IN",
    "gu": "gu-IN",
    "od": "od-IN",
    "en": "en-IN",
}

EDGE_TTS_VOICE_MAP: dict[str, str] = {
    "hi": "hi-IN-MadhurNeural",
    "pa": "pa-IN-GurpreetNeural",
    "te": "te-IN-MohanNeural",
    "ta": "ta-IN-ValluvarNeural",
    "ml": "ml-IN-MidhunNeural",
    "kn": "kn-IN-GaganNeural",
    "bn": "bn-IN-BashkarNeural",
    "mr": "mr-IN-ManoharNeural",
    "gu": "gu-IN-NiranjanNeural",
    "od": "hi-IN-MadhurNeural",  # Fallback voice for Odia if od-IN unindexed
    "en": "en-IN-NeerjaNeural",
}


# ---------------------------------------------------------------------------
# Text Chunking Helper (Delimiters: . ! ? । ॥ \n ;)
# ---------------------------------------------------------------------------


def _split_text(text: str, max_chars: int = 450) -> list[str]:
    """Split text into chunks <= max_chars, preferring sentence boundaries.

    Handles English punctuation (. ! ?), Devanagari danda (। U+0964, ॥ U+0965),
    semicolons, and newlines.
    """
    if not text:
        return []

    # Delimiters captured in matching groups
    raw_parts = re.split(r"([.!?।॥;\n]+)", text)
    sentences: list[str] = []
    i = 0
    while i < len(raw_parts):
        s = raw_parts[i].strip()
        delim = raw_parts[i + 1].strip() if i + 1 < len(raw_parts) else ""
        if s or delim:
            combined = f"{s}{delim}".strip()
            if combined:
                sentences.append(combined)
        i += 2

    chunks: list[str] = []
    current = ""
    for s in sentences:
        candidate = f"{current} {s}".strip() if current else s
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(s) > max_chars:
                # Word-level fallback for unusually long single sentence
                words = s.split()
                sub = ""
                for w in words:
                    sub_cand = f"{sub} {w}".strip() if sub else w
                    if len(sub_cand) <= max_chars:
                        sub = sub_cand
                    else:
                        if sub:
                            chunks.append(sub)
                        sub = w
                if sub:
                    chunks.append(sub)
                current = ""
            else:
                current = s

    if current:
        chunks.append(current)

    return chunks or [text[:max_chars]]


# ---------------------------------------------------------------------------
# WAV Frame Concatenation Helper
# ---------------------------------------------------------------------------


def merge_wav_bytes(wav_list: list[bytes]) -> bytes:
    """Merge multiple RIFF/WAV byte streams into a single valid WAV stream.

    Strips intermediate headers and recomputes the global RIFF header.
    """
    if not wav_list:
        return b""
    if len(wav_list) == 1 and wav_list[0].startswith(b"RIFF"):
        return wav_list[0]

    out_io = io.BytesIO()
    params = None
    frames: list[bytes] = []

    for w_bytes in wav_list:
        if not w_bytes.startswith(b"RIFF"):
            frames.append(w_bytes)
            continue
        try:
            with wave.open(io.BytesIO(w_bytes), "rb") as r:
                if params is None:
                    params = r.getparams()
                frames.append(r.readframes(r.getnframes()))
        except Exception:
            frames.append(w_bytes)

    if params is None:
        return b"".join(wav_list)

    with wave.open(out_io, "wb") as w:
        w.setparams(params)
        for f in frames:
            w.writeframes(f)

    return out_io.getvalue()


# ---------------------------------------------------------------------------
# Abstract Base Provider
# ---------------------------------------------------------------------------


class VoiceProvider(ABC):
    """Abstract voice synthesis provider."""

    @abstractmethod
    async def synthesize(self, text: str, language: str) -> bytes:
        """Return raw audio bytes (WAV)."""
        ...


# ---------------------------------------------------------------------------
# Tier 1: Sarvam Bulbul:v3
# ---------------------------------------------------------------------------


class SarvamVoiceProvider(VoiceProvider):
    """Calls the Sarvam AI Bulbul:v3 Text-to-Speech API.

    API: https://api.sarvam.ai/text-to-speech
    Model: bulbul:v3
    Speaker: shubh
    Max chunk: 450 characters
    Zero logging of text content.
    """

    API_URL = "https://api.sarvam.ai/text-to-speech"
    MAX_CHARS = 450

    async def synthesize(self, text: str, language: str) -> bytes:
        api_key = os.environ.get("SARVAM_API_KEY", "")
        if not api_key:
            raise RuntimeError("SARVAM_API_KEY not configured")

        target_lang = SARVAM_LANGUAGE_MAP.get(language, "en-IN")
        chunks = _split_text(text, self.MAX_CHARS)
        logger.debug("Sarvam request: %d chunks for target_lang=%s", len(chunks), target_lang)

        audio_parts: list[bytes] = []
        async with httpx.AsyncClient(timeout=30) as client:
            for chunk in chunks:
                resp = await client.post(
                    self.API_URL,
                    headers={"api-subscription-key": api_key},
                    json={
                        "inputs": [chunk],
                        "target_language_code": target_lang,
                        "model": "bulbul:v3",
                        "speaker": "shubh",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                audio_b64: str = data["audios"][0]
                audio_parts.append(base64.b64decode(audio_b64))

        return merge_wav_bytes(audio_parts)


# ---------------------------------------------------------------------------
# Tier 2: Edge-TTS Fallback Provider
# ---------------------------------------------------------------------------


class EdgeTTSVoiceProvider(VoiceProvider):
    """High-quality neural speech synthesis via Edge-TTS."""

    async def synthesize(self, text: str, language: str) -> bytes:
        try:
            import edge_tts  # type: ignore
        except ImportError as exc:
            raise RuntimeError("edge-tts package is not installed") from exc

        voice = EDGE_TTS_VOICE_MAP.get(language, "en-IN-NeerjaNeural")
        logger.debug("Edge-TTS request: voice=%s", voice)

        communicate = edge_tts.Communicate(text, voice)
        chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])

        if not chunks:
            raise RuntimeError("No audio received from Edge-TTS stream")

        return b"".join(chunks)


# ---------------------------------------------------------------------------
# Tier 3: Browser Fallback Sentinel
# ---------------------------------------------------------------------------


class BrowserFallbackProvider(VoiceProvider):
    """Sentinel provider — signals the client to use the Web Speech API."""

    async def synthesize(self, text: str, language: str) -> bytes:
        raise NotImplementedError("Use browser Web Speech API")


# ---------------------------------------------------------------------------
# Resilient Fallback Chain Provider
# ---------------------------------------------------------------------------


class ResilientVoiceProvider(VoiceProvider):
    """Chains Sarvam -> Edge-TTS -> Browser Fallback."""

    def __init__(self, sarvam_provider: Optional[VoiceProvider] = None, edge_provider: Optional[VoiceProvider] = None):
        self.sarvam = sarvam_provider or SarvamVoiceProvider()
        self.edge = edge_provider or EdgeTTSVoiceProvider()

    async def synthesize(self, text: str, language: str) -> bytes:
        # Tier 1: Try Sarvam if API key is configured
        if os.environ.get("SARVAM_API_KEY"):
            try:
                return await self.sarvam.synthesize(text, language)
            except Exception as exc:
                logger.warning("Sarvam TTS failed, trying Edge-TTS: %s", exc)

        # Tier 2: Try Edge-TTS
        try:
            return await self.edge.synthesize(text, language)
        except Exception as exc:
            logger.warning("Edge-TTS failed, falling back to browser speech: %s", exc)

        # Tier 3: Browser fallback sentinel
        raise NotImplementedError("Use browser Web Speech API")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_voice_provider() -> VoiceProvider:
    """Return the resilient multi-tier voice provider."""
    return ResilientVoiceProvider()
