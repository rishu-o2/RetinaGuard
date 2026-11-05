"""Voice synthesis API endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from retinaguard.core.logging import get_logger
from retinaguard.services.voice_provider import get_voice_provider

logger = get_logger("retinaguard.api.voice")
router = APIRouter(prefix="/voice", tags=["voice"])


class VoiceRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Generic advice text to synthesize. NO PHI.")
    language: str = Field("en", description="'en' or 'hi'")


@router.post("", summary="Synthesize voice for screening advice text")
async def synthesize_voice(request: VoiceRequest) -> Response:
    """Synthesize clinical advice text into spoken audio.

    Uses Sarvam AI if SARVAM_API_KEY is configured.
    Falls back to browser Web Speech API (returns 503 with fallback=browser_speech)
    if no API key is configured or if the provider call fails.
    """
    provider = get_voice_provider()
    try:
        audio_bytes = await provider.synthesize(text=request.text, language=request.language)
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={"Cache-Control": "no-cache"},
        )
    except NotImplementedError:
        # Browser fallback sentinel
        return JSONResponse(
            content={"fallback": "browser_speech", "message": "Using client-side browser speech synthesis"},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    except Exception as exc:
        logger.warning("Voice synthesis provider failed: %s — falling back to browser speech", exc)
        return JSONResponse(
            content={"fallback": "browser_speech", "message": "Voice service unavailable, use browser speech"},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

