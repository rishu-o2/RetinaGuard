"""Tests for voice synthesis provider and /voice endpoint.

Verifies:
- Sentence chunking with max_chars=450 and Devanagari danda (।) / double danda (॥)
- Multi-chunk WAV frame merging into valid RIFF WAV
- 11-language mapping (hi/pa/te/ta/ml/kn/bn/mr/gu/od/en -> xx-IN)
- Sarvam Bulbul:v3 API client payload (model=bulbul:v3, speaker=shubh)
- Multi-tier fallback chain (Sarvam -> Edge-TTS -> Browser 503 fallback)
"""

from __future__ import annotations

import base64
import io
import wave
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from httpx import ASGITransport, AsyncClient

from retinaguard.api.main import app
from retinaguard.services.voice_provider import (
    BrowserFallbackProvider,
    EdgeTTSVoiceProvider,
    ResilientVoiceProvider,
    SARVAM_LANGUAGE_MAP,
    SarvamVoiceProvider,
    _split_text,
    get_voice_provider,
    merge_wav_bytes,
)


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def _create_mock_wav(num_frames: int = 500, sample_rate: int = 16000) -> bytes:
    """Helper to generate valid dummy WAV bytes in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * num_frames)
    return buf.getvalue()


# ── 1. Chunking Tests ─────────────────────────────────────────────────────────


def test_text_chunking_english():
    text = (
        "Diabetic retinopathy screening completed. "
        "No apparent signs of retinopathy observed at this time. "
        "Please schedule your routine comprehensive dilated eye exam in twelve months. "
        "Maintain glycemic control as advised by your physician."
    )
    chunks = _split_text(text, max_chars=80)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 80


def test_text_chunking_hindi_danda():
    text = (
        "मधुमेह रेटिनोपैथी की जांच संपन्न हुई। "
        "वर्तमान में किसी गंभीर रेटिनोपैथी के लक्षण नहीं दिखे हैं। "
        "कृपया 12 महीने के भीतर नियमित अनुवर्ती जांच अवश्य कराएं। "
        "चिकित्सक द्वारा बताए गए रक्त शर्करा स्तर को नियंत्रित रखें।"
    )
    chunks = _split_text(text, max_chars=70)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 70


# ── 2. WAV Merge Tests ───────────────────────────────────────────────────────


def test_wav_merge():
    wav1 = _create_mock_wav(num_frames=600, sample_rate=16000)
    wav2 = _create_mock_wav(num_frames=400, sample_rate=16000)

    merged = merge_wav_bytes([wav1, wav2])
    assert merged.startswith(b"RIFF")

    with wave.open(io.BytesIO(merged), "rb") as r:
        assert r.getnframes() == 1000
        assert r.getframerate() == 16000
        assert r.getnchannels() == 1


# ── 3. 11-Language Mapping Tests ─────────────────────────────────────────────


def test_language_map():
    expected_11 = ["hi", "pa", "te", "ta", "ml", "kn", "bn", "mr", "gu", "od", "en"]
    for lang in expected_11:
        assert lang in SARVAM_LANGUAGE_MAP
        assert SARVAM_LANGUAGE_MAP[lang] == f"{lang}-IN"


# ── 4. Sarvam Bulbul:v3 API Client Payload Tests ─────────────────────────────


@pytest.mark.asyncio
async def test_sarvam_bulbul_v3_request_payload(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "mock_sarvam_api_key")
    provider = SarvamVoiceProvider()

    mock_wav = _create_mock_wav(num_frames=100)
    mock_b64 = base64.b64encode(mock_wav).decode("ascii")

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"audios": [mock_b64]}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result_bytes = await provider.synthesize(
            text="नियमित जांच 12 महीने बाद कराएं।",
            language="hi",
        )

        assert result_bytes.startswith(b"RIFF")
        assert mock_post.called

        # Verify call arguments
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["headers"]["api-subscription-key"] == "mock_sarvam_api_key"
        payload = call_kwargs["json"]
        assert payload["model"] == "bulbul:v3"
        assert payload["speaker"] == "shubh"
        assert payload["target_language_code"] == "hi-IN"


# ── 5. Multi-Tier Fallback Chain Tests ───────────────────────────────────────


@pytest.mark.asyncio
async def test_fallback_chain_sarvam_to_edge_tts(monkeypatch):
    """When Sarvam fails or raises, ResilientVoiceProvider falls back to Edge-TTS."""
    monkeypatch.setenv("SARVAM_API_KEY", "mock_key")

    mock_sarvam = MagicMock(spec=SarvamVoiceProvider)
    mock_sarvam.synthesize = AsyncMock(side_effect=Exception("Sarvam HTTP 500 error"))

    mock_edge = MagicMock(spec=EdgeTTSVoiceProvider)
    mock_edge.synthesize = AsyncMock(return_value=b"RIFF_FROM_EDGE_TTS")

    resilient = ResilientVoiceProvider(sarvam_provider=mock_sarvam, edge_provider=mock_edge)
    audio = await resilient.synthesize("Routine follow-up.", "en")

    assert audio == b"RIFF_FROM_EDGE_TTS"
    assert mock_sarvam.synthesize.called
    assert mock_edge.synthesize.called


@pytest.mark.asyncio
async def test_fallback_chain_all_fail_to_browser_speech(client: AsyncClient, monkeypatch):
    """When both Sarvam and Edge-TTS fail, endpoint returns HTTP 503 browser fallback."""
    monkeypatch.setenv("SARVAM_API_KEY", "mock_key")

    with patch("retinaguard.services.voice_provider.ResilientVoiceProvider.synthesize", new_callable=AsyncMock) as mock_synth:
        mock_synth.side_effect = NotImplementedError("Browser fallback")

        resp = await client.post("/voice", json={"text": "Annual review scheduled.", "language": "en"})
        assert resp.status_code == 503
        data = resp.json()
        assert data["fallback"] == "browser_speech"
        assert "browser" in data["message"]


# ── 6. Zero-PHI and No Logging Tests ─────────────────────────────────────────


def test_voice_text_generic_advice_no_phi():
    """Verify that voice text consists only of generic clinical advice without identifiers."""
    from retinaguard.agents.voice_agent import VoiceAgent
    from retinaguard.core.models import EyeSide, ICDRGrade, ReferralDecision, Urgency

    agent = VoiceAgent()
    # Simulate a pipeline state with patient pseudonym and clinical data
    mock_state = {
        "referral_decision": MagicMock(
            referral=ReferralDecision.REFERABLE,
            urgency=Urgency.PRIORITY,
            follow_up_interval_months=3,
        ),
        "inference_output": MagicMock(predicted_grade=ICDRGrade.MODERATE_NPDR),
        "patient_context": MagicMock(patient_pseudonym_id="PAT-DEL-2026-9999"),
        "image_metadata": MagicMock(
            image_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            eye_side=EyeSide.RIGHT_OD,
        ),
        "language_requested": "en",
    }

    result_state = agent(mock_state)
    voice_text = result_state.get("voice_text", "")

    assert voice_text, "Voice text must not be empty"
    # Must NOT contain patient pseudonym ID
    assert "PAT-DEL" not in voice_text
    assert "9999" not in voice_text
    # Must NOT contain image SHA-256 or hex digest
    assert "e3b0c442" not in voice_text
    # Must NOT contain raw internal enum strings or clinician notes
    assert "RIGHT_OD" not in voice_text
    # Must contain generic clinical advice
    assert any(term in voice_text.lower() for term in ["retina", "ophthalmologist", "specialist", "months", "screening"])


@pytest.mark.asyncio
async def test_voice_endpoint_zero_logging(client: AsyncClient, caplog):
    """Verify that text sent to /voice is never logged at any log level."""
    import logging
    caplog.set_level(logging.DEBUG)

    sensitive_phrase = "UniqueClinicalAdvicePhraseXYZZY12345"
    with patch("retinaguard.services.voice_provider.ResilientVoiceProvider.synthesize", new_callable=AsyncMock) as mock_synth:
        mock_synth.return_value = _create_mock_wav(100)
        resp = await client.post("/voice", json={"text": sensitive_phrase, "language": "en"})
        assert resp.status_code == 200

    # Ensure the text was NEVER logged anywhere
    for record in caplog.records:
        assert sensitive_phrase not in record.message
        assert sensitive_phrase not in str(record.args)
