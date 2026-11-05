"""Tests for voice synthesis provider and /voice endpoint."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, patch
import pytest
from httpx import ASGITransport, AsyncClient

from retinaguard.api.main import app
from retinaguard.services.voice_provider import (
    BrowserFallbackProvider,
    SarvamVoiceProvider,
    _split_text,
    get_voice_provider,
)


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def test_text_chunking_english():
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    chunks = _split_text(text, max_chars=30)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 30


def test_text_chunking_hindi_danda():
    text = "पहला वाक्य है। दूसरा वाक्य भी है। तीसरा वाक्य लंबा है और विवरण देता है।"
    chunks = _split_text(text, max_chars=35)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 35


def test_provider_factory_selection(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    p1 = get_voice_provider()
    assert isinstance(p1, BrowserFallbackProvider)

    monkeypatch.setenv("SARVAM_API_KEY", "mock_key_12345")
    p2 = get_voice_provider()
    assert isinstance(p2, SarvamVoiceProvider)


@pytest.mark.asyncio
async def test_voice_endpoint_fallback_when_no_key(client: AsyncClient, monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    resp = await client.post("/voice", json={"text": "Please return for annual follow-up in 12 months.", "language": "en"})
    assert resp.status_code == 503
    data = resp.json()
    assert data["fallback"] == "browser_speech"


@pytest.mark.asyncio
async def test_voice_endpoint_sarvam_success(client: AsyncClient, monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test_sarvam_key")
    mock_audio_bytes = b"RIFF....WAVEfmt ....data...."

    with patch("retinaguard.services.voice_provider.SarvamVoiceProvider.synthesize", new_callable=AsyncMock) as mock_synth:
        mock_synth.return_value = mock_audio_bytes
        resp = await client.post("/voice", json={"text": "Referral to ophthalmologist is recommended.", "language": "en"})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert resp.content == mock_audio_bytes


@pytest.mark.asyncio
async def test_voice_endpoint_sarvam_failure_falls_back(client: AsyncClient, monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test_sarvam_key")

    with patch("retinaguard.services.voice_provider.SarvamVoiceProvider.synthesize", new_callable=AsyncMock) as mock_synth:
        mock_synth.side_effect = Exception("Sarvam network timeout")
        resp = await client.post("/voice", json={"text": "Routine checkup advised.", "language": "en"})
        assert resp.status_code == 503
        assert resp.json()["fallback"] == "browser_speech"
