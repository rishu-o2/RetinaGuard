"""Voice Agent for RetinaGuard.

Generates accessible, plain-language spoken audio guidance in English or Hindi.
Designed for browser Web Speech API playback on the clinician/patient terminal.
Implements a swappable VoiceProvider interface for future cloud/edge TTS providers.

Strict Privacy Guard:
- Zero PHI in voice text: No patient names, ABHA IDs, or personal identifiers.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from retinaguard.agents.state import ScreeningAgentState
from retinaguard.core.logging import get_logger

logger = get_logger("retinaguard.agents.voice")

_LOCALES_CACHE: dict[str, dict[str, Any]] = {}


def _get_locale_dict(lang: str) -> dict[str, Any]:
    if lang in _LOCALES_CACHE:
        return _LOCALES_CACHE[lang]
    try:
        root = Path(__file__).resolve().parents[4]
        p = root / "configs" / "locales" / f"{lang}.json"
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            _LOCALES_CACHE[lang] = data
            return data
    except Exception:
        pass
    return {}


class VoiceProvider(ABC):
    """Abstract provider interface for Text-to-Speech synthesis."""

    @abstractmethod
    def synthesize_speech_payload(self, text: str, language: str) -> dict:
        """Produce an audio payload or browser speech synthesis directive."""
        pass


class WebSpeechVoiceProvider(VoiceProvider):
    """Browser Web Speech API provider delivering text and locale tags."""

    def synthesize_speech_payload(self, text: str, language: str) -> dict:
        lang_tag = "hi-IN" if language.lower() == "hi" else "en-IN"
        return {
            "provider": "web_speech_api",
            "text": text,
            "lang": lang_tag,
            "rate": 0.95,
            "pitch": 1.0,
        }


class VoiceAgent:
    """Agent that produces clear spoken narrative in English or Hindi."""

    def __init__(self, provider: Optional[VoiceProvider] = None) -> None:
        self.provider = provider or WebSpeechVoiceProvider()

    def generate_spoken_text(self, state: ScreeningAgentState) -> str:
        """Compose a concise, medical plain-language spoken sentence with zero PHI."""
        lang = state.get("language_requested", "en").lower()
        is_uncertain = state.get("is_uncertain", False)
        decision = state.get("referral_decision")
        inference_out = state.get("inference_output")

        # 1. UNCERTAIN outcome speech
        if is_uncertain or decision is None or decision.is_uncertain:
            if lang == "hi":
                return (
                    "सावधानी: रेटिना छवि की गुणवत्ता अनिश्चित है और इसकी पुष्टि नहीं हो सकी। "
                    "यह परिणाम अनिश्चित है। कृपया डॉक्टर से संपर्क करें या पुनः अच्छी रोशनी में नई फोटो लें।"
                )
            if lang not in ("en", "hi"):
                loc = _get_locale_dict(lang)
                if loc and loc.get("warnings", {}).get("quality_failure"):
                    q_msg = loc["warnings"]["quality_failure"]
                    ref_msg = loc.get("referral", {}).get("UNCERTAIN", "Clinician review required.")
                    return f"{q_msg} {ref_msg}"
            return (
                "Attention: Retinal image quality check could not be completed with confidence. "
                "The result is UNCERTAIN. A clinician review or image recapture is mandatory. "
                "Do not make clinical decisions on this image alone."
            )

        # 2. Confident result speech
        grade = inference_out.predicted_grade if inference_out else None
        grade_num = grade.value if grade is not None else 0
        urgency_str = decision.urgency.value.lower()
        interval = decision.follow_up_interval_months

        if lang == "hi":
            if decision.referral.value == "REFERRAL":
                return (
                    f"रेटिनागार्ड जांच पूर्ण हुई। डायबिटिक रेटिनोपैथी ग्रेड {grade_num} पाया गया है। "
                    f"विशेषज्ञ नेत्र चिकित्सक को दिखाना आवश्यक है। तत्परता: {urgency_str}। "
                    f"डॉक्टर द्वारा समीक्षा अनिवार्य है।"
                )
            return (
                f"रेटिनागार्ड जांच पूर्ण हुई। डायबिटिक रेटिनोपैथी ग्रेड {grade_num} पाया गया है। "
                f"वर्तमान में रेफरल आवश्यक नहीं है। {interval} महीने में अगली नियमित जांच की सलाह दी जाती है। "
                f"डॉक्टर द्वारा समीक्षा अनिवार्य है।"
            )

        # Other supported Indian locales
        if lang not in ("en", "hi"):
            loc = _get_locale_dict(lang)
            if loc and loc.get("grades"):
                g_label = loc.get("grades", {}).get(str(grade_num), f"Grade {grade_num}")
                ref_key = "REFERABLE" if decision.referral.value == "REFERRAL" else "NON_REFERABLE"
                ref_label = loc.get("referral", {}).get(ref_key, "")
                urg_label = loc.get("urgency", {}).get(urgency_str.upper(), urgency_str)
                return f"RetinaGuard: {g_label}. {ref_label}. {urg_label}."

        # English
        if decision.referral.value == "REFERRAL":
            return (
                f"RetinaGuard screening complete. Diabetic retinopathy grade {grade_num} detected. "
                f"This requires specialist ophthalmic referral with urgency level {urgency_str}. "
                f"Mandatory clinician review required."
            )
        return (
            f"RetinaGuard screening complete. Diabetic retinopathy grade {grade_num} detected. "
            f"Routine screening recommended in {interval} months. "
            f"Mandatory clinician review required."
        )

    def __call__(self, state: ScreeningAgentState) -> dict:
        """Synthesize spoken text and voice directive."""
        lang = state.get("language_requested", "en")
        spoken_text = self.generate_spoken_text(state)
        voice_payload = self.provider.synthesize_speech_payload(spoken_text, language=lang)

        logger.info("VoiceAgent: Generated voice narrative (%s, %d chars)", lang, len(spoken_text))
        return {
            "voice_text": spoken_text,
            "voice_payload": voice_payload,
        }

