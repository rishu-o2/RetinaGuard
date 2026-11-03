"""RetinaGuard 5-Agent Screening Graph (LangGraph implementation).

Orchestrates:
  1. vision_agent    (Quality gate -> OOD check -> ONNX model)
  2. safety_agent    (Deterministic ClinicalRulesEngine)
  3. guideline_agent (RAG retrieval with safety validator)
  4. audit_agent     (Hash-chained tamper-evident AuditLedger)
  5. voice_agent     (Multilingual speech narrative generator)

Conditional Routing:
  - Normal path: vision -> safety -> guideline -> audit -> voice -> finalize -> END
  - Early-exit path (quality fail / OOD / low confidence):
    vision -> safety -> audit -> voice -> finalize -> END (skips guideline agent)
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from langgraph.graph import END, StateGraph

from retinaguard.agents.audit_agent import AuditAgent
from retinaguard.agents.guideline_agent import GuidelineAgent
from retinaguard.agents.safety_agent import SafetyAgent
from retinaguard.agents.state import ScreeningAgentState
from retinaguard.agents.vision_agent import VisionAgent
from retinaguard.agents.voice_agent import VoiceAgent
from retinaguard.core.logging import get_logger
from retinaguard.core.models import (
    EyeSide,
    ICDRGrade,
    ImageMetadata,
    PatientContext,
    ReferralDecision,
    ScreeningResult,
    Urgency,
)

logger = get_logger("retinaguard.agents.graph")


def finalize_node(state: ScreeningAgentState) -> dict:
    """Assembles the final backward-compatible ScreeningResult model."""
    metadata = state["image_metadata"]
    context = state["patient_context"]
    decision = state.get("referral_decision")
    inference_out = state.get("inference_output")
    is_uncertain = state.get("is_uncertain", False)
    uncertainty_reasons = state.get("uncertainty_reasons", [])

    grade = inference_out.predicted_grade if (inference_out and not is_uncertain) else None
    confidence = inference_out.confidence if (inference_out and not is_uncertain) else None

    # Fallback decision if safety agent somehow didn't populate
    if decision is None:
        referral_enum = ReferralDecision.UNCERTAIN
        urgency_enum = Urgency.ROUTINE
        interval = 0
    else:
        referral_enum = decision.referral
        urgency_enum = decision.urgency
        interval = decision.follow_up_interval_months

    screening_res = ScreeningResult(
        screening_id=uuid4(),
        patient_pseudonym_id=context.patient_pseudonym_id,
        eye_side=metadata.eye_side,
        camera_type=metadata.camera_type,
        grade=grade,
        confidence=confidence,
        referral=referral_enum,
        urgency=urgency_enum,
        follow_up_interval_months=interval,
        is_uncertain=is_uncertain,
        uncertainty_reasons=uncertainty_reasons,
        clinician_review_mandatory=True,
        report_en=state.get("report_en", "Clinician review mandatory."),
        report_hi=state.get("report_hi", "डॉक्टर द्वारा समीक्षा अनिवार्य है।"),
        language_requested=state.get("language_requested", "en"),
        image_sha256=metadata.image_sha256,
        previous_audit_hash=state.get("previous_audit_hash"),
        audit_hash=state.get("audit_hash"),
        voice_text=state.get("voice_text"),
        guideline_explanation=state.get("guideline_explanation"),
        guideline_citations=state.get("guideline_citations", []),
        guideline_source_note=state.get("guideline_source_note"),
    )

    return {"final_result": screening_res}


def should_skip_guideline(state: ScreeningAgentState) -> str:
    """Determine whether to skip the guideline agent due to early-exit uncertainty."""
    if state.get("early_exit") or state.get("is_uncertain"):
        logger.info("Graph routing: Early exit active. Skipping guideline agent -> audit.")
        return "audit"
    return "guideline"


def build_screening_graph(
    vision_agent: Optional[VisionAgent] = None,
    safety_agent: Optional[SafetyAgent] = None,
    guideline_agent: Optional[GuidelineAgent] = None,
    audit_agent: Optional[AuditAgent] = None,
    voice_agent: Optional[VoiceAgent] = None,
) -> Any:
    """Construct and compile the 5-agent LangGraph workflow."""
    v_agent = vision_agent or VisionAgent()
    s_agent = safety_agent or SafetyAgent()
    g_agent = guideline_agent or GuidelineAgent()
    a_agent = audit_agent or AuditAgent()
    vo_agent = voice_agent or VoiceAgent()

    workflow = StateGraph(ScreeningAgentState)

    # 1. Add agent nodes
    workflow.add_node("vision", v_agent)
    workflow.add_node("safety", s_agent)
    workflow.add_node("guideline", g_agent)
    workflow.add_node("audit", a_agent)
    workflow.add_node("voice", vo_agent)
    workflow.add_node("finalize", finalize_node)

    # 2. Add edges & conditional branching
    workflow.set_entry_point("vision")
    workflow.add_edge("vision", "safety")

    # From safety, check if early-exit is active
    workflow.add_conditional_edges(
        "safety",
        should_skip_guideline,
        {
            "audit": "audit",
            "guideline": "guideline",
        },
    )

    workflow.add_edge("guideline", "audit")
    workflow.add_edge("audit", "voice")
    workflow.add_edge("voice", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()


# Module-level compiled graph runner
_compiled_graph = None


def get_screening_graph():
    """Retrieve or build the singleton compiled screening graph."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_screening_graph()
    return _compiled_graph


def run_screening_pipeline(
    image_bytes: bytes,
    image_metadata: ImageMetadata,
    patient_context: PatientContext,
    language_requested: str = "en",
    image_rgb: Optional[Any] = None,
) -> ScreeningResult:
    """Execute the full 5-agent screening pipeline through LangGraph."""
    graph = get_screening_graph()

    initial_state: ScreeningAgentState = {
        "image_bytes": image_bytes,
        "image_rgb": image_rgb,
        "image_metadata": image_metadata,
        "patient_context": patient_context,
        "language_requested": language_requested,
        "is_uncertain": False,
        "uncertainty_reasons": [],
        "early_exit": False,
        "guideline_citations": [],
    }

    result_state = graph.invoke(initial_state)
    return result_state["final_result"]
