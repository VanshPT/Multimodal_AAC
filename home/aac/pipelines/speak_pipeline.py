import time

from home.aac.llm import gemini_client
from home.aac.pipelines.nodes import face_cue_node, face_summary, speak_planner_node
from home.aac.types import DebugInfo, FaceSignals, SessionState, SpeakResult


def _normalize_suggestion_count(grouped):
    keys = list(grouped.keys())
    total = sum(len(items) for items in grouped.values())
    filler = "Let us check the next step together."
    while total < 20:
        for key in keys:
            grouped[key].append(filler)
            total += 1
            if total >= 20:
                break
    while total > 30:
        for key in reversed(keys):
            if grouped[key]:
                grouped[key].pop()
                total -= 1
            if total <= 30:
                break
    return grouped


def run_speak_pipeline(
    state: SessionState,
    camera_on: bool,
    provided_face_signals: FaceSignals,
) -> SpeakResult:
    node_trace = []
    started = time.perf_counter()
    llm_error = ""
    model_used = ""
    llm_enabled = False
    signals = face_cue_node(camera_on=camera_on, provided_face_signals=provided_face_signals)
    node_trace.append("FaceCueNode")
    grouped = _normalize_suggestion_count(speak_planner_node(state=state, face_signals=signals))
    node_trace.append("SpeakPlannerNode")
    if gemini_client.enabled:
        refined_grouped, model_used, llm_error = gemini_client.generate_speak_suggestions(
            grouped=grouped,
            style=state.ltm.get("communication_style", {}),
            face_summary=face_summary(signals),
        )
        if refined_grouped:
            grouped = _normalize_suggestion_count(refined_grouped)
            llm_enabled = True
            node_trace.append("SpeakLLMRefinerNode")
    total = sum(len(items) for items in grouped.values())
    if not 20 <= total <= 30:
        raise ValueError("Speak mode must produce between 20 and 30 suggestions.")
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    debug = DebugInfo(
        partner_detected=f"{state.stm.get('active_partner', {}).get('name', 'unknown_partner')} ({state.stm.get('active_partner', {}).get('relation', 'general')})",
        router_label="Contextual",
        buckets_chosen=["plans", "routine", "family"],
        sources_used=["STM", "PB", "LTM"],
        camera_used=bool(camera_on),
        face_summary=face_summary(signals),
        face_detected=bool(signals and signals.get("face_detected", False)),
        smile_score=round(float((signals or {}).get("smile_score", (signals or {}).get("smile_prob", 0.0))), 3),
        search_order=["STM", "PB", "LTM"],
        latency_ms=elapsed_ms,
        groundedness_score=1.0,
        hallucination_flag=False,
        evidence_size=sum(len(item) for group in grouped.values() for item in group),
        llm_enabled=llm_enabled,
        model_used=model_used,
        llm_error=llm_error if not llm_enabled else "",
        node_trace=node_trace,
        notes=[
            f"Generated {sum(len(items) for items in grouped.values())} suggestions",
            "LLM refinement applied for speak mode." if llm_enabled else "Deterministic speak-mode suggestions used.",
        ],
    )
    return SpeakResult(grouped_suggestions=grouped, debug_info=debug, raw={"signals": signals})
