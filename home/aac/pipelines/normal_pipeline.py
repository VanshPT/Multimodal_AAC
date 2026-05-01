import time
from typing import Dict, List

from home.aac.evaluation.metrics import groundedness_score
from home.aac.llm import gemini_client
from home.aac.pipelines.nodes import (
    bucket_selector_node,
    candidate_generator_node,
    enforce_memory_signature,
    enforce_option_quality,
    enforce_do_not_say,
    enforce_grounded_personal_facts,
    face_cue_node,
    face_summary,
    groundedness_guard_node,
    is_short_greeting,
    retrieve_from_pool_node,
    router_node,
    source_priority_planner_node,
)
from home.aac.types import DebugInfo, FaceSignals, PipelineResult, SessionState


def run_normal_pipeline(
    state: SessionState,
    partner_text: str,
    camera_on: bool,
    provided_face_signals: FaceSignals,
    top_k: int = 4,
    pb_enabled: bool = True,
) -> PipelineResult:
    node_trace: List[str] = []
    started = time.perf_counter()
    signals = face_cue_node(camera_on=camera_on, provided_face_signals=provided_face_signals)
    node_trace.append("FaceCueNode")
    notes = []
    llm_errors = []
    router_model = ""
    refiner_model = ""
    generator_model = ""
    llm_label = None
    greeting_query = is_short_greeting(partner_text)
    if gemini_client.enabled:
        llm_label, router_model, router_error = gemini_client.classify_router(partner_text)
        if not llm_label and router_error:
            llm_errors.append("Router fallback.")
    lower = partner_text.lower()
    contextual_markers = ["today", "tonight", "now", "later", "plan", "schedule", "meeting", "movie", "leave", "time"]
    label = llm_label or router_node(partner_text)
    if greeting_query or any(marker in lower for marker in contextual_markers):
        label = "Contextual"
    node_trace.append("RouterNode")
    search_order = ["PB", "STM", "LTM"] if is_short_greeting(partner_text) else source_priority_planner_node(label)
    node_trace.append("SourcePriorityPlannerNode")
    active_partner_relation = state.stm.get("active_partner", {}).get("relation", "general")
    chosen_buckets = bucket_selector_node(
        partner_text=partner_text,
        label=label,
        active_partner_relation=active_partner_relation,
        face_signals=signals,
    )
    evidence, retrieval_notes, retrieval_trace = retrieve_from_pool_node(
        state=state,
        partner_text=partner_text,
        search_order=search_order,
        chosen_buckets=chosen_buckets,
        top_k=top_k,
    )
    notes.extend(retrieval_notes)
    node_trace.append("RetrieveFromPoolNode")
    evidence_lines = [item.text for item in evidence]
    refined_lines = None
    if gemini_client.enabled:
        refined_lines, refiner_model, refiner_error = gemini_client.refine_evidence(partner_text, evidence_lines)
        if refined_lines is None and refiner_error:
            llm_errors.append("Evidence refiner fallback.")
    if refined_lines is not None:
        evidence_lines = [line for line in refined_lines if line.strip()]
    node_trace.append("EvidenceRefinerNode")
    # Rebuild evidence objects with refined lines for guard/generation.
    if evidence_lines:
        for idx, line in enumerate(evidence_lines):
            if idx < len(evidence):
                evidence[idx].text = line
    else:
        evidence = []
    guardrails = groundedness_guard_node(evidence)
    node_trace.append("GroundednessGuardNode")
    pb_exemplars: List[str] = []
    relation = state.stm.get("active_partner", {}).get("relation", "general")
    partner_type = "friend_or_general" if relation == "friend" else "general"
    style_key = "friends" if partner_type == "friend_or_general" else partner_type
    partner_style_hint = state.ltm.get("partner_style_preferences", {}).get(style_key, "")
    if pb_enabled:
        all_phrases = state.pb.get("phrases", [])
        active_partner_id = state.stm.get("active_partner", {}).get("person_id", "unknown_partner")
        partner_matches = [item.get("text", "") for item in all_phrases if item.get("partner_id") == active_partner_id]
        matching = partner_matches or [item.get("text", "") for item in all_phrases if item.get("partner_type") in {partner_type, "friend_or_general"}]
        pb_exemplars = (matching or [item.get("text", "") for item in all_phrases])[:3]
    options = candidate_generator_node(
        state=state,
        partner_text=partner_text,
        label=label,
        evidence=evidence,
        guardrails=guardrails,
        face_signals=signals,
        pb_exemplars=pb_exemplars,
        partner_style_hint=partner_style_hint,
    )
    if gemini_client.enabled:
        llm_options, generator_model, generator_error = gemini_client.generate_candidates(
            partner_text=partner_text,
            evidence_block=guardrails.get("instruction", ""),
            style=state.ltm.get("communication_style", {}),
            pb_exemplars=pb_exemplars,
            face_summary=face_summary(signals),
        )
        if llm_options:
            options = llm_options + [options[-1]]
        elif generator_error:
            llm_errors.append("Candidate generator fallback.")
    node_trace.append("CandidateGeneratorNode")
    template_trace = ""
    if options and options[-1].startswith("template_trace::"):
        template_trace = options.pop().split("template_trace::", 1)[1]
    options = enforce_memory_signature(options=options, evidence=evidence, label=label)
    options = enforce_grounded_personal_facts(options=options, evidence=evidence)
    options = enforce_do_not_say(options=options, do_not_say=state.ltm.get("do_not_say", []))
    options = enforce_option_quality(options)
    # Enforce exactly 3 options every run.
    if len(options) < 3:
        while len(options) < 3:
            options.append("Can you share one more detail so I answer clearly?")
    options = options[:3]
    evidence_texts = [item.text for item in evidence]
    option_ground_scores = [groundedness_score(option, evidence_texts) for option in options]
    avg_groundedness = round(sum(option_ground_scores) / len(option_ground_scores), 4) if option_ground_scores else 0.0
    hallucination_flag = any(score < 0.25 for score in option_ground_scores)
    evidence_size = sum(len(item.text) for item in evidence)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    llm_success = bool(router_model and refiner_model and generator_model)
    if llm_errors:
        notes.append("LLM fallback active.")
    debug = DebugInfo(
        partner_detected=f"{state.stm.get('active_partner', {}).get('name', 'unknown_partner')} ({state.stm.get('active_partner', {}).get('relation', 'general')})",
        router_label=label,
        buckets_chosen=chosen_buckets,
        sources_used=sorted(list({ev.pool for ev in evidence})),
        camera_used=bool(camera_on),
        face_summary=face_summary(signals),
        face_detected=bool(signals and signals.get("face_detected", False)),
        smile_score=round(float((signals or {}).get("smile_score", (signals or {}).get("smile_prob", 0.0))), 3),
        search_order=search_order,
        latency_ms=elapsed_ms,
        groundedness_score=avg_groundedness,
        hallucination_flag=hallucination_flag,
        evidence_size=evidence_size,
        llm_enabled=llm_success,
        model_used=generator_model if llm_success else "",
        llm_error=""
        if llm_success
        else (
            "LLM unavailable, deterministic fallback used."
            if gemini_client.enabled
            else (getattr(gemini_client, "_init_error", "") or "LLM unavailable, deterministic fallback used.")
        ),
        node_trace=node_trace,
        notes=notes,
        nod_score=round(float((signals or {}).get("nod_score", 0.0)), 3),
        shake_score=round(float((signals or {}).get("shake_score", 0.0)), 3),
        negative_score=round(float((signals or {}).get("negative_prob", 0.0)), 3),
    )
    return PipelineResult(
        options=options,
        evidence_used=evidence,
        debug_info=debug,
        raw={
            "guardrails": guardrails,
            "signals": signals,
            "template_trace": template_trace,
            "retrieval_trace": [
                {
                    "pool": step.pool,
                    "attempted": step.attempted,
                    "top_candidates_before_refine": step.top_candidates_before_refine,
                    "refined_evidence": step.refined_evidence,
                    "coverage": step.coverage,
                }
                for step in retrieval_trace
            ],
            "pb_enabled": pb_enabled,
        },
    )
