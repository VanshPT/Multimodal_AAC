import time
from typing import Any, Dict, List, Optional

from home.aac.evaluation.metrics import groundedness_score
from home.aac.llm import gemini_client
from home.aac.multimodal import DEFAULT_TONE_VARIANTS, map_multimodal
from home.aac.parallel_prep import run_parallel_prep
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
    *,
    gesture: Optional[str] = None,
    heart_rate_bpm: Optional[float] = None,
    gaze_region: Optional[str] = None,
    vocal_polarity: Optional[str] = None,
    air_sign_letter: Optional[str] = None,
    facial_emotion: Optional[str] = None,
) -> PipelineResult:
    node_trace: List[str] = []
    started = time.perf_counter()
    signals = face_cue_node(camera_on=camera_on, provided_face_signals=provided_face_signals)
    node_trace.append("FaceCueNode")

    # ---------------- Multimodal Mapping (new) ----------------
    mm_map = map_multimodal(
        face_signals=signals,
        facial_emotion=facial_emotion,
        gesture=gesture,
        heart_rate_bpm=heart_rate_bpm,
        gaze_region=gaze_region,
        vocal_polarity=vocal_polarity,
        air_sign_letter=air_sign_letter,
    )
    node_trace.append("MultimodalMappingNode")

    notes = list(mm_map.get("notes") or [])
    llm_errors = []
    router_model = ""
    refiner_model = ""
    generator_model = ""
    greeting_query = is_short_greeting(partner_text)

    # ---------------- Parallel Prep (intent + memory) ----------------
    prep = run_parallel_prep(
        partner_text=partner_text,
        llm_intent_fn=gemini_client.classify_router if gemini_client.enabled else None,
    )
    node_trace.extend(prep.get("node_trace") or [])
    if prep.get("intent_source") == "llm":
        router_model = prep.get("intent_model", "")
    elif prep.get("intent_error"):
        llm_errors.append(f"Router fallback: {prep['intent_error']}")
    llm_label = prep.get("intent")

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
        gaze_boosts=mm_map.get("bucket_boosts"),
        bucket_acceptance=state.stm.get("bucket_acceptance", {}),
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
    # Decide tone mode:
    #   - emotion known (face-api.js or simulated)  -> all 3 options in that tone
    #   - no emotion (camera off / unavailable)     -> 3 options in 3 tones
    tone_locked = bool(mm_map.get("tone_locked"))
    tone_variants = None if tone_locked else list(DEFAULT_TONE_VARIANTS)
    if tone_locked:
        notes.append(f"tone_mode=locked tone={mm_map.get('tone')}")
    else:
        notes.append(f"tone_mode=variants tones={DEFAULT_TONE_VARIANTS}")

    if gemini_client.enabled:
        llm_options, generator_model, generator_error = gemini_client.generate_candidates(
            partner_text=partner_text,
            evidence_block=guardrails.get("instruction", ""),
            style=state.ltm.get("communication_style", {}),
            pb_exemplars=pb_exemplars,
            face_summary=face_summary(signals),
            polarity=mm_map.get("polarity", ""),
            tone=mm_map.get("tone", ""),
            verbosity=mm_map.get("verbosity", ""),
            tone_variants=tone_variants,
        )
        if llm_options:
            options = llm_options + [options[-1]]
        elif generator_error:
            llm_errors.append("Candidate generator fallback.")
    elif tone_variants:
        # LLM disabled -> apply tone-variant prefixes to the rule-based options
        # so the user still sees three obviously-different tones.
        options = _apply_tone_variants_fallback(options, tone_variants)
    node_trace.append("CandidateGeneratorNode")
    template_trace = ""
    if options and options[-1].startswith("template_trace::"):
        template_trace = options.pop().split("template_trace::", 1)[1]
    options = enforce_memory_signature(options=options, evidence=evidence, label=label)
    options = enforce_grounded_personal_facts(options=options, evidence=evidence)
    options = enforce_do_not_say(options=options, do_not_say=state.ltm.get("do_not_say", []))
    options = enforce_option_quality(options)
    options = _apply_polarity_guard(options, mm_map)
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
            "multimodal_map": mm_map,
            "parallel_prep": prep,
        },
    )


# ---------------------------------------------------------------
# Polarity guard: rewrite first option to start with the multimodal-cued
# polarity word so polarity adherence stays at 100% even when the LLM is
# offline.
# ---------------------------------------------------------------
def _apply_polarity_guard(options: List[str], mm_map: Dict[str, Any]) -> List[str]:
    """Soft polarity enforcement.

    Strategy:
      - Identify positive/negative/clarify lexicons.
      - For each option, check if any token from the *target* lexicon appears
        in the first 8 words. If yes, leave it alone (already aligned).
      - If not, replace the leading clause (up to first comma, max 4 words)
        with the canonical opener, preserving the rest of the sentence.
    """
    if not options:
        return options
    polarity = mm_map.get("polarity")
    if polarity not in {"positive", "negative", "clarify"}:
        return options

    pos_lex = {"yes", "sure", "okay", "ok", "totally", "agreed", "of course",
               "i'm in", "sounds good", "works for me", "yep", "yeah"}
    neg_lex = {"no", "not", "nope", "sorry", "i can't", "cannot", "i'd rather not",
               "would rather not", "pass", "not now"}
    clr_lex = {"clarify", "what do you mean", "could you", "can you repeat",
               "say that again", "not sure"}

    if polarity == "positive":
        target_lex, opener = pos_lex, "Yes,"
    elif polarity == "negative":
        target_lex, opener = neg_lex, "No,"
    else:
        target_lex, opener = clr_lex, "Could you clarify"

    def _has_token(text: str, lex) -> bool:
        head = " ".join(text.split()[:8]).lower()
        return any(tok in head for tok in lex)

    # Only enforce polarity on the lead option; preserve others for variety.
    lead = options[0]
    if _has_token(lead, target_lex):
        return options
    # Strip any existing polarity opener (Yes, / No, / Sure, / Nope, etc.)
    import re as _re
    stripped = _re.sub(
        r"^\s*(yes|no|sure|nope|okay|ok|yeah|yep)[,!.\s]+",
        "",
        lead.lstrip(),
        flags=_re.IGNORECASE,
    ).rstrip(",.!?")
    new_lead = f"{opener} {stripped}".strip()
    if not new_lead.endswith((".", "?", "!")):
        new_lead += "."
    return [new_lead] + list(options[1:])


# ---------------------------------------------------------------
# Tone-variants fallback used when the LLM is offline AND the camera is off
# (no detected emotion). Re-prefixes existing rule-based options so the user
# still sees three obviously different tones.
# ---------------------------------------------------------------
TONE_PREFIX = {
    "warm":          ("Sure thing! ", " 😊"),  # cheerful — kept emoji-free below
    "neutral":       ("",            ""),
    "brief":         ("",            ""),
    "gentle":        ("If it helps, ", ""),
    "assertive":     ("To be clear, ", ""),
    "curious":       ("Wait, really? ", ""),
    "reassuring":    ("Don't worry, ", ""),
    "polite_decline":("Thanks, but ",  ""),
}


def _apply_tone_variants_fallback(options: List[str], tones: List[str]) -> List[str]:
    if not options:
        return options
    base = list(options)[:3]
    while len(base) < 3:
        base.append(base[-1] if base else "Could you share one more detail?")
    out: List[str] = []
    for tone, opt in zip(tones, base):
        if tone == "brief":
            out.append(_shorten_for_brief(opt))
            continue
        prefix, suffix = TONE_PREFIX.get(tone, ("", ""))
        text = (prefix + opt + suffix).strip()
        out.append(text)
    return out + list(options[3:])


def _shorten_for_brief(text: str) -> str:
    # Take the first sentence, cap at 8 words.
    first = text.split(".")[0].strip()
    words = first.split()
    if len(words) > 8:
        first = " ".join(words[:8])
    if not first.endswith((".", "!", "?")):
        first += "."
    return first
