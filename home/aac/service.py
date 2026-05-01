from __future__ import annotations

from dataclasses import asdict
from difflib import SequenceMatcher
import re
from uuid import uuid4
from typing import Any, Dict, Optional

from home.aac.evaluation.metrics import edit_distance_ratio, groundedness_score
from home.aac.logging_utils import log_event
from home.aac.memory.store import memory_store
from home.aac.pipelines.nodes import coarse_face_tag, router_node, utc_now_iso
from home.aac.pipelines.normal_pipeline import run_normal_pipeline
from home.aac.pipelines.speak_pipeline import run_speak_pipeline
from home.aac.types import SessionState


def start_session(user_id: str) -> Dict[str, Any]:
    state = memory_store.start_session(user_id)
    state.stm["loaded_summary"] = {
        "ltm_keys": sorted(list(state.ltm.keys())),
        "stm_sections": sorted(list(state.stm.keys())),
        "pb_phrase_count": len(state.pb.get("phrases", [])),
    }
    state.stm["final_output"] = None
    state.stm["last_options"] = []
    state.stm["active_partner"] = {"person_id": "unknown_partner", "name": "unknown_partner", "relation": "general"}
    state.stm["query_metrics_history"] = []
    return {
        "session_id": state.session_id,
        "user_id": state.user_id,
        "started_at": state.started_at,
        "message": "AAC session started",
        "loaded_summary": state.stm["loaded_summary"],
    }


def _session_or_raise(session_id: str) -> SessionState:
    state = memory_store.get_session(session_id)
    if not state:
        raise ValueError("Session not found. Press START first.")
    return state


def handle_partner_message(
    session_id: str,
    partner_text: str,
    camera_on: bool,
    face_signals: Optional[Dict[str, float]],
    pb_enabled: bool = True,
    partner_name: str = "",
    client_now: str = "",
) -> Dict[str, Any]:
    state = _session_or_raise(session_id)
    if not partner_text or not partner_text.strip():
        raise ValueError("Partner message cannot be empty.")
    partner_context = _resolve_partner(state=state, partner_name=partner_name)
    state.stm["active_partner"] = partner_context
    if client_now:
        state.stm["runtime_clock"] = client_now
    else:
        state.stm["runtime_clock"] = utc_now_iso()
    result = run_normal_pipeline(
        state=state,
        partner_text=partner_text,
        camera_on=camera_on,
        provided_face_signals=face_signals,
        pb_enabled=pb_enabled,
    )
    state.stm["last_partner_text"] = partner_text
    state.stm["last_options"] = result.options
    response = {
        "options": result.options,
        "debug_info": asdict(result.debug_info),
        "evidence_used": [asdict(item) for item in result.evidence_used],
        "template_trace": result.raw.get("template_trace", ""),
        "retrieval_trace": result.raw.get("retrieval_trace", []),
    }
    query_id = str(uuid4())
    evidence_texts = [item.text for item in result.evidence_used]
    option_scores = [groundedness_score(option, evidence_texts) for option in result.options]
    avg_groundedness = round(sum(option_scores) / len(option_scores), 4) if option_scores else 0.0
    unsupported_claims_count = len([score for score in option_scores if score < 0.25])
    hallucination_flag = unsupported_claims_count > 0
    query_metrics = {
        "query_id": query_id,
        "event_type": "normal",
        "timestamp": utc_now_iso(),
        "partner_text": partner_text,
        "partner_detected": result.debug_info.partner_detected,
        "latency_ms": result.debug_info.latency_ms,
        "groundedness_score": avg_groundedness,
        "hallucination_flag": hallucination_flag,
        "unsupported_claims_count": unsupported_claims_count,
        "retrieval_stats": {
            "sources_used": result.debug_info.sources_used,
            "chunks_used": len(result.evidence_used),
            "search_order": result.debug_info.search_order,
            "buckets_chosen": result.debug_info.buckets_chosen,
        },
        "camera_used": bool(camera_on),
        "face_summary": result.debug_info.face_summary,
        "edit_distance": None,
        "acceptance": None,
    }
    state.stm.setdefault("query_metrics_history", []).append(query_metrics)
    response["query_metrics"] = query_metrics
    response["query_metrics_history"] = state.stm["query_metrics_history"][-30:]
    response["query_id"] = query_id
    state.stm["last_generation_context"] = {
        "query_id": query_id,
        "router_label": result.debug_info.router_label,
        "sources_searched_in_order": [step.get("pool") for step in response["retrieval_trace"]],
        "search_order": result.debug_info.search_order,
        "evidence_used": response["evidence_used"],
        "options": result.options,
        "latency_ms": result.debug_info.latency_ms,
        "camera_used": bool(camera_on),
        "face_signals": face_signals,
        "template_trace": response["template_trace"],
        "query_metrics": query_metrics,
    }
    log_event(
        {
            "event": "normal_mode_generation",
            "session_id": state.session_id,
            "user_id": state.user_id,
            "input": {"partner_text": partner_text, "camera_on": camera_on, "face_signals": face_signals},
            "partner_name": partner_context.get("name", ""),
            "router_label": result.debug_info.router_label,
            "buckets_chosen": result.debug_info.buckets_chosen,
            "sources": result.debug_info.sources_used,
            "sources_searched_in_order": [step.get("pool") for step in response["retrieval_trace"]],
            "search_order": result.debug_info.search_order,
            "evidence_used": response["evidence_used"],
            "retrieval_trace": response["retrieval_trace"],
            "template_trace": response["template_trace"],
            "options": result.options,
            "latency_ms": result.debug_info.latency_ms,
            "camera_used": bool(camera_on),
            "face_signals_used": face_signals,
            "pb_enabled": pb_enabled,
            "partner_detected": result.debug_info.partner_detected,
            "groundedness_score": avg_groundedness,
            "hallucination_flag": hallucination_flag,
            "unsupported_claims_count": unsupported_claims_count,
            "retrieval_stats": query_metrics["retrieval_stats"],
            "query_id": query_id,
            "evidence_size": result.debug_info.evidence_size,
            "llm_enabled": result.debug_info.llm_enabled,
            "model_used": result.debug_info.model_used,
            "llm_error": result.debug_info.llm_error,
            "node_trace": result.debug_info.node_trace,
        }
    )
    return response


def handle_speak_mode(
    session_id: str,
    camera_on: bool,
    face_signals: Optional[Dict[str, float]],
    client_now: str = "",
) -> Dict[str, Any]:
    state = _session_or_raise(session_id)
    if client_now:
        state.stm["runtime_clock"] = client_now
    else:
        state.stm["runtime_clock"] = utc_now_iso()
    result = run_speak_pipeline(state=state, camera_on=camera_on, provided_face_signals=face_signals)
    grouped = result.grouped_suggestions
    speak_query_id = str(uuid4())
    speak_metrics = {
        "query_id": speak_query_id,
        "timestamp": utc_now_iso(),
        "event_type": "speak_mode",
        "partner_detected": result.debug_info.partner_detected,
        "latency_ms": result.debug_info.latency_ms,
        "groundedness_score": 1.0,
        "hallucination_flag": False,
        "unsupported_claims_count": 0,
        "retrieval_stats": {
            "sources_used": result.debug_info.sources_used,
            "chunks_used": sum(len(items) for items in grouped.values()),
            "search_order": result.debug_info.search_order,
            "buckets_chosen": result.debug_info.buckets_chosen,
        },
        "camera_used": bool(camera_on),
        "face_summary": result.debug_info.face_summary,
        "edit_distance": None,
        "acceptance": None,
    }
    state.stm.setdefault("query_metrics_history", []).append(speak_metrics)
    log_event(
        {
            "event": "speak_mode_generation",
            "session_id": state.session_id,
            "user_id": state.user_id,
            "input": {"camera_on": camera_on, "face_signals": face_signals},
            "partner_detected": result.debug_info.partner_detected,
            "buckets_chosen": result.debug_info.buckets_chosen,
            "grouped_suggestions": grouped,
            "latency_ms": result.debug_info.latency_ms,
            "query_id": speak_query_id,
            "groundedness_score": 1.0,
            "hallucination_flag": False,
            "unsupported_claims_count": 0,
            "retrieval_stats": speak_metrics["retrieval_stats"],
            "camera_used": bool(camera_on),
            "face_signals_used": face_signals,
            "llm_enabled": result.debug_info.llm_enabled,
            "model_used": result.debug_info.model_used,
            "node_trace": result.debug_info.node_trace,
            "evidence_size": result.debug_info.evidence_size,
        }
    )
    return {
        "grouped_suggestions": grouped,
        "debug_info": asdict(result.debug_info),
        "query_metrics": speak_metrics,
        "query_metrics_history": state.stm["query_metrics_history"][-30:],
        "query_id": speak_query_id,
    }


def get_session_state(session_id: str):
    return memory_store.session_snapshot(session_id)


def _update_pb(state: SessionState, selected_text: str, final_text: str):
    ratio = SequenceMatcher(None, selected_text.strip(), final_text.strip()).ratio() if selected_text else 0.0
    heavily_edited = ratio < 0.6 if selected_text else True
    phrases = state.pb.setdefault("phrases", [])
    if not heavily_edited and selected_text:
        for phrase in phrases:
            if phrase.get("text", "").strip() == selected_text.strip():
                phrase["weight"] = round(float(phrase.get("weight", 1.0)) + 0.15, 2)
                phrase["last_used"] = utc_now_iso()
                return {"action": "boost_existing_phrase", "weight": phrase["weight"]}

    phrases.append(
        {
            "text": final_text,
            "intent": "custom",
            "tone": "neutral",
            "length": "short",
            "partner_type": "friend_or_general",
            "bucket_id": "clarify_calm",
            "weight": 1.1,
            "source": "session_confirmation",
            "last_used": utc_now_iso(),
        }
    )
    return {"action": "add_new_phrase", "heavily_edited": heavily_edited}


def _update_stm(
    state: SessionState,
    partner_text: str,
    final_text: str,
    router_label: str,
    face_signals: Optional[Dict[str, float]],
    active_partner_relation: str,
):
    state.stm.setdefault("recent_turns", [])
    state.stm.setdefault("situation_hints", [])
    state.stm["recent_turns"].append({"partner": partner_text, "response": final_text, "timestamp": utc_now_iso()})
    state.stm["recent_turns"] = state.stm["recent_turns"][-25:]
    state.stm["situation_hints"].append(
        {
            "signature": {
                "intent": router_label,
                "topic": partner_text[:60],
                "partner_type": active_partner_relation or "general",
                "face_cue": coarse_face_tag(face_signals),
            },
            "preference": "short_calm",
            "ttl": "session_only",
        }
    )
    state.stm["situation_hints"] = state.stm["situation_hints"][-30:]


def _infer_partner_memory_bucket(partner_text: str) -> str:
    lower = (partner_text or "").lower()
    if any(token in lower for token in ["tomorrow", "next ", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]):
        return "next_days_plans"
    if any(token in lower for token in ["remind", "reminder", "remember to", "don't forget", "dont forget"]):
        return "reminders"
    if any(token in lower for token in ["today", "tonight", "at ", "movie", "meeting", "check-in", "therapy", "class", "cricket", "lab"]):
        return "today_plans"
    return "current_topic_hints"


def _extract_partner_memory_candidate(partner_text: str) -> str:
    raw = (partner_text or "").strip()
    if not raw:
        return ""
    raw = re.sub(r"\?+$", "", raw).strip()
    if re.match(r"^(are|is|do|did|can|could|would|will|should)\s+", raw.lower()):
        return ""
    if len(raw.split()) < 4:
        return ""
    return raw


def _append_unique_memory_item(items: list, candidate: str) -> bool:
    lowered = candidate.lower()
    for existing in items:
        if str(existing).strip().lower() == lowered:
            return False
    items.append(candidate)
    return True


def _update_stm_partner_memory(state: SessionState, partner_text: str) -> Dict[str, Any]:
    candidate = _extract_partner_memory_candidate(partner_text)
    if not candidate:
        state.stm["memory_update_last_note"] = "No new partner memory item detected to save."
        return {"action": "skipped", "bucket": "", "value": ""}
    bucket = _infer_partner_memory_bucket(candidate)
    bucket_items = state.stm.setdefault(bucket, [])
    if not isinstance(bucket_items, list):
        state.stm[bucket] = []
        bucket_items = state.stm[bucket]
    inserted = _append_unique_memory_item(bucket_items, candidate)
    state.stm["memory_update_last_note"] = (
        f"Noted. I saved this in {bucket}: {candidate}"
        if inserted
        else f"I already have this in {bucket}: {candidate}"
    )
    return {
        "action": "added" if inserted else "already_present",
        "bucket": bucket,
        "value": candidate,
    }


def confirm_response(
    session_id: str,
    partner_text: str,
    selected_text: str,
    final_text: str,
    memory_update_on: bool,
    face_signals: Optional[Dict[str, float]],
    query_id: str = "",
):
    state = _session_or_raise(session_id)
    if not final_text or not final_text.strip():
        raise ValueError("Final text cannot be empty. Select or type a response, then confirm.")
    router_label = router_node(partner_text or final_text)
    transcript_item = {
        "partner_text": partner_text,
        "selected_text": selected_text,
        "final_text": final_text,
        "confirmed_at": utc_now_iso(),
        "router_label": router_label,
    }
    state.transcript.append(transcript_item)
    state.stm.setdefault("confirmed_outputs", []).append(transcript_item)
    state.stm["final_output"] = final_text
    pb_action = {"action": "skipped"}
    stm_action = "skipped"
    partner_memory_action = {"action": "skipped", "bucket": "", "value": ""}
    ltm_action = "requires_approval"
    if memory_update_on:
        pb_action = _update_pb(state, selected_text=selected_text, final_text=final_text)
        _update_stm(
            state=state,
            partner_text=partner_text or "",
            final_text=final_text,
            router_label=router_label,
            face_signals=face_signals,
            active_partner_relation=state.stm.get("active_partner", {}).get("relation", "general"),
        )
        partner_memory_action = _update_stm_partner_memory(state=state, partner_text=partner_text)
        stm_action = "updated"
        memory_store.persist_session_memories(state)
    generation_context = state.stm.get("last_generation_context", {})
    metrics_update = _update_query_metrics_after_confirm(
        state=state,
        query_id=query_id or generation_context.get("query_id", ""),
        selected_text=selected_text,
        final_text=final_text,
    )

    log_event(
        {
            "event": "confirm_response",
            "session_id": session_id,
            "user_id": state.user_id,
            "input": {
                "partner_text": partner_text,
                "selected_text": selected_text,
                "final_text": final_text,
                "memory_update_on": memory_update_on,
            },
            "router_label": generation_context.get("router_label", router_label),
            "sources_searched_in_order": generation_context.get("sources_searched_in_order", []),
            "evidence_used": generation_context.get("evidence_used", []),
            "options": generation_context.get("options", []),
            "latency_ms": generation_context.get("latency_ms", 0),
            "camera_used": generation_context.get("camera_used", False),
            "face_signals_used": generation_context.get("face_signals"),
            "query_id": metrics_update.get("query_id"),
            "edit_distance": metrics_update.get("edit_distance"),
            "acceptance": metrics_update.get("acceptance"),
            "memory_update_actions": {"pb": pb_action, "stm": stm_action, "partner_memory": partner_memory_action, "ltm": ltm_action},
        }
    )
    return {
        "final_output": final_text,
        "transcript": state.transcript,
        "memory_update_actions": {"pb": pb_action, "stm": stm_action, "partner_memory": partner_memory_action, "ltm": ltm_action},
        "memory_update_ack": state.stm.get("memory_update_last_note", ""),
        "metrics_update": metrics_update,
        "query_metrics_history": state.stm.get("query_metrics_history", [])[-30:],
    }


def _resolve_partner(state: SessionState, partner_name: str):
    raw = (partner_name or "").strip().lower()
    people = state.ltm.get("people_and_relationships", [])
    if raw:
        for person in people:
            if person.get("name", "").strip().lower() == raw:
                return {
                    "person_id": person.get("person_id", raw),
                    "name": person.get("name", partner_name),
                    "relation": person.get("relation", "general"),
                }
        return {"person_id": "unknown_partner", "name": partner_name.strip(), "relation": "general"}
    previous = state.stm.get("active_partner", {})
    if previous:
        return previous
    return {"person_id": "unknown_partner", "name": "unknown_partner", "relation": "general"}


def _update_query_metrics_after_confirm(state: SessionState, query_id: str, selected_text: str, final_text: str):
    history = state.stm.get("query_metrics_history", [])
    if not query_id:
        return {"query_id": "", "edit_distance": None, "acceptance": None}
    for item in reversed(history):
        if item.get("query_id") == query_id:
            item["edit_distance"] = edit_distance_ratio(selected_text or "", final_text or "")
            item["acceptance"] = bool((selected_text or "").strip() == (final_text or "").strip() and bool(selected_text))
            return {"query_id": query_id, "edit_distance": item["edit_distance"], "acceptance": item["acceptance"]}
    return {"query_id": query_id, "edit_distance": None, "acceptance": None}
