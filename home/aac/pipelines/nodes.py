from datetime import datetime, timezone
import re
from typing import Dict, List, Tuple

from home.aac.types import FaceSignals, RetrievalEvidence, RetrievalTraceStep, SessionState
from home.aac.utils import cosine_like_score, keyword_overlap_score, pick_top, tokenize


BUCKET_KEYWORDS = {
    "greeting_smalltalk": {"hi", "hii", "hello", "hey", "yo", "sup"},
    "family": {"mom", "dad", "sister", "brother", "family", "home"},
    "food": {"food", "lunch", "dinner", "eat", "snack", "coffee"},
    "plans": {"today", "tonight", "plan", "movie", "schedule", "event"},
    "routine": {"routine", "morning", "evening", "daily", "usually"},
    "medical": {"pain", "doctor", "med", "therapy", "tired"},
    "decline_polite": {"no", "can't", "not", "later"},
    "agree_casual": {"yes", "sure", "okay", "sounds"},
    "clarify_calm": {"mean", "clarify", "repeat", "explain"},
}

GREETING_TOKENS = {"hi", "hii", "hiii", "hello", "hey", "yo", "sup", "hiya"}
GREETING_VOCATIVES = {"bro", "broo", "man", "dude", "buddy", "bhai"}
GREETING_PHRASES = {"how are you", "how is it going", "how's it going", "good morning", "good afternoon", "good evening", "nice to see you"}
PARTNER_TOPIC_KEYWORDS = {
    "omer": {"movie", "regal", "bus", "stop", "cricket"},
    "vansh": {"project", "check-in", "check", "slides", "cse", "lab", "deadline", "rehearsal"},
    "siddharth": {"grocery", "timeline", "schedule", "charger", "reminder"},
    "aditya": {"cricket", "weekend", "casual"},
}

MEDICATION_QUERY_MARKERS = {
    "medication",
    "medicine",
    "meds",
    "pill",
    "pills",
    "dose",
    "doses",
}


def face_cue_node(camera_on: bool, provided_face_signals: FaceSignals) -> FaceSignals:
    if not camera_on:
        return None
    if not provided_face_signals:
        return None
    smile = float(provided_face_signals.get("smile_score", provided_face_signals.get("smile_prob", 0.0)))
    confused = float(provided_face_signals.get("confused_prob", 0.0))
    negative = float(provided_face_signals.get("negative_prob", provided_face_signals.get("anger_prob", 0.0)))
    nod = float(provided_face_signals.get("nod_score", provided_face_signals.get("head_nod_score", 0.0)))
    shake = float(provided_face_signals.get("shake_score", provided_face_signals.get("head_shake_score", 0.0)))
    hand_gesture = str(provided_face_signals.get("hand_gesture_label", "") or "").strip()
    hand_gesture_score = float(provided_face_signals.get("hand_gesture_score", 0.0) or 0.0)
    if hand_gesture_score >= 0.45:
        if hand_gesture in {"Thumb_Up", "Victory", "ILoveYou"}:
            nod = max(nod, min(1.0, 0.58 + 0.3 * hand_gesture_score))
            smile = max(smile, min(1.0, 0.42 + 0.25 * hand_gesture_score))
            shake = min(shake, 0.12)
            negative = min(negative, 0.15)
        elif hand_gesture in {"Thumb_Down", "Closed_Fist"}:
            shake = max(shake, min(1.0, 0.58 + 0.3 * hand_gesture_score))
            negative = max(negative, min(1.0, 0.4 + 0.35 * hand_gesture_score))
            nod = min(nod, 0.12)
            smile = min(smile, 0.22)
        elif hand_gesture in {"Open_Palm", "Pointing_Up"}:
            confused = max(confused, min(1.0, 0.34 + 0.28 * hand_gesture_score))
            nod = min(nod, 0.18)
            shake = min(shake, 0.18)
    neutral = float(provided_face_signals.get("neutral_prob", max(0.0, 1.0 - smile - confused)))
    strongest = max(smile, confused, neutral, negative, nod, shake)
    inferred_detected = strongest > 0.08 or (smile + confused + neutral) > 0.2
    face_detected = bool(provided_face_signals.get("face_detected", inferred_detected))
    hand_detected = bool(provided_face_signals.get("hand_detected", bool(hand_gesture)))
    return {
        "face_detected": face_detected,
        "smile_score": max(0.0, min(1.0, smile)),
        "smile_prob": max(0.0, min(1.0, smile)),
        "confused_prob": max(0.0, min(1.0, confused)),
        "neutral_prob": max(0.0, min(1.0, neutral)),
        "negative_prob": max(0.0, min(1.0, negative)),
        "nod_score": max(0.0, min(1.0, nod)),
        "shake_score": max(0.0, min(1.0, shake)),
        "hand_detected": hand_detected,
        "hand_gesture_label": hand_gesture,
        "hand_gesture_score": max(0.0, min(1.0, hand_gesture_score)),
        "handedness": str(provided_face_signals.get("handedness", "") or "").strip(),
    }


def router_node(partner_text: str) -> str:
    lower = partner_text.lower()
    if is_short_greeting(partner_text):
        return "Contextual"
    if _is_medication_status_query(partner_text):
        return "Contextual"
    personal_markers = ["your", "family", "favorite", "prefer", "you like", "your mom", "your dad"]
    contextual_markers = ["today", "tonight", "now", "later", "plan", "reminder", "meeting", "movie"]
    open_markers = ["what is", "define", "explain", "who discovered", "how does", "why is"]
    if any(marker in lower for marker in personal_markers):
        return "Personal"
    if any(marker in lower for marker in open_markers):
        return "Open-domain"
    if any(marker in lower for marker in contextual_markers):
        return "Contextual"
    if len(tokenize(partner_text)) <= 3:
        return "Contextual"
    return "Personal" if "you" in lower else "Contextual"


def source_priority_planner_node(label: str) -> List[str]:
    if label == "Contextual":
        return ["STM", "LTM", "PB"]
    if label == "Personal":
        return ["LTM", "PB", "STM"]
    return ["LTM", "PB", "STM"]


def is_short_greeting(partner_text: str) -> bool:
    tokens = tokenize(partner_text)
    cleaned = re.sub(r"[^a-zA-Z]", "", (partner_text or "").lower())
    normalized = re.sub(r"[^a-zA-Z'\s]+", " ", (partner_text or "").lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if cleaned in GREETING_TOKENS:
        return True
    if not tokens:
        return False
    if any(phrase in normalized for phrase in GREETING_PHRASES) and len(tokens) <= 6:
        return True
    if tokens[0] in GREETING_TOKENS and len(tokens) <= 3:
        tail = tokens[1:]
        return not tail or all(token in GREETING_VOCATIVES for token in tail)
    if tokens[0] in GREETING_TOKENS and any(phrase in normalized for phrase in {"how are you", "how's it going", "how is it going"}):
        return True
    return len(tokens) <= 2 and any(token in GREETING_TOKENS for token in tokens)


def bucket_selector_node(partner_text: str, label: str, active_partner_relation: str, face_signals: FaceSignals) -> List[str]:
    selected = []
    if is_short_greeting(partner_text):
        selected.append("greeting_smalltalk")
    inferred = _infer_bucket(partner_text)
    if inferred != "general":
        selected.append(inferred)
    if label == "Contextual":
        selected.extend(["plans", "routine"])
    if _is_medication_status_query(partner_text):
        selected.extend(["medical", "routine"])
    if label == "Personal" and active_partner_relation == "friend":
        selected.append("family")
    if face_signals and face_signals.get("confused_prob", 0.0) > 0.45:
        selected.append("clarify_calm")
    if not selected:
        selected = ["clarify_calm", "routine"]
    # De-duplicate while preserving order
    seen = set()
    ordered = []
    for item in selected:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered[:4]


def _infer_bucket(topic_text: str) -> str:
    text_tokens = set(tokenize(topic_text))
    best_bucket = "general"
    best_score = 0
    for bucket, words in BUCKET_KEYWORDS.items():
        score = len(text_tokens.intersection(words))
        if score > best_score:
            best_score = score
            best_bucket = bucket
    return best_bucket


def _ltm_chunks(state: SessionState):
    chunks = []
    for key in ["favorites", "relationships", "personality_traits", "communication_style", "do_not_say"]:
        value = state.ltm.get(key)
        if isinstance(value, dict):
            text = " ".join([f"{k}: {v}" for k, v in value.items()])
        elif isinstance(value, list):
            text = ", ".join(str(item) for item in value)
        else:
            text = str(value or "")
        chunks.append(
            {
                "bucket_id": _infer_bucket(f"{key} {text}"),
                "text": f"{key}: {text}",
                "source_type": "ltm_profile",
            }
        )
    for person in state.ltm.get("people_and_relationships", []):
        chunks.append(
            {
                "bucket_id": _infer_bucket(f"{person.get('name', '')} {person.get('style_hint', '')}"),
                "text": f"partner profile: {person.get('name', '')} {person.get('style_hint', '')}",
                "partner_id": str(person.get("person_id", "")).lower(),
                "partner_name": person.get("name", ""),
                "source_type": "ltm_partner_profile",
            }
        )
    for topic in state.ltm.get("people_topics", []):
        lower = topic.lower()
        matched_person = next((person for person in state.ltm.get("people_and_relationships", []) if person.get("name", "").lower() in lower), None)
        chunks.append(
            {
                "bucket_id": _infer_bucket(topic),
                "text": f"people topic: {topic}",
                "partner_id": str(matched_person.get("person_id", "")).lower() if matched_person else "",
                "partner_name": matched_person.get("name", "") if matched_person else "",
                "source_type": "ltm_people_topic",
            }
        )
    return chunks


def _stm_chunks(state: SessionState):
    chunks = []
    today_plans = state.stm.get("today_plans", [])
    next_days = state.stm.get("next_days_plans", [])
    reminders = state.stm.get("reminders", [])
    recent_turns = state.stm.get("recent_turns", [])
    topic_hints = state.stm.get("current_topic_hints", [])
    active_partner = state.stm.get("active_partner", {})
    for item in today_plans:
        chunks.append({"bucket_id": "plans", "text": f"today plan: {item}", "source_type": "today_plan"})
    for item in next_days:
        chunks.append({"bucket_id": "plans", "text": f"upcoming: {item}", "source_type": "next_plan"})
    for item in reminders:
        chunks.append({"bucket_id": "routine", "text": f"reminder: {item}", "source_type": "reminder"})
    for item in topic_hints:
        chunks.append({"bucket_id": _infer_bucket(item), "text": f"topic hint: {item}", "source_type": "topic_hint"})
    if active_partner:
        chunks.append(
            {
                "bucket_id": "routine",
                "text": f"active partner: {active_partner.get('name', 'Partner')} ({active_partner.get('type', 'friend_or_general')})",
                "partner_id": str(active_partner.get("person_id", "")).lower(),
                "partner_name": active_partner.get("name", ""),
                "source_type": "active_partner",
            }
        )
    for turn in recent_turns[-5:]:
        partner_name = turn.get("partner_name", "")
        partner = turn.get("partner", "")
        response = turn.get("response", "")
        chunks.append(
            {
                "bucket_id": _infer_bucket(f"{partner} {response}"),
                "text": f"recent turn partner:{partner} response:{response}",
                "partner_name": partner_name,
                "source_type": "recent_turn",
            }
        )
    return chunks


def _pb_chunks(state: SessionState):
    chunks = []
    phrases = state.pb.get("phrases", [])
    for phrase in phrases:
        chunks.append(
            {
                "bucket_id": phrase.get("bucket_id", "general"),
                "text": phrase.get("text", ""),
                "weight": float(phrase.get("weight", 1.0)),
                "tone": phrase.get("tone", "neutral"),
                "partner_id": phrase.get("partner_id", "unknown_partner"),
                "partner_name": phrase.get("partner_name", ""),
                "source_type": "phrase_bank",
            }
        )
    return chunks


def _is_schedule_query(partner_text: str) -> bool:
    text = (partner_text or "").lower()
    return any(
        phrase in text
        for phrase in ["plan", "plans", "schedule", "today", "tomorrow", "later", "check-in", "rehearsal", "movie", "medication", "what do i have"]
    )


def _active_partner_topic_bonus(active_partner_id: str, text: str) -> float:
    if not active_partner_id:
        return 0.0
    keywords = PARTNER_TOPIC_KEYWORDS.get(active_partner_id, set())
    token_set = set(tokenize(text))
    overlap = len(token_set.intersection(keywords))
    if overlap == 0:
        return 0.0
    return min(0.3, 0.08 * overlap)


def _evidence_refiner_node(pool: str, ranked_scored_chunks: List[Tuple[float, Dict]], min_score: float = 0.2):
    before = []
    refined = []
    for score, chunk in ranked_scored_chunks:
        before.append(
            {
                "bucket_id": chunk.get("bucket_id", "general"),
                "text": chunk["text"],
                "score": round(score, 4),
            }
        )
        if score <= min_score:
            continue
        refined.append(
            RetrievalEvidence(
                pool=pool,
                bucket_id=chunk.get("bucket_id", "general"),
                text=chunk["text"],
                score=round(score, 4),
            )
        )
    return before, refined


def retrieve_from_pool_node(
    state: SessionState,
    partner_text: str,
    search_order: List[str],
    chosen_buckets: List[str],
    top_k: int = 4,
) -> Tuple[List[RetrievalEvidence], List[str], List[RetrievalTraceStep]]:
    notes: List[str] = []
    evidence: List[RetrievalEvidence] = []
    trace: List[RetrievalTraceStep] = []
    query_bucket = _infer_bucket(partner_text)
    greeting_query = is_short_greeting(partner_text)
    schedule_query = _is_schedule_query(partner_text)
    active_partner = state.stm.get("active_partner", {})
    active_partner_name = str(active_partner.get("name", "")).lower()
    active_partner_id = str(active_partner.get("person_id", "unknown_partner")).lower()
    greeting_words = {"hi", "hii", "hello", "hey", "yo", "sup"}
    greeting_tokens = set(tokenize(partner_text))

    pool_builders = {"LTM": _ltm_chunks, "STM": _stm_chunks, "PB": _pb_chunks}
    for pool in search_order[:3]:
        chunks = pool_builders[pool](state)
        scored = []
        for chunk in chunks:
            text = chunk["text"]
            overlap = keyword_overlap_score(partner_text, text)
            cosine = cosine_like_score(partner_text, text)
            bucket_bonus = 0.15 if chunk.get("bucket_id") == query_bucket else 0.0
            if greeting_query and chunk.get("bucket_id") == "greeting_smalltalk":
                bucket_bonus += 0.35
            if chunk.get("bucket_id") in set(chosen_buckets):
                bucket_bonus += 0.1
            partner_bonus = 0.0
            if active_partner_name and str(chunk.get("partner_name", "")).lower() == active_partner_name:
                partner_bonus += 0.18
            if active_partner_id and str(chunk.get("partner_id", "")).lower() == active_partner_id:
                partner_bonus += 0.18
            partner_bonus += _active_partner_topic_bonus(active_partner_id, text)
            phrase_weight = float(chunk.get("weight", 1.0))
            score = (0.55 * overlap + 0.35 * cosine + bucket_bonus + partner_bonus) * phrase_weight
            if greeting_query and pool == "STM" and chunk.get("bucket_id") in {"plans", "routine"}:
                score -= 0.35
            if greeting_query and pool == "PB":
                chunk_tokens = set(tokenize(text))
                if chunk_tokens.intersection(greeting_words):
                    score += 0.25
                if chunk_tokens.intersection({"movie", "therapy", "grocery", "medication", "class", "project"}):
                    score -= 0.2
            if schedule_query:
                source_type = chunk.get("source_type", "")
                if source_type in {"today_plan", "next_plan", "reminder"}:
                    score += 0.22
                elif source_type == "topic_hint":
                    score += 0.08
                elif source_type == "recent_turn":
                    score -= 0.28
                if "check-in" in partner_text.lower() and "check-in" in text.lower():
                    score += 0.22
                if "rehearsal" in partner_text.lower() and "rehearsal" in text.lower():
                    score += 0.22
                if "movie" in partner_text.lower() and "movie" in text.lower():
                    score += 0.22
            scored.append((score, chunk))
        top = pick_top(scored, key_fn=lambda item: item[0], k=top_k)
        before_refine, refined = _evidence_refiner_node(pool=pool, ranked_scored_chunks=top, min_score=0.2)
        if refined:
            evidence.extend(refined[:2])
            notes.append(f"{pool}: coverage partial/sufficient with {len(refined)} hits")
            strongest = max(item.score for item in refined)
            coverage = "sufficient_stop" if strongest > 0.5 and len(evidence) >= 2 else "partial_continue"
            trace.append(
                RetrievalTraceStep(
                    pool=pool,
                    attempted=True,
                    top_candidates_before_refine=before_refine,
                    refined_evidence=[{"bucket_id": item.bucket_id, "text": item.text, "score": item.score} for item in refined],
                    coverage=coverage,
                )
            )
            if strongest > 0.5 and len(evidence) >= 2:
                break
        else:
            notes.append(f"{pool}: insufficient evidence; fallback")
            trace.append(
                RetrievalTraceStep(
                    pool=pool,
                    attempted=True,
                    top_candidates_before_refine=before_refine,
                    refined_evidence=[],
                    coverage="insufficient_fallback",
                )
            )

    return evidence, notes, trace


def groundedness_guard_node(evidence: List[RetrievalEvidence]) -> Dict[str, str]:
    if not evidence:
        return {
            "mode": "safe",
            "instruction": "Evidence unavailable. Use neutral options and ask a clarification.",
        }
    merged = " | ".join([ev.text for ev in evidence[:6]])
    return {"mode": "grounded", "instruction": merged}


def _tone_prefix(face_signals: FaceSignals) -> str:
    if not face_signals:
        return ""
    hand_gesture = face_signals.get("hand_gesture_label", "")
    hand_score = float(face_signals.get("hand_gesture_score", 0.0) or 0.0)
    if hand_score > 0.55:
        if hand_gesture in {"Thumb_Down", "Closed_Fist"}:
            return "I do not think so"
        if hand_gesture in {"Thumb_Up", "Victory", "ILoveYou"}:
            return "Yes, that seems right"
        if hand_gesture in {"Open_Palm", "Pointing_Up"}:
            return "Could you clarify a little"
    if face_signals.get("shake_score", 0.0) > 0.55:
        return "I do not think so"
    if face_signals.get("nod_score", 0.0) > 0.55:
        return "Yes, that seems right"
    if face_signals.get("negative_prob", 0.0) > 0.5:
        return "I am not comfortable with that"
    if face_signals.get("confused_prob", 0.0) > 0.45:
        return "Could you clarify a little"
    if face_signals.get("smile_prob", 0.0) > 0.55:
        return "Sure, happy to chat"
    return ""


def _is_binary_prompt(partner_text: str) -> bool:
    text = (partner_text or "").strip().lower()
    if not text:
        return False
    starts = ("are ", "is ", "do ", "did ", "can ", "could ", "would ", "will ", "should ", "want ", "you ", "u ")
    binary_words = {"yes", "no", "okay", "available", "free", "ready", "want", "go", "come", "confirm", "remember", "know", "recall"}
    token_set = set(tokenize(text))
    recall_query = bool(token_set.intersection({"remember", "know", "recall"}))
    return text.endswith("?") and (text.startswith(starts) or bool(token_set.intersection(binary_words)) or recall_query)


def _is_plan_change_prompt(partner_text: str) -> bool:
    text = (partner_text or "").strip().lower()
    if not text:
        return False
    has_schedule_marker = any(
        phrase in text
        for phrase in ["movie", "plan", "plans", "7 pm", "7:00", "at ", "time", "schedule"]
    )
    has_explicit_change_marker = any(
        phrase in text
        for phrase in ["tomorrow", "instead", "some other time", "sometime else", "another time", "reschedule", "move it", "change it", "let us go tomorrow", "lets go tomorrow"]
    )
    return has_schedule_marker and has_explicit_change_marker


def _is_schedule_summary_query(partner_text: str) -> bool:
    text = (partner_text or "").strip().lower()
    if not text:
        return False
    return any(
        phrase in text
        for phrase in [
            "what do i have later today",
            "what do i have today",
            "what do i have left today",
            "what's left today",
            "whats left today",
            "later today",
            "today schedule",
            "my plans today",
        ]
    )


def _is_medication_status_query(partner_text: str) -> bool:
    text = (partner_text or "").strip().lower()
    if not text:
        return False
    tokens = set(tokenize(text))
    has_medication_word = bool(tokens.intersection(MEDICATION_QUERY_MARKERS)) or "medication" in text
    if not has_medication_word:
        return False
    return any(
        phrase in text
        for phrase in [
            "did you already take",
            "did you take",
            "have you taken",
            "did you already have",
            "already take",
            "already took",
            "take your medication",
            "take your medicine",
            "taken your medication",
            "taken your medicine",
        ]
    )


def _extract_medication_schedule_hint(state: SessionState, evidence: List[RetrievalEvidence]) -> str:
    sources = [ev.text for ev in evidence[:4]] + [str(item) for item in state.stm.get("today_plans", [])] + [str(item) for item in state.stm.get("reminders", [])]
    for line in sources:
        lower = line.lower()
        if any(word in lower for word in ["medication", "medicine", "meds", "pill"]):
            match = re.search(r"\b\d{1,2}(?::\d{2})?\s*(am|pm)\b", line, flags=re.IGNORECASE)
            if match:
                return match.group(0).upper()
    return ""


def _generate_medication_status_options(state: SessionState, evidence: List[RetrievalEvidence], face_signals: FaceSignals) -> List[str]:
    schedule_hint = _extract_medication_schedule_hint(state, evidence)
    schedule_clause = f" It is scheduled for {schedule_hint}." if schedule_hint else ""
    careful = f"I am not sure if I already took it.{schedule_clause}"
    clarify = "Please check with me or remind me again."
    memory_only = f"I remember medication is planned{f' for {schedule_hint}' if schedule_hint else ''}, but I cannot confirm it is done."
    if (face_signals or {}).get("confused_prob", 0.0) > 0.45:
        return [careful, clarify, memory_only]
    if (face_signals or {}).get("shake_score", 0.0) > 0.55 or (face_signals or {}).get("negative_prob", 0.0) > 0.5:
        return [careful, memory_only, clarify]
    return [careful, memory_only, clarify]


def _extract_memory_time_token(evidence_lines: List[str]) -> str:
    for line in evidence_lines:
        match = re.search(r"\b\d{1,2}(:\d{2})?\s*(am|pm)\b", line, flags=re.IGNORECASE)
        if match:
            return match.group(0).upper()
    return ""


def _compact_time_token(time_token: str) -> str:
    token = (time_token or "").strip().upper()
    if not token:
        return ""
    match = re.match(r"(\d{1,2})(?::(\d{2}))?\s*(AM|PM)", token)
    if not match:
        return token
    hour = int(match.group(1))
    minute = match.group(2) or "00"
    ampm = match.group(3)
    if minute == "00":
        return f"{hour} {ampm}"
    return f"{hour}:{minute} {ampm}"


def _generate_plan_change_options(evidence: List[RetrievalEvidence]) -> List[str]:
    evidence_lines = [ev.text for ev in evidence[:3]]
    time_token = _compact_time_token(_extract_memory_time_token(evidence_lines))
    keep_time = f"No, let us go at {time_token}." if time_token else "No, let us keep the original time."
    return [
        keep_time,
        "No, let us go sometime else.",
        "Yes, it is fine with me.",
    ]


def _schedule_sort_key(item: str) -> Tuple[int, str]:
    match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(AM|PM)\b", item, flags=re.IGNORECASE)
    if not match:
        return (9999, item.lower())
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    ampm = match.group(3).upper()
    if ampm == "PM" and hour != 12:
        hour += 12
    if ampm == "AM" and hour == 12:
        hour = 0
    return (hour * 60 + minute, item.lower())


def _compress_plan_item(plan: str) -> str:
    text = re.sub(r"^today plan:\s*", "", (plan or "").strip(), flags=re.IGNORECASE)
    text = re.sub(r"\bMovie with Omer at (\d{1,2}:\d{2}\s*[AP]M) at Regal North screen 6\b", r"movie with Omer at \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bCSE 635 project check-in\b", "CSE 635 check-in", text, flags=re.IGNORECASE)
    return text


def _generate_schedule_summary_options(state: SessionState, face_signals: FaceSignals) -> List[str]:
    plans = sorted([_compress_plan_item(item) for item in state.stm.get("today_plans", [])], key=_schedule_sort_key)
    if not plans:
        return [
            "You do not have a stored plan for later today yet.",
            "I do not see any later-today plans in memory right now.",
            "Do you want me to check tomorrow instead?",
        ]
    full_plan = "You have " + ", ".join(plans[:-1]) + f", and {plans[-1]}." if len(plans) > 1 else f"You have {plans[0]}."
    concise_plan = "Later today: " + "; ".join(plans[:3]) + ("." if len(plans) <= 3 else f"; and {len(plans) - 3} more.")
    next_step = f"The next few things are {plans[0]}" if len(plans) == 1 else f"The next few things are {plans[0]}, then {plans[1]}."
    if (face_signals or {}).get("confused_prob", 0.0) > 0.45:
        return [full_plan, concise_plan, "Do you want the full list again or just the next thing?"]
    return [full_plan, concise_plan, next_step]


def _binary_memory_stance(evidence: List[RetrievalEvidence], partner_text: str) -> str:
    query_tokens = set(tokenize(partner_text))
    if not query_tokens:
        return "unknown"
    for item in evidence:
        line = item.text.lower()
        overlap = len(query_tokens.intersection(set(tokenize(line))))
        if overlap < 2:
            continue
        if any(token in line for token in [" no ", " can't", " cannot", " not ", "later"]):
            return "decline"
        if any(token in line for token in [" yes", " works for me", "sounds good", "still works"]):
            return "agree"
    return "unknown"


def candidate_generator_node(
    state: SessionState,
    partner_text: str,
    label: str,
    evidence: List[RetrievalEvidence],
    guardrails: Dict[str, str],
    face_signals: FaceSignals,
    pb_exemplars: List[str],
    partner_style_hint: str,
) -> List[str]:
    if is_short_greeting(partner_text):
        if (face_signals or {}).get("smile_prob", 0.0) > 0.55:
            return ["Hey! Good to see you.", "Hi! How are you?", "Hi!"]
        return ["Hi, good to see you.", "Hey! How are you?", "Hi!"]
    style = state.ltm.get("communication_style", {})
    brevity = style.get("preferred_length", "short")
    emoji_light = style.get("emoji_level", "low")
    tone_boost = _tone_prefix(face_signals)

    if guardrails["mode"] == "safe":
        base = [
            "Can you tell me a bit more so I answer clearly?",
            "I want to respond well. Could you repeat that in another way?",
            "I am not sure yet. Give me one more detail please.",
        ]
        return base

    if _is_schedule_summary_query(partner_text):
        return _generate_schedule_summary_options(state=state, face_signals=face_signals)

    if _is_medication_status_query(partner_text):
        return _generate_medication_status_options(state=state, evidence=evidence, face_signals=face_signals)

    if _is_plan_change_prompt(partner_text):
        return _generate_plan_change_options(evidence)

    if _is_binary_prompt(partner_text):
        return _generate_binary_options(partner_text=partner_text, evidence=evidence, face_signals=face_signals)

    evidence_lines = [ev.text for ev in evidence[:3]]
    anchor = _clean_fact_fragment(evidence_lines[0].split(":", 1)[-1].strip()) if evidence_lines else ""
    second_anchor = _clean_fact_fragment(evidence_lines[1].split(":", 1)[-1].strip()) if len(evidence_lines) > 1 else anchor
    memory_detail = _extract_memory_detail(evidence_lines)
    face_detected = bool((face_signals or {}).get("face_detected", False))
    if not face_detected:
        base_detail = memory_detail or anchor or second_anchor or "that plan"
        fallback_anchor = second_anchor or anchor or base_detail
        return [
            f"From what I remember, {base_detail}. That sounds good to me.",
            f"{fallback_anchor}. I can go with that plan.",
            f"I remember {base_detail}. Do you want to confirm one detail first?",
        ]
    memory_lead = "From what I remember, " if label in {"Contextual", "Personal"} and memory_detail else ""
    emoji = " 🙂" if emoji_light == "low" and (face_signals or {}).get("smile_prob", 0) > 0.7 else ""
    exemplar_fragment = pb_exemplars[0] if pb_exemplars else "I want to keep it simple."
    face_summary_for_prompt = face_summary(face_signals)
    template_payload = (
        f"refined_evidence={guardrails.get('instruction', '')}; "
        f"ltm_style={style}; "
        f"pb_exemplar={exemplar_fragment}; "
        f"partner_style_hint={partner_style_hint}; "
        f"face_signals={face_summary_for_prompt}; "
        f"runtime_clock={state.stm.get('runtime_clock', '')}"
    )

    style_prefix = ""
    if tone_boost:
        style_prefix = f"{tone_boost}, "
    tone_mode = _tone_mode(face_signals)
    if tone_mode == "decline":
        reason_detail = memory_detail or anchor or second_anchor or "that"
        short_no = "No." if not reason_detail else f"No, I do not want to right now."
        option_1 = short_no
        option_2 = f"{style_prefix}{memory_lead}{reason_detail}. I cannot do that right now."
        option_3 = f"No, let us go some other time. {second_anchor or reason_detail}."
    elif tone_mode == "clarify":
        option_1 = f"{style_prefix}{memory_lead}{memory_detail or anchor}. Can you confirm one detail for me?"
        option_2 = f"{second_anchor}. I want to answer carefully."
        option_3 = f"{memory_detail or anchor}. Could you repeat the key part?"
    elif tone_mode == "agree":
        option_1 = f"{style_prefix}{memory_lead}{memory_detail or anchor}. That works for me."
        option_2 = f"{second_anchor}. Yes, I remember this detail{emoji}."
        option_3 = f"{memory_detail or anchor}. I am good with that plan."
    elif tone_mode == "warm":
        option_1 = f"{style_prefix}{memory_lead}{memory_detail or anchor}. That sounds nice."
        option_2 = f"{second_anchor}. Yes, I remember this detail{emoji}."
        option_3 = f"{memory_detail or anchor}. I am happy with that."
    else:
        option_1 = f"{style_prefix}{memory_lead}{memory_detail or anchor}. That works for me."
        option_2 = f"{second_anchor}. I remember this detail, so yes{emoji}."
        option_3 = f"{memory_detail or anchor}. Works for me."

    if brevity == "short":
        option_1 = option_1[:150]
        option_2 = option_2[:140]
        option_3 = option_3[:155]

    return [option_1, option_2, option_3, f"template_trace::{template_payload}"]


def _generate_binary_options(partner_text: str, evidence: List[RetrievalEvidence], face_signals: FaceSignals) -> List[str]:
    memory_detail = _extract_memory_detail([ev.text for ev in evidence[:3]])
    stance = _binary_memory_stance(evidence=evidence, partner_text=partner_text)
    agree = f"Yes, that works for me{'.' if not memory_detail else f'. I remember {memory_detail}.'}"
    decline_polite = "No, I do not want to right now."
    decline_firm = "No."
    decline_reschedule = f"No, let us do it some other time{'.' if not memory_detail else f'. I remember {memory_detail}.'}"
    no_memory = "No, I do not remember right now."
    memory_based = (
        f"I think {memory_detail}, but I am not fully sure."
        if memory_detail
        else "I think there was a plan, but I am not fully sure."
    )
    clarify = "I am not sure yet. Can you give me one more detail?"
    if stance == "agree":
        agree = f"Yes, I am okay with it. I remember {memory_detail or 'that plan'}."
    if stance == "decline":
        decline_polite = "No thanks, I will pass for now."
        decline_reschedule = "No, not now. Maybe some other time."
    nod = (face_signals or {}).get("nod_score", 0.0)
    shake = (face_signals or {}).get("shake_score", 0.0)
    negative = (face_signals or {}).get("negative_prob", 0.0)
    confused = (face_signals or {}).get("confused_prob", 0.0)
    hand_gesture = (face_signals or {}).get("hand_gesture_label", "")
    hand_score = float((face_signals or {}).get("hand_gesture_score", 0.0) or 0.0)
    if hand_score > 0.55:
        if hand_gesture in {"Thumb_Down", "Closed_Fist"}:
            shake = max(shake, 0.8)
            negative = max(negative, 0.65)
        elif hand_gesture in {"Thumb_Up", "Victory", "ILoveYou"}:
            nod = max(nod, 0.8)
        elif hand_gesture in {"Open_Palm", "Pointing_Up"}:
            confused = max(confused, 0.6)
    if shake > 0.55 or negative > 0.5:
        return [decline_firm, no_memory, memory_based]
    if nod > 0.55:
        return [agree, decline_polite, clarify]
    if confused > 0.45:
        return [clarify, agree, decline_polite]
    return [agree, decline_polite, clarify]


def postprocess_candidate_text(text: str) -> str:
    cleaned = text.strip()
    bad_prefixes = [
        "casual and playful:",
        "warm and appreciative:",
        "neutral and polite:",
        "i would say this in my style:",
        "from what i prefer,",
        "from today,",
        "my take is:",
    ]
    lower = cleaned.lower()
    for prefix in bad_prefixes:
        if lower.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            lower = cleaned.lower()
    banned_fragments = [
        "i would say this in my style",
        "my take is",
        "response:",
        "casual and playful:",
        "warm and appreciative:",
        "neutral and polite:",
        "from today,",
        "thanks for asking",
    ]
    for fragment in banned_fragments:
        cleaned = re.sub(re.escape(fragment), "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\brecent turn partner:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bPartner asked\b", "You asked", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bI remember\s+Partner is [^.?!]+[.?!]?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bPartner is [^.?!]+[.?!]?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bpartner profile:\s*[^.?!]+[.?!]?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bI want to say\s*[.:,-]*\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bQuick note\s*[,:-]*\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^[\s\.,;:!\?-]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Keep first one or two short sentences to avoid overflow/noisy carryover.
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    cleaned = " ".join(parts[:2]).strip()
    cleaned = cleaned.replace("..", ".").strip(" -")
    return cleaned


def enforce_option_quality(options: List[str]) -> List[str]:
    processed = [postprocess_candidate_text(option) for option in options]
    unique = []
    seen = set()
    for text in processed:
        key = text.lower()
        if key not in seen and text:
            unique.append(text)
            seen.add(key)
    while len(unique) < 3:
        fallback = [
            "Yes, that works for me.",
            "Sounds good, I can do that.",
            "I am okay with that plan.",
        ][len(unique)]
        unique.append(fallback)
    # Keep variants meaningful and concise.
    unique[0] = unique[0].rstrip(".!?") + "."
    unique[1] = unique[1].rstrip(".!?") + "."
    unique[2] = re.sub(r"\bThat works for me\b", "Works for me", unique[2], flags=re.IGNORECASE)
    unique = [re.sub(r"\.\s*\.", ".", item).strip() for item in unique]
    if any(is_short_greeting(item) for item in options):
        unique = _sanitize_greeting_options(unique)
    return unique[:3]


def enforce_memory_signature(options: List[str], evidence: List[RetrievalEvidence], label: str) -> List[str]:
    if label not in {"Contextual", "Personal"} or not evidence:
        return options
    detail = _extract_memory_detail([item.text for item in evidence[:3]])
    if not detail:
        return options
    if detail.lower().startswith("partner is ") or detail.lower().startswith("partner profile:"):
        return options
    revised = []
    for idx, option in enumerate(options):
        lowered = option.lower()
        if lowered.startswith(("no", "can you", "could you", "i am not sure")):
            revised.append(option)
        elif idx < 2 and detail.lower() not in lowered:
            revised.append(f"I remember {detail}, {option}")
        else:
            revised.append(option)
    return revised


def _clean_fact_fragment(text: str) -> str:
    raw = text.strip()
    lower = raw.lower()
    if "recent turn partner:" in lower and "response:" in lower:
        raw = raw[lower.index("response:") + len("response:") :].strip()
        lower = raw.lower()
    if "response:" in lower:
        raw = raw[lower.index("response:") + len("response:") :].strip()
    elif "partner:" in lower:
        raw = raw[lower.index("partner:") + len("partner:") :].strip()
    cleaned = postprocess_candidate_text(raw)
    if not cleaned:
        return "That plan is okay"
    # Keep concise factual anchor.
    return cleaned[:120].strip(" ,.;")


def _extract_memory_detail(evidence_lines: List[str]) -> str:
    for line in evidence_lines:
        lower = line.lower()
        match = re.search(r"\b\d{1,2}(:\d{2})?\s*(am|pm)\b", lower)
        if match:
            token = match.group(0).upper()
            if "movie" in lower:
                return f"the movie is at {token}"
            if "therapy" in lower:
                return f"therapy is at {token}"
            if "meeting" in lower or "check-in" in lower:
                return f"the check-in is at {token}"
            return f"that is planned for {token}"
    for line in evidence_lines:
        cleaned = _clean_fact_fragment(line)
        if cleaned:
            return cleaned
    return ""


def _sanitize_greeting_options(options: List[str]) -> List[str]:
    banned = {"movie", "therapy", "grocery", "medication", "class", "project", "dentist"}
    clean = []
    defaults = ["Hi, good to see you.", "Hey! How are you?", "Hello!"]
    for idx, option in enumerate(options[:3]):
        tokens = set(tokenize(option))
        if tokens.intersection(banned) or re.search(r"\bpartner is\b|\bpartner profile\b", option, flags=re.IGNORECASE):
            clean.append(defaults[idx])
        else:
            clean.append(option if option else defaults[idx])
    while len(clean) < 3:
        clean.append(defaults[len(clean)])
    return clean[:3]


def enforce_grounded_personal_facts(options: List[str], evidence: List[RetrievalEvidence]) -> List[str]:
    evidence_text = " ".join(item.text.lower() for item in evidence)
    protected_tokens = {
        "mom",
        "dad",
        "brother",
        "sister",
        "movie",
        "therapy",
        "medication",
        "dentist",
        "grocery",
        "class",
    }
    fallback_variants = [
        "I want to answer carefully. Could you confirm one detail so I stay accurate?",
        "I might be missing context. Can you confirm the key personal detail first?",
        "Before I answer, can you verify that personal detail for me?",
    ]
    sanitized = []
    for idx, option in enumerate(options):
        token_set = set(re.findall(r"[a-zA-Z']+", option.lower()))
        missing = [token for token in protected_tokens if token in token_set and token not in evidence_text]
        if missing:
            sanitized.append(fallback_variants[idx % len(fallback_variants)])
        else:
            sanitized.append(option)
    return sanitized


def enforce_do_not_say(options: List[str], do_not_say: List[str]) -> List[str]:
    taboo_fragments = [phrase.lower().strip() for phrase in do_not_say if phrase.strip()]
    rewrite_variants = [
        "I cannot phrase it that way. Let me answer with a safer wording.",
        "That wording is not appropriate for me. I can share a safer response.",
        "I should avoid that phrase. Here is a safer way to say it.",
    ]
    blocked = []
    for idx, option in enumerate(options):
        lower = option.lower()
        if any(fragment in lower for fragment in taboo_fragments):
            blocked.append(rewrite_variants[idx % len(rewrite_variants)])
        else:
            blocked.append(option)
    return blocked


def speak_planner_node(state: SessionState, face_signals: FaceSignals) -> Dict[str, List[str]]:
    smile = (face_signals or {}).get("smile_prob", 0.0)
    upbeat_suffix = " 🙂" if smile > 0.6 else ""
    grouped = {
        "Today": [],
        "Next few days": [],
        "Reminders/tasks": [],
        "People/topics": [],
    }
    now = datetime.now(timezone.utc)
    for item in state.stm.get("today_plans", []):
        grouped["Today"].append(f"I want to share this for today: {item}.{upbeat_suffix}")
        grouped["Today"].append(f"Can you help me plan around: {item}?")
        lowered = item.lower()
        if "movie" in lowered:
            grouped["Today"].append("I have a movie at 7 PM. Are you free before 6:30?")
            grouped["Today"].append("I should leave soon for the movie. Can we wrap up in 15 minutes?")
        if "at " in lowered and ("pm" in lowered or "am" in lowered):
            grouped["Today"].append(f"I need timing help for this: {item}.")

        # Urgency bias: if event is within 60 mins, add urgent prompts.
        match = re.search(r"at (\d{1,2}):(\d{2})\s*(AM|PM)", item, re.IGNORECASE)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            ampm = match.group(3).upper()
            if ampm == "PM" and hour != 12:
                hour += 12
            if ampm == "AM" and hour == 12:
                hour = 0
            event_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            delta = (event_time - now).total_seconds() / 60.0
            if 0 <= delta <= 60:
                grouped["Today"].insert(0, f"I need to leave soon for this: {item}.")
                grouped["Today"].insert(1, f"Can we quickly finalize plans before {hour:02d}:{minute:02d}?")
    for item in state.stm.get("next_days_plans", []):
        grouped["Next few days"].append(f"This is coming up soon: {item}.")
        grouped["Next few days"].append(f"Are we ready for {item}?")
    for item in state.stm.get("reminders", []):
        grouped["Reminders/tasks"].append(f"Reminder for me: {item}.")
    people_topics = state.ltm.get("people_topics", [])
    for topic in people_topics:
        grouped["People/topics"].append(f"I want to ask about {topic}.")

    phrase_samples = state.pb.get("phrases", [])[:10]
    for phrase in phrase_samples:
        bucket = phrase.get("bucket_id", "general")
        text = phrase.get("text", "")
        if bucket in {"plans", "routine"} and len(grouped["Today"]) < 8:
            grouped["Today"].append(text)
        elif bucket.startswith("clarify") and len(grouped["Reminders/tasks"]) < 8:
            grouped["Reminders/tasks"].append(text)
        elif bucket in {"family", "food"} and len(grouped["People/topics"]) < 8:
            grouped["People/topics"].append(text)
        elif len(grouped["Next few days"]) < 8:
            grouped["Next few days"].append(text)

    return grouped


def face_summary(face_signals: FaceSignals) -> str:
    if not face_signals or not face_signals.get("face_detected", False):
        if face_signals and face_signals.get("hand_detected") and face_signals.get("hand_gesture_label"):
            return f"hand gesture: {face_signals.get('hand_gesture_label')}"
        return "Face not detected"
    parts = []
    if face_signals.get("shake_score", 0.0) > 0.55:
        parts.append("head shake high")
    elif face_signals.get("nod_score", 0.0) > 0.55:
        parts.append("head nod high")
    elif face_signals.get("negative_prob", 0.0) > 0.5:
        parts.append("negative affect high")
    elif face_signals.get("smile_score", face_signals.get("smile_prob", 0.0)) > 0.55:
        parts.append("smile high")
    elif face_signals.get("confused_prob", 0.0) > 0.45:
        parts.append("confused high")
    else:
        parts.append("neutral/low signal")
    if face_signals.get("hand_detected") and face_signals.get("hand_gesture_label"):
        parts.append(f"hand {face_signals.get('hand_gesture_label')}")
    return " + ".join(parts)


def coarse_face_tag(face_signals: FaceSignals) -> str:
    if not face_signals or not face_signals.get("face_detected", False):
        if face_signals and face_signals.get("hand_gesture_label") in {"Thumb_Up", "Victory", "ILoveYou"}:
            return "agree_gesture"
        if face_signals and face_signals.get("hand_gesture_label") in {"Thumb_Down", "Closed_Fist"}:
            return "decline_gesture"
        return "none"
    return "smile" if face_signals.get("smile_score", face_signals.get("smile_prob", 0.0)) > 0.55 else "not_smile"


def _tone_mode(face_signals: FaceSignals) -> str:
    if not face_signals:
        return "neutral"
    hand_gesture = face_signals.get("hand_gesture_label", "")
    hand_score = float(face_signals.get("hand_gesture_score", 0.0) or 0.0)
    if hand_score > 0.55:
        if hand_gesture in {"Thumb_Down", "Closed_Fist"}:
            return "decline"
        if hand_gesture in {"Thumb_Up", "Victory", "ILoveYou"}:
            return "agree"
        if hand_gesture in {"Open_Palm", "Pointing_Up"}:
            return "clarify"
    if face_signals.get("shake_score", 0.0) > 0.55 or face_signals.get("negative_prob", 0.0) > 0.5:
        return "decline"
    if face_signals.get("nod_score", 0.0) > 0.55:
        return "agree"
    if face_signals.get("confused_prob", 0.0) > 0.45:
        return "clarify"
    if face_signals.get("smile_prob", 0.0) > 0.55:
        return "warm"
    return "neutral"


def has_strong_tone_signal(face_signals: FaceSignals) -> bool:
    return _tone_mode(face_signals) in {"agree", "decline", "clarify"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
