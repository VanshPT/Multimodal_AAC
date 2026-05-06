from datetime import datetime, timezone
import math
import re
from typing import Dict, List, Optional, Tuple

from home.aac.types import FaceSignals, RetrievalEvidence, RetrievalTraceStep, SessionState
from home.aac.utils import cosine_like_score, keyword_overlap_score, pick_top, tokenize


# Exclusive keyword priors — no overlap between buckets, word-boundary
# matched to avoid substring collisions ("am" inside "name", etc.).
# Used by the new bucket scorer in `score_buckets`.
BUCKET_KEYWORDS_EXCLUSIVE = {
    "greeting_smalltalk": {"hi", "hii", "hello", "hey", "yo", "sup", "hiya"},
    "family": {"mom", "dad", "mum", "father", "mother", "sister", "brother",
               "family", "parents", "grandma", "grandpa", "cousin", "aunt", "uncle"},
    "food": {"food", "lunch", "dinner", "breakfast", "eat", "snack", "coffee",
             "tea", "drink", "meal"},
    "plans": {"movie", "event", "tonight", "weekend"},
    "routine": {"routine", "morning", "evening", "daily", "usually", "every day"},
    "medical": {"pain", "doctor", "med", "medication", "therapy", "tired",
                "appointment", "dentist", "prescription", "clinic", "hospital"},
    "scheduling": {"meeting", "schedule", "calendar", "appointment", "lab",
                   "class", "check-in", "deadline", "by", "before", "tomorrow"},
    "work": {"slides", "project", "report", "deadline", "submit", "draft",
             "code", "exam", "lecture", "professor", "assignment"},
    "social": {"friend", "buddy", "party", "hang out", "chat", "movie night"},
    "open_domain": {"weather", "news", "what is", "who is", "define", "explain"},
    "decline_polite": {"no", "cannot", "later", "not now"},
    "agree_casual": {"yes", "sure", "okay", "sounds good"},
    "clarify_calm": {"clarify", "repeat", "explain", "what do you mean"},
}


# Backwards-compatible alias used elsewhere in the codebase.
BUCKET_KEYWORDS = {k: set(v) for k, v in BUCKET_KEYWORDS_EXCLUSIVE.items()}


# Generic time-of-day mention -> mild nudge towards the scheduling bucket.
_GENERIC_TIME_NUDGE_RE = re.compile(
    r"\b(\d{1,2}(:\d{2})?\s*(am|pm)|noon|midnight|tomorrow|tonight|today|by\s+\d+)\b",
    re.IGNORECASE,
)


def _kw_hits(query_tokens: set, query_lower: str, keywords: set) -> int:
    hits = 0
    for kw in keywords:
        if " " in kw or "-" in kw:
            if re.search(r"\b" + re.escape(kw) + r"\b", query_lower):
                hits += 1
        else:
            if kw in query_tokens:
                hits += 1
    return hits


def score_buckets(
    partner_text: str,
    *,
    gaze_boosts: Optional[Dict[str, float]] = None,
    bucket_acceptance: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """Returns a {bucket -> score} dict combining:
        - Jaccard-ish overlap on exclusive keyword sets (word-boundary)
        - Generic time-of-day nudge for the scheduling bucket
        - Gaze-derived bucket boosts (Bonus #1)
        - Log-scaled acceptance prior (Bonus #3 - Bucket Priors)
    """
    qlow = (partner_text or "").lower()
    qtoks = set(tokenize(partner_text))
    scores: Dict[str, float] = {}
    for bucket, words in BUCKET_KEYWORDS_EXCLUSIVE.items():
        hits = _kw_hits(qtoks, qlow, words)
        scores[bucket] = float(hits)

    # Generic-time nudge: any concrete time mention boosts scheduling.
    if _GENERIC_TIME_NUDGE_RE.search(qlow):
        scores["scheduling"] = scores.get("scheduling", 0.0) + 0.5

    # Gaze-derived boosts
    if gaze_boosts:
        for bucket, boost in gaze_boosts.items():
            scores[bucket] = scores.get(bucket, 0.0) + float(boost)

    # Bucket Priors: log-scaled acceptance bonus.
    if bucket_acceptance:
        for bucket, count in bucket_acceptance.items():
            if count > 0:
                scores[bucket] = scores.get(bucket, 0.0) + 0.03 * math.log1p(int(count))

    return scores


def top_buckets(
    partner_text: str,
    *,
    gaze_boosts: Optional[Dict[str, float]] = None,
    bucket_acceptance: Optional[Dict[str, int]] = None,
    k: int = 3,
) -> List[str]:
    scores = score_buckets(
        partner_text,
        gaze_boosts=gaze_boosts,
        bucket_acceptance=bucket_acceptance,
    )
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [bucket for bucket, score in ranked[:k] if score > 0]


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
    neutral = float(provided_face_signals.get("neutral_prob", max(0.0, 1.0 - smile - confused)))
    strongest = max(smile, confused, neutral, negative, nod, shake)
    inferred_detected = strongest > 0.08 or (smile + confused + neutral) > 0.2
    face_detected = bool(provided_face_signals.get("face_detected", inferred_detected))
    return {
        "face_detected": face_detected,
        "smile_score": max(0.0, min(1.0, smile)),
        "smile_prob": max(0.0, min(1.0, smile)),
        "confused_prob": max(0.0, min(1.0, confused)),
        "neutral_prob": max(0.0, min(1.0, neutral)),
        "negative_prob": max(0.0, min(1.0, negative)),
        "nod_score": max(0.0, min(1.0, nod)),
        "shake_score": max(0.0, min(1.0, shake)),
    }


def router_node(partner_text: str) -> str:
    lower = partner_text.lower()
    if is_short_greeting(partner_text):
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
    cleaned = re.sub(r"[^a-zA-Z]", "", (partner_text or "").lower())
    return cleaned in {"hi", "hii", "hiii", "hello", "hey", "yo", "sup"} or (
        len(tokenize(partner_text)) <= 2 and any(token in {"hi", "hello", "hey", "yo", "sup"} for token in tokenize(partner_text))
    )


def bucket_selector_node(
    partner_text: str,
    label: str,
    active_partner_relation: str,
    face_signals: FaceSignals,
    *,
    gaze_boosts: Optional[Dict[str, float]] = None,
    bucket_acceptance: Optional[Dict[str, int]] = None,
) -> List[str]:
    selected = []
    if is_short_greeting(partner_text):
        selected.append("greeting_smalltalk")

    # NEW: prefer the priors-aware scorer that uses exclusive keyword sets
    # with word-boundary matching + generic-time nudge + gaze + acceptance.
    priors_top = top_buckets(
        partner_text,
        gaze_boosts=gaze_boosts,
        bucket_acceptance=bucket_acceptance,
        k=3,
    )
    selected.extend(priors_top)

    # Fallback to legacy single-bucket inference for stability.
    inferred = _infer_bucket(partner_text)
    if inferred != "general":
        selected.append(inferred)

    if label == "Contextual":
        selected.extend(["plans", "routine"])
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
        chunks.append({"bucket_id": "plans", "text": f"today plan: {item}"})
    for item in next_days:
        chunks.append({"bucket_id": "plans", "text": f"upcoming: {item}"})
    for item in reminders:
        chunks.append({"bucket_id": "routine", "text": f"reminder: {item}"})
    for item in topic_hints:
        chunks.append({"bucket_id": _infer_bucket(item), "text": f"topic hint: {item}"})
    if active_partner:
        chunks.append(
            {
                "bucket_id": "routine",
                "text": f"active partner: {active_partner.get('name', 'Partner')} ({active_partner.get('type', 'friend_or_general')})",
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
            }
        )
    return chunks


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
    starts = ("are ", "is ", "do ", "did ", "can ", "could ", "would ", "will ", "should ", "want ")
    binary_words = {"yes", "no", "okay", "available", "free", "ready", "want", "go", "come", "confirm"}
    token_set = set(tokenize(text))
    return text.endswith("?") and (text.startswith(starts) or bool(token_set.intersection(binary_words)))


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
        prefix = _tone_prefix(face_signals)
        if prefix:
            return [f"{prefix}! Hi.", "Hey! Good to see you.", "Hi!"]
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

    if _is_binary_prompt(partner_text):
        return _generate_binary_options(partner_text=partner_text, evidence=evidence, face_signals=face_signals)

    evidence_lines = [ev.text for ev in evidence[:3]]
    anchor = _clean_fact_fragment(evidence_lines[0].split(":", 1)[-1].strip()) if evidence_lines else ""
    second_anchor = _clean_fact_fragment(evidence_lines[1].split(":", 1)[-1].strip()) if len(evidence_lines) > 1 else anchor
    memory_detail = _extract_memory_detail(evidence_lines)
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
    decline_polite = "I would rather not right now, but thanks for asking."
    decline_firm = "No, I do not want that right now."
    clarify = "I am not sure yet. Can you give me one more detail?"
    if stance == "agree":
        agree = f"Yes, I am okay with it. I remember {memory_detail or 'that plan'}."
    if stance == "decline":
        decline_polite = "No thanks, I will pass for now."
    nod = (face_signals or {}).get("nod_score", 0.0)
    shake = (face_signals or {}).get("shake_score", 0.0)
    negative = (face_signals or {}).get("negative_prob", 0.0)
    confused = (face_signals or {}).get("confused_prob", 0.0)
    if shake > 0.55 or negative > 0.5:
        return [decline_firm, decline_polite, clarify]
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
    revised = []
    for idx, option in enumerate(options):
        lowered = option.lower()
        if idx < 2 and detail.lower() not in lowered:
            revised.append(f"I remember {detail}, {option}")
        else:
            revised.append(option)
    return revised


def _clean_fact_fragment(text: str) -> str:
    raw = text.strip()
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
        if tokens.intersection(banned):
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
        return "Face not detected"
    if face_signals.get("shake_score", 0.0) > 0.55:
        return "head shake high"
    if face_signals.get("nod_score", 0.0) > 0.55:
        return "head nod high"
    if face_signals.get("negative_prob", 0.0) > 0.5:
        return "negative affect high"
    if face_signals.get("smile_score", face_signals.get("smile_prob", 0.0)) > 0.55:
        return "smile high"
    if face_signals.get("confused_prob", 0.0) > 0.45:
        return "confused high"
    return "neutral/low signal"


def coarse_face_tag(face_signals: FaceSignals) -> str:
    if not face_signals or not face_signals.get("face_detected", False):
        return "none"
    return "smile" if face_signals.get("smile_score", face_signals.get("smile_prob", 0.0)) > 0.55 else "not_smile"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
