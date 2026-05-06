"""
Parallel Prep node.

Fires intent classification and lightweight memory extraction concurrently
behind a thread pool, halving wall-clock latency. A regex pre-filter
short-circuits the memory LLM when no schedule/fact trigger is present in
the partner utterance, so identity/opinion turns pay zero LLM cost for the
memory branch.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Tuple


# Regex pre-filter: only fire the memory extractor when the partner
# utterance plausibly contains a fact, schedule item, or directive.
_MEMORY_TRIGGER_RE = re.compile(
    r"\b("
    r"tomorrow|tonight|today|sunday|monday|tuesday|wednesday|thursday|friday|saturday"
    r"|next\s+\w+|in\s+\d+\s+(min|minute|minutes|hour|hours|day|days)"
    r"|at\s+\d{1,2}(:\d{2})?\s*(am|pm)?"
    r"|by\s+\d{1,2}(:\d{2})?\s*(am|pm)?"
    r"|before\s+\d{1,2}(:\d{2})?\s*(am|pm)?"
    r"|remind(er)?|don'?t\s+forget|remember\s+to|please\s+remember"
    r"|appointment|meeting|class|movie|therapy|medication|prescription"
    r"|cricket|lab|grocery|dentist|exam"
    r")\b",
    re.IGNORECASE,
)


# Cheap rule-based intent fallback (used when LLM is disabled or fails).
def _rule_intent(text: str) -> str:
    lower = (text or "").lower().strip()
    if not lower:
        return "Open-domain"
    if any(p in lower for p in ["?"]) and any(
        m in lower for m in [
            "what is", "what's", "who is", "who's", "define", "explain",
            "how does", "why is", "when did", "where is",
        ]
    ):
        return "Open-domain"
    if any(
        m in lower for m in [
            "your", "you like", "favorite", "favourite", "prefer", "feel about",
            "what do you", "how are you", "are you",
        ]
    ):
        return "Personal"
    return "Contextual"


def _rule_memory_extract(text: str) -> Dict[str, Any]:
    """Best-effort rule-based memory extraction so we never depend on the LLM
    for the memory branch when running in offline / fallback mode."""
    if not _MEMORY_TRIGGER_RE.search(text or ""):
        return {"has_memory": False, "bucket": "", "value": ""}
    lower = text.lower()
    if any(d in lower for d in ["tomorrow", "next ", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]):
        bucket = "next_days_plans"
    elif any(d in lower for d in ["remind", "don't forget", "dont forget", "remember to"]):
        bucket = "reminders"
    elif any(d in lower for d in ["today", "tonight", "at ", "movie", "therapy", "meeting", "class", "lab", "cricket"]):
        bucket = "today_plans"
    else:
        bucket = "current_topic_hints"
    return {"has_memory": True, "bucket": bucket, "value": text.strip()}


def run_parallel_prep(
    partner_text: str,
    *,
    llm_intent_fn=None,   # callable: (text) -> Tuple[Optional[str], str_model, str_err]
    timeout_sec: float = 4.0,
) -> Dict[str, Any]:
    """
    Run intent classification and memory extraction concurrently.

    `llm_intent_fn` is optional. When provided, the LLM intent classifier
    runs in parallel with the rule-based memory extractor. When the LLM is
    disabled / unavailable, both branches use rule-based fallbacks.

    The memory branch always uses the regex pre-filter; if no memory trigger
    is present, the memory call is skipped entirely (zero work).

    Returns:
        {
          "intent": str,
          "intent_source": "llm" | "rule",
          "memory": {has_memory, bucket, value},
          "intent_error": str,
          "intent_model": str,
          "node_trace": List[str],
        }
    """
    node_trace = ["ParallelPrepNode"]
    text = partner_text or ""

    has_memory_trigger = bool(_MEMORY_TRIGGER_RE.search(text))
    if not has_memory_trigger:
        node_trace.append("MemoryPreFilter:skip(no_trigger)")

    intent_value = None
    intent_model = ""
    intent_error = ""
    intent_source = "rule"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {}

        def _intent_branch():
            if llm_intent_fn is not None:
                try:
                    label, model, err = llm_intent_fn(text)
                    if label in {"Personal", "Contextual", "Open-domain"}:
                        return label, model, "", "llm"
                    return _rule_intent(text), model, err or "router_invalid", "rule"
                except Exception as exc:
                    return _rule_intent(text), "", str(exc)[:120], "rule"
            return _rule_intent(text), "", "llm_disabled", "rule"

        def _memory_branch():
            if has_memory_trigger:
                return _rule_memory_extract(text)
            return {"has_memory": False, "bucket": "", "value": ""}

        futures["intent"] = pool.submit(_intent_branch)
        futures["memory"] = pool.submit(_memory_branch)

        try:
            intent_value, intent_model, intent_error, intent_source = futures["intent"].result(timeout=timeout_sec)
        except Exception as exc:
            intent_value = _rule_intent(text)
            intent_error = f"intent_timeout:{type(exc).__name__}"
            intent_source = "rule"
        try:
            memory = futures["memory"].result(timeout=timeout_sec)
        except Exception:
            memory = {"has_memory": False, "bucket": "", "value": ""}

    return {
        "intent": intent_value or _rule_intent(text),
        "intent_source": intent_source,
        "intent_model": intent_model,
        "intent_error": intent_error,
        "memory": memory,
        "node_trace": node_trace,
    }
