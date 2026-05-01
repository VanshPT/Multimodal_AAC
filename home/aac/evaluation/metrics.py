from difflib import SequenceMatcher
from statistics import mean
from typing import Dict, List

from home.aac.utils import tokenize


def groundedness_score(option_text: str, evidence_texts: List[str]) -> float:
    option_tokens = set(tokenize(option_text))
    evidence_tokens = set(tokenize(" ".join(evidence_texts)))
    if not option_tokens:
        return 1.0
    if not evidence_tokens:
        return 0.0
    overlap = len(option_tokens.intersection(evidence_tokens))
    return round(overlap / max(len(option_tokens), 1), 4)


def hallucination_rate(scores: List[float], threshold: float = 0.25) -> float:
    if not scores:
        return 0.0
    misses = len([score for score in scores if score < threshold])
    return round(misses / len(scores), 4)


def edit_distance_ratio(source: str, target: str) -> float:
    return round(1.0 - SequenceMatcher(None, source, target).ratio(), 4)


def summarize_metrics(latencies_normal: List[int], latencies_speak: List[int], grounded_scores: List[float], effort: List[Dict]):
    accepted_without_edit = [item for item in effort if item["accepted_without_edit"]]
    return {
        "latency": {
            "avg_time_to_options_ms": round(mean(latencies_normal), 2) if latencies_normal else 0,
            "avg_time_to_speak_suggestions_ms": round(mean(latencies_speak), 2) if latencies_speak else 0,
            "target_time_to_options_under_6000_ms": (round(mean(latencies_normal), 2) if latencies_normal else 0) < 6000,
        },
        "groundedness": {
            "avg_groundedness_score": round(mean(grounded_scores), 4) if grounded_scores else 0,
            "hallucination_rate": hallucination_rate(grounded_scores),
        },
        "user_effort": {
            "avg_edit_distance": round(mean([item["edit_distance"] for item in effort]), 4) if effort else 0,
            "acceptance_rate_without_edit": round(len(accepted_without_edit) / len(effort), 4) if effort else 0,
        },
    }
