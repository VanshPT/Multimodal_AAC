import math
import re
from collections import Counter
from typing import Iterable, List, Set


STOPWORDS: Set[str] = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "is",
    "are",
    "of",
    "in",
    "on",
    "for",
    "at",
    "it",
    "this",
    "that",
    "i",
    "you",
    "we",
    "me",
    "my",
    "our",
}


def tokenize(text: str) -> List[str]:
    raw_tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return [token for token in raw_tokens if token not in STOPWORDS]


def keyword_overlap_score(query: str, candidate: str) -> float:
    q_tokens = tokenize(query)
    c_tokens = tokenize(candidate)
    if not q_tokens or not c_tokens:
        return 0.0
    q_counts = Counter(q_tokens)
    c_counts = Counter(c_tokens)
    intersection = sum(min(q_counts[token], c_counts[token]) for token in q_counts)
    return intersection / max(len(set(q_tokens)), 1)


def cosine_like_score(query: str, candidate: str) -> float:
    q_tokens = tokenize(query)
    c_tokens = tokenize(candidate)
    if not q_tokens or not c_tokens:
        return 0.0
    q_counts = Counter(q_tokens)
    c_counts = Counter(c_tokens)
    shared = set(q_counts).intersection(c_counts)
    dot = sum(q_counts[t] * c_counts[t] for t in shared)
    q_norm = math.sqrt(sum(v * v for v in q_counts.values()))
    c_norm = math.sqrt(sum(v * v for v in c_counts.values()))
    if q_norm == 0 or c_norm == 0:
        return 0.0
    return dot / (q_norm * c_norm)


def pick_top(items: Iterable, key_fn, k: int):
    return sorted(items, key=key_fn, reverse=True)[:k]
