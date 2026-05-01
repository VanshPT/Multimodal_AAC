from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RetrievalEvidence:
    pool: str
    bucket_id: str
    text: str
    score: float


@dataclass
class DebugInfo:
    partner_detected: str
    router_label: str
    buckets_chosen: List[str]
    sources_used: List[str]
    camera_used: bool
    face_summary: str
    face_detected: bool
    smile_score: float
    search_order: List[str]
    latency_ms: int
    groundedness_score: float = 0.0
    hallucination_flag: bool = False
    evidence_size: int = 0
    llm_enabled: bool = False
    model_used: str = ""
    llm_error: str = ""
    node_trace: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    nod_score: float = 0.0
    shake_score: float = 0.0
    negative_score: float = 0.0


@dataclass
class SessionState:
    user_id: str
    session_id: str
    started_at: str
    ltm: Dict[str, Any]
    stm: Dict[str, Any]
    pb: Dict[str, Any]
    transcript: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PipelineResult:
    options: List[str]
    evidence_used: List[RetrievalEvidence]
    debug_info: DebugInfo
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalTraceStep:
    pool: str
    attempted: bool
    top_candidates_before_refine: List[Dict[str, Any]]
    refined_evidence: List[Dict[str, Any]]
    coverage: str


@dataclass
class SpeakResult:
    grouped_suggestions: Dict[str, List[str]]
    debug_info: DebugInfo
    raw: Dict[str, Any] = field(default_factory=dict)


FaceSignals = Optional[Dict[str, float]]
