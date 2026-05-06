"""
Multimodal Mapping node.

Translates raw wearable / sensor signals into stylistic and routing constraints
the rest of the pipeline can consume:

    polarity        : "positive" | "negative" | "clarify" | "neutral"
    tone            : "warm" | "calm" | "urgent" | "frustrated" | "neutral"
    verbosity       : "short" | "medium"
    bucket_boosts   : Dict[str, float]   (additive scores per bucket)
    notes           : List[str]          (human-readable mapping log)
    conflict_resolved : Optional[str]    (e.g. "vocal_vs_air_sign:trust_air_sign")

Sensors supported:
  - face cues   (smile_prob, confused_prob, nod_score, shake_score, ...)
  - gesture     (thumbs_up, thumbs_down, none)
  - head        (already in face cues but exposed here for clarity)
  - heart_rate  (BPM, resting baseline configurable)
  - gaze        (gaze_region: "family_photo" | "schedule_panel" | "med_card" | ...)
  - vocalization(vocal_polarity: "yes" | "no" | "unclear")
  - air_writing (letter: "Y" | "N" | other)

The mapping is intentionally training-free and rule-based, mirroring the
LangGraph node described in the Milestone 2 report.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


# Map gaze regions onto memory bucket names used by the rest of the pipeline.
GAZE_TO_BUCKET = {
    "family_photo": "family",
    "family": "family",
    "schedule_panel": "plans",
    "schedule": "plans",
    "calendar": "plans",
    "med_card": "medical",
    "medical": "medical",
    "food_card": "food",
    "food": "food",
    "people": "people_topics",
    "social": "greeting_smalltalk",
}


# Map face-api.js emotions (FER+ 7-class) to AAC reply tones.
# These tones are passed to the LLM (and to the rule-based fallback) so the
# generated options match how the AAC user is actually feeling.
EMOTION_TO_TONE = {
    "happy":     "warm",            # cheerful, enthusiastic
    "sad":       "gentle",          # soft, empathetic
    "angry":     "assertive",       # firm, direct, no hedging
    "surprised": "curious",         # excited, asks back
    "fearful":   "reassuring",      # calm, soothing
    "disgusted": "polite_decline",  # firm but courteous refusal
    "neutral":   "neutral",         # matter-of-fact
}

# When camera is off (no emotion read), produce 3 candidates in 3 different
# tones so the AAC user can pick whichever matches their intent.
DEFAULT_TONE_VARIANTS = ["warm", "neutral", "brief"]


def emotion_to_tone(emotion: Optional[str]) -> Optional[str]:
    if not emotion:
        return None
    return EMOTION_TO_TONE.get(str(emotion).strip().lower())


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _polarity_from_face(face: Dict[str, float]) -> Optional[str]:
    if not face:
        return None
    if face.get("shake_score", 0.0) > 0.55 or face.get("negative_prob", 0.0) > 0.55:
        return "negative"
    if face.get("nod_score", 0.0) > 0.55 or face.get("smile_prob", face.get("smile_score", 0.0)) > 0.6:
        return "positive"
    if face.get("confused_prob", 0.0) > 0.45:
        return "clarify"
    return None


def map_multimodal(
    *,
    face_signals: Optional[Dict[str, float]] = None,
    facial_emotion: Optional[str] = None,
    gesture: Optional[str] = None,
    heart_rate_bpm: Optional[float] = None,
    resting_hr_bpm: float = 70.0,
    gaze_region: Optional[str] = None,
    vocal_polarity: Optional[str] = None,
    air_sign_letter: Optional[str] = None,
) -> Dict[str, Any]:
    """Pure-function multimodal mapping.

    Returns a dict with keys: polarity, tone, verbosity, bucket_boosts,
    notes, conflict_resolved.
    """
    notes: List[str] = []
    bucket_boosts: Dict[str, float] = {}
    polarity: Optional[str] = None
    tone: str = "neutral"
    verbosity: str = "short"
    conflict_resolved: Optional[str] = None

    face = face_signals or {}

    # ---------------------------------------------------------------
    # 1. Face cues -> initial polarity / tone
    # ---------------------------------------------------------------
    face_polarity = _polarity_from_face(face)
    if face_polarity:
        polarity = face_polarity
        notes.append(f"face_polarity={face_polarity}")

    # Tone from explicit facial-emotion classifier (face-api.js) if available.
    explicit_tone = emotion_to_tone(facial_emotion)
    if explicit_tone:
        tone = explicit_tone
        notes.append(f"facial_emotion={facial_emotion} -> tone={tone}")
        # Confused/fearful face also nudges polarity toward clarify when no
        # other channel has spoken yet.
        if str(facial_emotion).lower() in {"fearful", "surprised"} and polarity is None:
            polarity = "clarify"
            notes.append("emotion=fearful/surprised -> polarity=clarify")
    else:
        smile_prob = _safe_float(face.get("smile_prob", face.get("smile_score", 0.0)))
        if smile_prob > 0.6:
            tone = "warm"
            notes.append("tone=warm (face smile)")
        elif face.get("confused_prob", 0.0) > 0.45:
            tone = "calm"
            notes.append("tone=calm (face confused)")
        elif face.get("negative_prob", 0.0) > 0.55:
            tone = "frustrated"
            notes.append("tone=frustrated (face negative)")

    # ---------------------------------------------------------------
    # 2. Gesture (thumbs-up/down) - strong polarity prior
    # ---------------------------------------------------------------
    if gesture:
        g = gesture.strip().lower()
        if g in {"thumbs_up", "thumbs-up", "yes", "ok", "agree"}:
            polarity = "positive"
            notes.append("gesture=thumbs_up -> polarity=positive")
        elif g in {"thumbs_down", "thumbs-down", "no", "decline"}:
            polarity = "negative"
            notes.append("gesture=thumbs_down -> polarity=negative")
        elif g in {"open_palm", "wait", "clarify"}:
            polarity = "clarify"
            notes.append("gesture=open_palm -> polarity=clarify")
        elif g in {"shake_hand", "uncertain", "shake", "wobble"}:
            polarity = "clarify"
            notes.append("gesture=shake_hand -> polarity=clarify (uncertain)")

    # ---------------------------------------------------------------
    # 3. Heart rate -> tone modulation + verbosity
    # ---------------------------------------------------------------
    if heart_rate_bpm is not None:
        hr = _safe_float(heart_rate_bpm)
        delta = hr - resting_hr_bpm
        if delta >= 25:
            # Elevated HR -> short, calmer wording
            tone = "calm" if tone == "neutral" else tone
            verbosity = "short"
            notes.append(f"hr={int(hr)} (>+25 over rest) -> verbosity=short, tone-soften")
        elif delta >= 10:
            verbosity = "short"
            notes.append(f"hr={int(hr)} mildly elevated -> verbosity=short")
        elif delta <= -10:
            verbosity = "medium"
            notes.append(f"hr={int(hr)} below rest -> verbosity=medium")

    # ---------------------------------------------------------------
    # 4. Gaze -> bucket boosts (Bonus #1: Gaze-based retrieval activation)
    # ---------------------------------------------------------------
    if gaze_region:
        bucket = GAZE_TO_BUCKET.get(str(gaze_region).strip().lower())
        if bucket:
            bucket_boosts[bucket] = bucket_boosts.get(bucket, 0.0) + 0.30
            notes.append(f"gaze={gaze_region} -> +0.30 boost to bucket={bucket}")

    # ---------------------------------------------------------------
    # 5. Vocal vs air-sign conflict resolution (Bonus #2)
    #    Spatial/deliberate channel wins when they disagree.
    # ---------------------------------------------------------------
    air_polarity: Optional[str] = None
    if air_sign_letter:
        letter = str(air_sign_letter).strip().upper()
        if letter in {"Y", "YES"}:
            air_polarity = "positive"
        elif letter in {"N", "NO"}:
            air_polarity = "negative"
        elif letter in {"?", "Q"}:
            air_polarity = "clarify"

    voc_polarity: Optional[str] = None
    if vocal_polarity:
        v = str(vocal_polarity).strip().lower()
        if v in {"yes", "positive", "y"}:
            voc_polarity = "positive"
        elif v in {"no", "negative", "n"}:
            voc_polarity = "negative"
        elif v in {"unclear", "?"}:
            voc_polarity = "clarify"

    if air_polarity and voc_polarity and air_polarity != voc_polarity:
        # Conflict: trust the deliberate spatial channel.
        polarity = air_polarity
        conflict_resolved = (
            f"vocal_vs_air_sign:vocal={voc_polarity} air_sign={air_polarity} -> trust_air_sign"
        )
        notes.append(conflict_resolved)
    elif air_polarity:
        polarity = air_polarity
        notes.append(f"air_sign={air_sign_letter} -> polarity={air_polarity}")
    elif voc_polarity:
        polarity = voc_polarity
        notes.append(f"vocal={vocal_polarity} -> polarity={voc_polarity}")

    # ---------------------------------------------------------------
    # 6. Defaults
    # ---------------------------------------------------------------
    if polarity is None:
        polarity = "neutral"

    return {
        "polarity": polarity,
        "tone": tone,
        "tone_locked": explicit_tone is not None,
        "facial_emotion": facial_emotion if explicit_tone else None,
        "verbosity": verbosity,
        "bucket_boosts": bucket_boosts,
        "notes": notes,
        "conflict_resolved": conflict_resolved,
    }


def polarity_opening(polarity: str) -> str:
    """Stable opening-word for a given polarity (used by the rule-based
    fallback generator and as a hard-constraint for LLM candidates)."""
    if polarity == "positive":
        return "Yes"
    if polarity == "negative":
        return "No"
    if polarity == "clarify":
        return "Could you clarify"
    return ""
