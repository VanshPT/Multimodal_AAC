from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Dict
from uuid import uuid4


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class VoiceTurnRecord:
    session_id: str
    voice_turn_id: str
    room_name: str
    transcript: str = ""
    ready: bool = False
    consumed: bool = False
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)


class LiveKitVoiceBridge:
    def __init__(self) -> None:
        self._records: Dict[str, VoiceTurnRecord] = {}
        self._lock = Lock()

    def create_turn(self, session_id: str, room_name: str) -> VoiceTurnRecord:
        record = VoiceTurnRecord(
            session_id=session_id,
            voice_turn_id=str(uuid4()),
            room_name=room_name,
            created_at=_utc_now(),
            updated_at=_utc_now(),
        )
        with self._lock:
            self._cleanup_locked()
            self._records[self._key(session_id, record.voice_turn_id)] = record
        return record

    def save_transcript(self, session_id: str, voice_turn_id: str, transcript: str) -> VoiceTurnRecord | None:
        cleaned = (transcript or "").strip()
        if not cleaned:
            return None
        with self._lock:
            self._cleanup_locked()
            record = self._records.get(self._key(session_id, voice_turn_id))
            if not record:
                return None
            record.transcript = cleaned
            record.ready = True
            record.consumed = False
            record.updated_at = _utc_now()
            return record

    def fetch_transcript(self, session_id: str, voice_turn_id: str, consume: bool = True) -> dict:
        with self._lock:
            self._cleanup_locked()
            record = self._records.get(self._key(session_id, voice_turn_id))
            if not record:
                return {"ready": False, "transcript": "", "found": False}
            payload = {
                "ready": bool(record.ready),
                "transcript": record.transcript if record.ready else "",
                "found": True,
                "consumed": bool(record.consumed),
                "room_name": record.room_name,
            }
            if consume and record.ready:
                record.consumed = True
                record.updated_at = _utc_now()
            return payload

    def _cleanup_locked(self) -> None:
        cutoff = _utc_now() - timedelta(minutes=20)
        stale_keys = [key for key, value in self._records.items() if value.updated_at < cutoff]
        for key in stale_keys:
            del self._records[key]

    @staticmethod
    def _key(session_id: str, voice_turn_id: str) -> str:
        return f"{session_id}:{voice_turn_id}"


livekit_voice_bridge = LiveKitVoiceBridge()
