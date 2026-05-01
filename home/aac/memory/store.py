import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

from django.conf import settings

from home.aac.types import SessionState


class MemoryStoreError(Exception):
    """Raised when user memory files are missing or invalid."""


class MemoryStore:
    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}

    @property
    def base_dir(self) -> Path:
        return Path(settings.BASE_DIR) / "data" / "synthetic_users"

    def list_users(self):
        if not self.base_dir.exists():
            return []
        users = [path.name for path in self.base_dir.iterdir() if path.is_dir()]
        return sorted(users)

    def _load_json(self, user_id: str, filename: str):
        path = self.base_dir / user_id / filename
        if not path.exists():
            raise MemoryStoreError(f"Required memory file not found: {filename}")
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError as error:
            raise MemoryStoreError(f"Invalid JSON in memory file: {filename}") from error

    def _write_json(self, user_id: str, filename: str, payload):
        path = self.base_dir / user_id / filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def start_session(self, user_id: str) -> SessionState:
        session_id = str(uuid4())
        started_at = datetime.now(timezone.utc).isoformat()
        ltm = self._load_json(user_id, "long_term_profile.json")
        stm = self._load_json(user_id, "short_term_memory.json")
        pb = self._load_json(user_id, "phrases.json")
        stm.setdefault("active_session", {})
        stm["active_session"].update(
            {
                "session_id": session_id,
                "started_at": started_at,
                "updated_at": started_at,
            }
        )
        state = SessionState(
            user_id=user_id,
            session_id=session_id,
            started_at=started_at,
            ltm=deepcopy(ltm),
            stm=deepcopy(stm),
            pb=deepcopy(pb),
            transcript=[],
        )
        self._sessions[session_id] = state
        return state

    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)

    def persist_session_memories(self, state: SessionState):
        state.stm.setdefault("active_session", {})
        state.stm["active_session"]["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write_json(state.user_id, "short_term_memory.json", state.stm)
        self._write_json(state.user_id, "phrases.json", state.pb)

    def session_snapshot(self, session_id: str):
        state = self.get_session(session_id)
        if not state:
            raise MemoryStoreError("Session not found. Press START first.")
        return {
            "session_id": session_id,
            "final_output": state.stm.get("final_output"),
            "pending_options": state.stm.get("last_options", []),
            "active_partner": state.stm.get("active_partner", {}),
            "active_session": state.stm.get("active_session", {}),
            "loaded_summary": state.stm.get("loaded_summary", {}),
            "query_metrics_history": state.stm.get("query_metrics_history", [])[-30:],
        }


memory_store = MemoryStore()
