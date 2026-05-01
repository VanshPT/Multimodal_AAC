import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from django.conf import settings


def log_event(event: Dict[str, Any]) -> None:
    out_dir = Path(settings.BASE_DIR) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
    with (out_dir / "run_logs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
