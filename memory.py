import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import WORKSPACE


class JsonMemoryStore:
    """Simple per-user JSON memory store for cross-session persistence."""

    def __init__(self, path: Path | None = None, max_reports: int = 20):
        self.path = path or (WORKSPACE / "memory" / "user_memory.json")
        self.max_reports = max_reports
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load_all(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_all(self, data: dict[str, Any]) -> None:
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_user_memory(self, user_id: str) -> dict[str, Any]:
        data = self._load_all()
        return data.get(user_id, {"preferences": {}, "reports": []})

    def upsert_preferences(self, user_id: str, preferences: dict[str, Any]) -> None:
        if not preferences:
            return
        with self._lock:
            data = self._load_all()
            user_mem = data.setdefault(user_id, {"preferences": {}, "reports": []})
            user_mem.setdefault("preferences", {}).update(preferences)
            data[user_id] = user_mem
            self._save_all(data)

    def append_report_memory(self, user_id: str, goal: str, summary: str, pdf_path: str) -> None:
        with self._lock:
            data = self._load_all()
            user_mem = data.setdefault(user_id, {"preferences": {}, "reports": []})
            reports = user_mem.setdefault("reports", [])
            reports.append(
                {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "goal": goal,
                    "summary": (summary or "")[:2000],
                    "pdf_path": pdf_path,
                }
            )
            user_mem["reports"] = reports[-self.max_reports :]
            data[user_id] = user_mem
            self._save_all(data)


memory_store = JsonMemoryStore()
