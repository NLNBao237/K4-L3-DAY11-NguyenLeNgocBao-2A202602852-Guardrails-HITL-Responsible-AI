"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}
        self._pending: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store input + start timestamp keyed by request_id; return the request_id."""
        request_id = request_id or uuid.uuid4().hex[:12]
        self._open[request_id] = time.perf_counter()
        self._pending[request_id] = {
            "request_id": request_id,
            "user_id": user_id,
            "input": text,
            "timestamp_in": utc_now_iso(),
        }
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store output, layer decision, latency; append to self.logs."""
        started = self._open.pop(request_id, None) if request_id else None
        entry = self._pending.pop(request_id, None) if request_id else None
        entry = entry or {"request_id": request_id, "user_id": user_id, "input": None}
        entry.update(
            {
                "output": text,
                "blocked": blocked,
                "layer": layer,
                "timestamp_out": utc_now_iso(),
                "latency_ms": (
                    round((time.perf_counter() - started) * 1000, 1)
                    if started is not None
                    else None
                ),
            }
        )
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
