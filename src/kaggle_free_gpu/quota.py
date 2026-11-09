"""Local rolling 7-day GPU quota ledger for Kaggle Free-GPU.

Kaggle provides ~30 GPU hours per rolling 7-day window.
Tesla T4 x2 consumes 2 GPU-hours per wall-clock hour (~15 wall-hours/week).
Because Kaggle has NO API endpoint for remaining quota, we maintain a local ledger of
pushes and measured runtimes, and always direct the user to Settings -> GPU Quota
(https://www.kaggle.com/settings) for the exact authoritative number.
"""

from __future__ import annotations

import contextlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

WEEKLY_GPU_HOURS_BUDGET = 30.0
T4_X2_MULTIPLIER = 2.0  # Tesla T4 x2 burns 2 GPU-hours per wall-clock hour
DEFAULT_ESTIMATED_WALL_MINUTES_PER_PUSH = 5.0  # Conservative fallback until actual runtime is measured


class QuotaRunEntry(BaseModel):
    """A single recorded GPU kernel run in the rolling ledger."""

    ref: str
    title: str = ""
    version_number: int | None = None
    pushed_at: str  # ISO-8601 UTC
    enable_gpu: bool = True
    gpu_multiplier: float = T4_X2_MULTIPLIER
    wall_seconds: float | None = None
    wall_seconds_source: str = "estimated"  # "measured_log", "measured_poll", "manual", or "estimated"
    status: str = "PUSHED"

    @property
    def effective_wall_seconds(self) -> float:
        if not self.enable_gpu:
            return 0.0
        if self.wall_seconds is not None:
            return max(0.0, self.wall_seconds)
        return DEFAULT_ESTIMATED_WALL_MINUTES_PER_PUSH * 60.0

    @property
    def gpu_hours_used(self) -> float:
        if not self.enable_gpu:
            return 0.0
        return (self.effective_wall_seconds / 3600.0) * self.gpu_multiplier


def _norm_ref(ref: str) -> str:
    cleaned = ref.strip().strip("/").lower()
    if cleaned.startswith("code/"):
        cleaned = cleaned[len("code/") :].strip("/")
    return cleaned


class QuotaLedger:
    """Persistent local ledger stored at ~/.kaggle-free-gpu/usage.json."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (Path.home() / ".kaggle-free-gpu" / "usage.json")

    def _load_entries(self) -> list[QuotaRunEntry]:
        if not self.path.is_file():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            items = raw.get("runs", []) if isinstance(raw, dict) else []
            return [QuotaRunEntry.model_validate(item) for item in items]
        except Exception:
            return []

    def _save_entries(self, entries: list[QuotaRunEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "runs": [e.model_dump() for e in entries],
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with contextlib.suppress(OSError):
            self.path.chmod(0o600)

    def record_push(
        self,
        ref: str,
        title: str = "",
        version_number: int | None = None,
        enable_gpu: bool = True,
    ) -> QuotaRunEntry:
        """Record a new kernel push in the local ledger."""
        entries = self._load_entries()
        entry = QuotaRunEntry(
            ref=_norm_ref(ref),
            title=title,
            version_number=version_number,
            pushed_at=datetime.now(timezone.utc).isoformat(),
            enable_gpu=enable_gpu,
            gpu_multiplier=T4_X2_MULTIPLIER if enable_gpu else 0.0,
            wall_seconds=0.0 if not enable_gpu else None,
            wall_seconds_source="cpu" if not enable_gpu else "estimated",
            status="RUNNING",
        )
        entries.append(entry)
        self._save_entries(entries)
        return entry

    def update_run_duration(
        self,
        ref: str,
        wall_seconds: float | None = None,
        source: str = "measured_log",
        status: str = "COMPLETE",
    ) -> QuotaRunEntry | None:
        """Update the most recent run entry for `ref` with actual runtime or status."""
        entries = self._load_entries()
        ref_lower = _norm_ref(ref)
        target: QuotaRunEntry | None = None
        for entry in reversed(entries):
            if _norm_ref(entry.ref) == ref_lower:
                target = entry
                break

        if target is None:
            # If the kernel was pushed before local ledger existed, record it now
            target = QuotaRunEntry(
                ref=ref_lower,
                pushed_at=datetime.now(timezone.utc).isoformat(),
                enable_gpu=True,
                gpu_multiplier=T4_X2_MULTIPLIER,
                wall_seconds=wall_seconds,
                wall_seconds_source=source if wall_seconds is not None else "estimated",
                status=status,
            )
            entries.append(target)
        else:
            target.status = status
            if wall_seconds is not None and wall_seconds >= 0:
                target.wall_seconds = round(wall_seconds, 2)
                target.wall_seconds_source = source
            elif status in ("COMPLETE", "ERROR") and target.wall_seconds is None:
                # Estimate elapsed time from pushed_at if within reasonable bounds (<12h)
                try:
                    pushed_dt = datetime.fromisoformat(target.pushed_at)
                    elapsed = (datetime.now(timezone.utc) - pushed_dt).total_seconds()
                    if 0 < elapsed <= 12 * 3600:
                        target.wall_seconds = round(elapsed, 2)
                        target.wall_seconds_source = "measured_poll"
                except Exception:
                    pass

        self._save_entries(entries)
        return target

    def record_manual_adjustment(self, gpu_hours_used: float, note: str = "Manual sync from Kaggle site") -> None:
        """Allow syncing the local ledger with the Kaggle site's reported GPU quota."""
        entries = self._load_entries()
        wall_seconds = (gpu_hours_used / T4_X2_MULTIPLIER) * 3600.0
        entries.append(
            QuotaRunEntry(
                ref="manual/quota-sync",
                title=note,
                pushed_at=datetime.now(timezone.utc).isoformat(),
                enable_gpu=True,
                gpu_multiplier=T4_X2_MULTIPLIER,
                wall_seconds=round(wall_seconds, 2),
                wall_seconds_source="manual",
                status="COMPLETE",
            )
        )
        self._save_entries(entries)

    def estimate(self) -> dict[str, Any]:
        """Compute rolling 7-day GPU quota usage and remaining hours."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=7)
        entries = self._load_entries()

        rolling_entries: list[QuotaRunEntry] = []
        for entry in entries:
            try:
                dt = datetime.fromisoformat(entry.pushed_at)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt >= cutoff:
                    rolling_entries.append(entry)
            except Exception:
                rolling_entries.append(entry)

        gpu_hours_used = sum(e.gpu_hours_used for e in rolling_entries)
        wall_hours_used = sum(e.effective_wall_seconds / 3600.0 for e in rolling_entries if e.enable_gpu)
        gpu_hours_remaining = max(0.0, WEEKLY_GPU_HOURS_BUDGET - gpu_hours_used)
        t4_x2_wall_hours_remaining = gpu_hours_remaining / T4_X2_MULTIPLIER

        return {
            "weekly_budget_gpu_hours": WEEKLY_GPU_HOURS_BUDGET,
            "t4_x2_multiplier": T4_X2_MULTIPLIER,
            "rolling_7d_gpu_run_count": sum(1 for e in rolling_entries if e.enable_gpu),
            "rolling_7d_total_pushes": len(rolling_entries),
            "estimated_gpu_hours_used": round(gpu_hours_used, 3),
            "estimated_wall_hours_used_t4_x2": round(wall_hours_used, 3),
            "estimated_gpu_hours_remaining": round(gpu_hours_remaining, 3),
            "estimated_wall_hours_remaining_t4_x2": round(t4_x2_wall_hours_remaining, 3),
            "single_session_cap_hours": 12.0,
            "window_type": "rolling 7-day window (hours free up 7 days after they were used, not a Monday reset)",
            "ledger_path": str(self.path),
            "recent_runs": [
                {
                    "ref": e.ref,
                    "version": e.version_number,
                    "pushed_at": e.pushed_at,
                    "enable_gpu": e.enable_gpu,
                    "status": e.status,
                    "wall_seconds": round(e.effective_wall_seconds, 1),
                    "wall_seconds_source": e.wall_seconds_source,
                    "gpu_hours": round(e.gpu_hours_used, 4),
                }
                for e in rolling_entries[-10:]
            ],
            "notice": (
                "IMPORTANT: Kaggle provides no API endpoint for remaining GPU quota. "
                "This number is estimated from local push history in ~/.kaggle-free-gpu/usage.json. "
                "Always check Settings -> GPU Quota at https://www.kaggle.com/settings for your exact balance."
            ),
        }
