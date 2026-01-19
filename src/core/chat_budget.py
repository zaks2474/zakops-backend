#!/usr/bin/env python3
"""
Chat Budget Manager

Tracks and enforces budget/rate limits for cloud LLM providers (Gemini).
Persists daily state to JSON file for cost control.

Features:
- Daily budget limit (USD)
- Per-minute rate limiting
- Usage tracking with cost estimation
- Graceful degradation when limits exceeded
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configuration
ALLOW_CLOUD_DEFAULT = os.getenv("ALLOW_CLOUD_DEFAULT", "false").lower() == "true"
GEMINI_DAILY_BUDGET = float(os.getenv("GEMINI_DAILY_BUDGET", "5.0"))  # USD
GEMINI_RPM_LIMIT = int(os.getenv("GEMINI_RPM_LIMIT", "60"))  # Requests per minute
BUDGET_STATE_FILE = os.getenv("CHAT_BUDGET_STATE", "/tmp/chat_budget_state.json")

# Gemini pricing (approximate, per 1M tokens)
# https://cloud.google.com/vertex-ai/generative-ai/pricing
GEMINI_PRICING = {
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},  # USD per 1M tokens
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
}


@dataclass
class UsageRecord:
    """Single usage record."""
    timestamp: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class BudgetState:
    """Daily budget state."""
    date: str  # YYYY-MM-DD
    spent_usd: float = 0.0
    request_count: int = 0
    records: List[UsageRecord] = field(default_factory=list)
    request_timestamps: List[float] = field(default_factory=list)  # For RPM tracking

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "spent_usd": round(self.spent_usd, 6),
            "request_count": self.request_count,
            "records": [asdict(r) for r in self.records],
            "request_timestamps": self.request_timestamps[-100:],  # Keep last 100
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BudgetState":
        records = [UsageRecord(**r) for r in data.get("records", [])]
        return cls(
            date=data["date"],
            spent_usd=data.get("spent_usd", 0.0),
            request_count=data.get("request_count", 0),
            records=records,
            request_timestamps=data.get("request_timestamps", []),
        )


class BudgetManager:
    """
    Manages cloud LLM budget and rate limits.

    Usage:
        budget = BudgetManager()
        if budget.can_use_cloud("gemini-flash"):
            # Make request
            budget.record_usage("gemini-flash", "gemini-1.5-flash", 100, 50)
    """

    def __init__(self, state_file: str = BUDGET_STATE_FILE):
        self._state_file = Path(state_file)
        self._state: Optional[BudgetState] = None
        self._load_state()

    def _load_state(self):
        """Load or initialize budget state."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if self._state_file.exists():
            try:
                with open(self._state_file, "r") as f:
                    data = json.load(f)
                    self._state = BudgetState.from_dict(data)

                    # Reset if it's a new day
                    if self._state.date != today:
                        self._state = BudgetState(date=today)
                        self._save_state()
            except (json.JSONDecodeError, KeyError):
                self._state = BudgetState(date=today)
        else:
            self._state = BudgetState(date=today)

    def _save_state(self):
        """Persist state to file."""
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._state_file, "w") as f:
            json.dump(self._state.to_dict(), f, indent=2)

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a request."""
        pricing = GEMINI_PRICING.get(model, GEMINI_PRICING["gemini-1.5-flash"])
        cost = (
            (input_tokens / 1_000_000) * pricing["input"] +
            (output_tokens / 1_000_000) * pricing["output"]
        )
        return cost

    def _clean_old_timestamps(self):
        """Remove timestamps older than 1 minute."""
        now = time.time()
        cutoff = now - 60
        self._state.request_timestamps = [
            ts for ts in self._state.request_timestamps if ts > cutoff
        ]

    def can_use_cloud(
        self,
        provider: str = "gemini-flash",
        allow_cloud_override: Optional[bool] = None
    ) -> tuple[bool, str]:
        """
        Check if cloud provider can be used.

        Args:
            provider: Provider name (gemini-flash, gemini-pro)
            allow_cloud_override: Override the default cloud permission

        Returns:
            (allowed, reason) tuple
        """
        # Check cloud permission
        allow_cloud = allow_cloud_override if allow_cloud_override is not None else ALLOW_CLOUD_DEFAULT
        if not allow_cloud:
            return False, "Cloud disabled (ALLOW_CLOUD_DEFAULT=false)"

        # Check daily budget
        remaining = GEMINI_DAILY_BUDGET - self._state.spent_usd
        if remaining <= 0:
            return False, f"Daily budget exhausted (${GEMINI_DAILY_BUDGET:.2f})"

        # Check rate limit
        self._clean_old_timestamps()
        rpm_current = len(self._state.request_timestamps)
        if rpm_current >= GEMINI_RPM_LIMIT:
            return False, f"Rate limit exceeded ({GEMINI_RPM_LIMIT} RPM)"

        return True, f"OK (${remaining:.2f} remaining, {GEMINI_RPM_LIMIT - rpm_current} RPM available)"

    def record_usage(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int
    ):
        """Record a cloud API usage."""
        cost = self._calculate_cost(model, input_tokens, output_tokens)

        record = UsageRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

        self._state.records.append(record)
        self._state.spent_usd += cost
        self._state.request_count += 1
        self._state.request_timestamps.append(time.time())

        self._save_state()

    def get_status(self) -> Dict[str, Any]:
        """Get current budget status."""
        self._clean_old_timestamps()
        remaining = max(0, GEMINI_DAILY_BUDGET - self._state.spent_usd)

        return {
            "date": self._state.date,
            "spent_usd": round(self._state.spent_usd, 4),
            "budget_usd": GEMINI_DAILY_BUDGET,
            "remaining_usd": round(remaining, 4),
            "request_count": self._state.request_count,
            "rpm_current": len(self._state.request_timestamps),
            "rpm_limit": GEMINI_RPM_LIMIT,
            "allow_cloud_default": ALLOW_CLOUD_DEFAULT,
        }

    def reset_daily(self):
        """Manually reset daily budget (for testing)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._state = BudgetState(date=today)
        self._save_state()


# Singleton instance
_budget_instance: Optional[BudgetManager] = None


def get_budget_manager() -> BudgetManager:
    """Get or create the singleton budget manager."""
    global _budget_instance
    if _budget_instance is None:
        _budget_instance = BudgetManager()
    return _budget_instance


if __name__ == "__main__":
    # Quick test
    budget = get_budget_manager()

    print("Budget status:")
    print(json.dumps(budget.get_status(), indent=2))

    print("\nCan use cloud?")
    can_use, reason = budget.can_use_cloud("gemini-flash")
    print(f"  gemini-flash: {can_use} ({reason})")

    can_use, reason = budget.can_use_cloud("gemini-flash", allow_cloud_override=True)
    print(f"  gemini-flash (override=True): {can_use} ({reason})")

    # Simulate usage
    print("\nRecording test usage...")
    budget.record_usage("gemini-flash", "gemini-1.5-flash", 1000, 500)
    print(f"New status: {json.dumps(budget.get_status(), indent=2)}")
