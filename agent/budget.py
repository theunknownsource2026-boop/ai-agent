"""
Budget tracker for AI API spending.

Logs every API call to a ``costs.jsonl`` file and provides helpers to
query daily / monthly totals and enforce spending limits.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from agent import config

logger = logging.getLogger(__name__)

# Default cost log lives in the project root
_DEFAULT_COST_FILE = Path(__file__).resolve().parent.parent / "costs.jsonl"


class BudgetTracker:
    """Track API spending and enforce budget limits."""

    def __init__(self, cost_file: str | Path | None = None):
        self._cost_file = Path(cost_file) if cost_file else _DEFAULT_COST_FILE
        # Ensure the file exists
        self._cost_file.touch(exist_ok=True)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_call(
        self,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Calculate cost for an API call and append it to the cost log.

        Returns the cost in USD for this single call.
        """
        pricing = config.COST_PER_1K_TOKENS.get(
            model, {"input": 0.0, "output": 0.0}
        )
        cost = (
            input_tokens * pricing["input"] + output_tokens * pricing["output"]
        ) / 1000.0

        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "date": date.today().isoformat(),
            "model": model,
            "provider": provider,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 8),
        }

        try:
            with open(self._cost_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError as exc:
            logger.error("Failed to write cost log: %s", exc)

        return cost

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def _load_entries(self) -> list[dict]:
        """Load all JSONL entries from the cost file."""
        entries: list[dict] = []
        try:
            with open(self._cost_file, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            pass
        return entries

    def get_daily_total(self, target_date: date | None = None) -> float:
        """Sum costs for a given date (defaults to today)."""
        target = (target_date or date.today()).isoformat()
        return sum(
            e.get("cost_usd", 0.0)
            for e in self._load_entries()
            if e.get("date") == target
        )

    def get_monthly_total(self, year: int | None = None, month: int | None = None) -> float:
        """Sum costs for a given month (defaults to current month)."""
        now = date.today()
        y = year or now.year
        m = month or now.month
        prefix = f"{y:04d}-{m:02d}"
        return sum(
            e.get("cost_usd", 0.0)
            for e in self._load_entries()
            if e.get("date", "").startswith(prefix)
        )

    def is_over_budget(self) -> bool:
        """Return True if daily OR monthly spending exceeds the configured limits."""
        daily = self.get_daily_total()
        monthly = self.get_monthly_total()
        return (
            daily > config.DAILY_BUDGET_LIMIT
            or monthly > config.MONTHLY_BUDGET_LIMIT
        )

    def get_summary(self) -> str:
        """Return a human-readable spending summary."""
        daily = self.get_daily_total()
        monthly = self.get_monthly_total()
        status = "OVER BUDGET" if self.is_over_budget() else "within budget"
        return (
            f"Budget Summary ({status})\n"
            f"  Today:   ${daily:,.4f} / ${config.DAILY_BUDGET_LIMIT:,.2f} daily limit\n"
            f"  Month:   ${monthly:,.4f} / ${config.MONTHLY_BUDGET_LIMIT:,.2f} monthly limit"
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_breakdown(self, target_date: date | None = None) -> dict[str, float]:
        """Return cost breakdown by model for a given date."""
        target = (target_date or date.today()).isoformat()
        breakdown: dict[str, float] = {}
        for e in self._load_entries():
            if e.get("date") == target:
                model = e.get("model", "unknown")
                breakdown[model] = breakdown.get(model, 0.0) + e.get("cost_usd", 0.0)
        return breakdown

    def reset(self) -> None:
        """Clear the cost log. Use with caution."""
        with open(self._cost_file, "w", encoding="utf-8") as fh:
            fh.truncate(0)
        logger.info("Cost log cleared: %s", self._cost_file)
