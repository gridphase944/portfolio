"""Compact reconstruction helpers for explicit agent-history diagnostics."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import pandas as pd


ELAPSED_BUCKET_UPPER_SECONDS_V01 = (5, 10, 15, 20, 25, 30)


def reconstruct_close_reentry_diagnostics_v01(
    decisions: Sequence[Mapping[str, Any]],
    *,
    history_depth: int,
) -> dict[str, Any]:
    """Reconstruct successful-normal-Close follow-up decisions in one pass."""

    if history_depth not in {1, 2, 3}:
        raise ValueError("history_depth must be 1, 2, or 3")
    anchor: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    successful_close_count = 0
    invariant_violation_count = 0
    for index, source in enumerate(decisions):
        row = dict(source)
        timestamp = _timestamp(row.get("timestamp"))
        session = str(row.get("session", ""))
        position_before = str(row.get("position_before", ""))
        position_after = str(row.get("position_after", ""))
        action = str(row.get("selected_action", ""))
        execution_failed = bool(row.get("execution_failed", False))
        forced_exit = bool(row.get("forced_exit", False))

        close_side = _successful_normal_close_side(
            position_before=position_before,
            position_after=position_after,
            action=action,
            execution_failed=execution_failed,
            forced_exit=forced_exit,
        )
        if close_side is not None:
            anchor = {
                "index": index,
                "timestamp": timestamp,
                "session": session,
                "close_side": close_side,
                "close_action": action,
            }
            successful_close_count += 1
            continue
        if anchor is None:
            continue
        if session != anchor["session"]:
            anchor = None
            continue
        elapsed_seconds = (timestamp - anchor["timestamp"]).total_seconds()
        if elapsed_seconds <= 0.0 or not math.isfinite(elapsed_seconds):
            invariant_violation_count += 1
            continue
        if position_before != "Flat" or forced_exit:
            anchor = None
            continue

        groups = list(row.get("history_groups", ()))
        if len(groups) != history_depth:
            raise ValueError("history_groups length must match history_depth")
        group_availability = [bool(group.get("available", False)) for group in groups]
        if any(
            group_availability[lag] and not group_availability[lag - 1]
            for lag in range(1, len(group_availability))
        ):
            raise ValueError("history group availability must be a newest-first prefix")
        original_close_context_present = any(
            bool(group.get("available", False))
            and group.get("position_before") == anchor["close_side"]
            and group.get("effective_action") == anchor["close_action"]
            for group in groups
        )
        classification = _reentry_direction(
            close_side=anchor["close_side"],
            action=action,
        )
        q_values = _q_values(row)
        best_q = max(q_values.values()) if q_values is not None else None
        selected_q = q_values.get(action) if q_values is not None else None
        sorted_q = sorted(q_values.values()) if q_values is not None else []
        q_margin = (
            sorted_q[-1] - sorted_q[-2] if len(sorted_q) >= 2 else None
        )
        rows.append(
            {
                "close_index": int(anchor["index"]),
                "decision_index": index,
                "close_side": anchor["close_side"],
                "elapsed_seconds": float(elapsed_seconds),
                "elapsed_bucket": _elapsed_bucket(elapsed_seconds),
                "selected_action": action,
                "reentry_direction": classification,
                "same_side_reentry": classification == "same_side_reentry",
                "reversal": classification == "reversal",
                "hold": classification == "hold",
                "history_availability": group_availability,
                "original_close_context_present": original_close_context_present,
                "first_decision_without_original_close_context": bool(
                    elapsed_seconds == 5.0 * (history_depth + 1)
                    and not original_close_context_present
                ),
                "q_hold": q_values.get("Hold") if q_values is not None else None,
                "q_buy": q_values.get("Buy") if q_values is not None else None,
                "q_sell": q_values.get("Sell") if q_values is not None else None,
                "best_q": best_q,
                "q_margin": q_margin,
                "selected_q": selected_q,
            }
        )
        if classification in {"same_side_reentry", "reversal"} and not execution_failed:
            anchor = None
    return {
        "schema_id": "agent_history_close_reentry_reconstruction_v01",
        "history_depth": history_depth,
        "expected_first_decision_without_close_context_seconds": 5 * (
            history_depth + 1
        ),
        "successful_normal_close_count": successful_close_count,
        "followup_decision_count": len(rows),
        "invariant_violation_count": invariant_violation_count,
        "rows": rows,
    }


def _successful_normal_close_side(
    *,
    position_before: str,
    position_after: str,
    action: str,
    execution_failed: bool,
    forced_exit: bool,
) -> str | None:
    if execution_failed or forced_exit or position_after != "Flat":
        return None
    if position_before == "Long" and action == "Sell":
        return "Long"
    if position_before == "Short" and action == "Buy":
        return "Short"
    return None


def _reentry_direction(*, close_side: str, action: str) -> str:
    if action == "Hold":
        return "hold"
    if close_side == "Long":
        return "same_side_reentry" if action == "Buy" else "reversal"
    if close_side == "Short":
        return "same_side_reentry" if action == "Sell" else "reversal"
    raise ValueError("close_side must be Long or Short")


def _elapsed_bucket(elapsed_seconds: float) -> str:
    for upper in ELAPSED_BUCKET_UPPER_SECONDS_V01:
        if elapsed_seconds <= upper:
            return f"<={upper}s"
    return ">30s"


def _timestamp(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tz is None:
        raise ValueError("decision timestamp must be timezone-aware")
    return result


def _q_values(row: Mapping[str, Any]) -> dict[str, float] | None:
    keys = {"Hold": "q_hold", "Buy": "q_buy", "Sell": "q_sell"}
    if not any(row.get(key) is not None for key in keys.values()):
        return None
    values = {action: float(row.get(key)) for action, key in keys.items()}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Q diagnostic values must be finite when present")
    return values


__all__ = [
    "ELAPSED_BUCKET_UPPER_SECONDS_V01",
    "reconstruct_close_reentry_diagnostics_v01",
]
