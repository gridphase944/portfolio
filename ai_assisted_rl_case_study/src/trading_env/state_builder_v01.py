"""TradingEnv-to-state-vector adapter utilities v0.1.

This module builds the minimal internal-state dictionary consumed by
features.state_vector_v01. It uses the current row's liquidation-side Top5 book
snapshot for position distance / unrealized return and does not use
CurrentPrice as an execution or valuation price.
"""

from __future__ import annotations

import math

import pandas as pd
from features.state_vector_v01 import build_state_vector_v01

from .reward_v01 import ORDER_QTY, OrderBookLevel, compute_top5_weighted_price


def build_env_internal_state_v01(
    position_side: str,
    entry_price: float | None,
    holding_time_seconds: float,
    row: dict,
    cooldown_remaining_steps: int = 0,
    cooldown_length_steps: int = 0,
) -> dict[str, float | str]:
    """Build v0.1 internal state for one Env row."""

    if position_side == "Flat":
        return {
            "position_side": "Flat",
            "position_size": 0,
            "holding_time_seconds": 0.0,
            "entry_price_distance": 0.0,
            "unrealized_return": 0.0,
            "cooldown_remaining_steps": int(max(0, cooldown_remaining_steps)),
            "cooldown_length_steps": int(max(0, cooldown_length_steps)),
        }

    liquidation_price = _liquidation_price(position_side, row)
    position_return = _position_return(position_side, entry_price, liquidation_price)

    return {
        "position_side": position_side,
        "position_size": ORDER_QTY,
        "holding_time_seconds": float(holding_time_seconds),
        "entry_price_distance": position_return,
        "unrealized_return": position_return,
        "cooldown_remaining_steps": int(max(0, cooldown_remaining_steps)),
        "cooldown_length_steps": int(max(0, cooldown_length_steps)),
    }


def build_env_state_vector_v01(
    row: dict,
    position_side: str,
    entry_price: float | None,
    holding_time_seconds: float,
    cooldown_remaining_steps: int = 0,
    cooldown_length_steps: int = 0,
    state_candidate_name: str | None = None,
    previous_decision_position: str | None = None,
    previous_effective_action: str | None = None,
    agent_transition_history: tuple[object, ...] | list[object] | None = None,
):
    """Build StateVectorResult from one Env row and current internal state."""

    internal_state = build_env_internal_state_v01(
        position_side=position_side,
        entry_price=entry_price,
        holding_time_seconds=holding_time_seconds,
        row=row,
        cooldown_remaining_steps=cooldown_remaining_steps,
        cooldown_length_steps=cooldown_length_steps,
    )
    internal_state["previous_decision_position"] = previous_decision_position
    internal_state["previous_effective_action"] = previous_effective_action
    internal_state["agent_transition_history"] = agent_transition_history
    market_row = row if isinstance(row, pd.Series) else pd.Series(row)
    return build_state_vector_v01(
        market_row,
        internal_state,
        state_candidate_name=state_candidate_name,
    )


def _liquidation_price(position_side: str, row: dict) -> float | None:
    if position_side == "Long":
        used_book_side = "BuyTop5"
        prefix = "Buy"
    elif position_side == "Short":
        used_book_side = "SellTop5"
        prefix = "Sell"
    else:
        return None

    levels = [
        OrderBookLevel(
            price=row.get(f"{prefix}{level}_Price"),
            qty=row.get(f"{prefix}{level}_Qty"),
        )
        for level in range(1, 6)
    ]
    result = compute_top5_weighted_price(
        levels,
        qty=ORDER_QTY,
        used_book_side=used_book_side,
    )
    return result.price if result.executable else None


def _position_return(
    position_side: str,
    entry_price: float | None,
    liquidation_price: float | None,
) -> float:
    if entry_price is None or liquidation_price is None:
        return float("nan")

    try:
        entry = float(entry_price)
        liquidation = float(liquidation_price)
    except (TypeError, ValueError):
        return float("nan")

    if not math.isfinite(entry) or not math.isfinite(liquidation) or entry <= 0:
        return float("nan")

    if position_side == "Long":
        return liquidation / entry - 1.0
    if position_side == "Short":
        return entry / liquidation - 1.0
    return float("nan")
