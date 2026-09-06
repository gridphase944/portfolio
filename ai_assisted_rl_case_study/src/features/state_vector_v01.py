"""State-vector assembly utilities v0.1.

This module combines already-built market features, 1570 proxy features, and
agent internal state into a fixed-order one-dimensional vector for model input.
It does not perform normalization, feature selection, reward calculation,
TradingEnv integration, execution-price approximation, or 1357 proxy handling.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np
import pandas as pd

from .state_candidate_features_v01 import (
    SPREAD_LIQUIDITY_STATE_COLUMNS,
    VOLATILITY_MOVEMENT_STATE_COLUMNS,
    WINDOW_900S_STATE_COLUMNS,
)


PRICE_STATE_COLUMNS = (
    "return_5s",
    "return_30s",
    "return_120s",
    "return_300s",
    "log_return_30s",
    "log_return_120s",
    "log_return_300s",
    "gap_from_prev_close",
    "gap_from_open",
    "gap_from_vwap",
    "intraday_range_position",
)
VOLUME_VALUE_STATE_COLUMNS = (
    "volume_delta_30s",
    "volume_delta_120s",
    "volume_delta_300s",
    "value_delta_30s",
    "value_delta_120s",
    "value_delta_300s",
    "volume_intensity_30s",
    "value_intensity_30s",
)
L1_STATE_COLUMNS = (
    "best_sell_qty",
    "best_buy_qty",
    "spread",
    "relative_spread",
    "l1_book_imbalance",
)
TOP5_STATE_COLUMNS = (
    "book_imbalance_top3",
    "book_imbalance_top5",
    "buy_qty_top3_ratio",
    "sell_qty_top3_ratio",
    "buy_qty_top5_ratio",
    "sell_qty_top5_ratio",
    "buy_distance_top3",
    "sell_distance_top3",
    "buy_distance_top5",
    "sell_distance_top5",
)
STALENESS_STATE_COLUMNS = (
    "trade_staleness_seconds",
    "quote_staleness_seconds",
    "high_update_age_seconds",
    "low_update_age_seconds",
)
TIME_STATE_COLUMNS = (
    "minutes_from_session_start",
    "minutes_to_session_end",
    "is_morning_session",
    "is_afternoon_session",
)
PROXY_1570_STATE_COLUMNS = (
    "mkt_1570_return_30s",
    "mkt_1570_return_120s",
    "mkt_1570_return_300s",
    "mkt_1570_gap_from_prev_close",
    "mkt_1570_gap_from_vwap",
    "mkt_1570_volume_delta_120s",
    "mkt_1570_value_delta_120s",
    "mkt_1570_trade_staleness_seconds",
)
PROXY_1357_STATE_COLUMNS = (
    "mkt_1357_return_5s",
    "mkt_1357_return_30s",
    "mkt_1357_return_120s",
    "mkt_1357_return_300s",
    "mkt_1357_log_return_30s",
    "mkt_1357_log_return_120s",
    "mkt_1357_log_return_300s",
    "mkt_1357_volume_delta_30s",
    "mkt_1357_volume_delta_120s",
    "mkt_1357_volume_delta_300s",
    "mkt_1357_value_delta_30s",
    "mkt_1357_value_delta_120s",
    "mkt_1357_value_delta_300s",
    "mkt_1357_volume_intensity_30s",
    "mkt_1357_volume_intensity_120s",
    "mkt_1357_volume_intensity_300s",
    "mkt_1357_value_intensity_30s",
    "mkt_1357_value_intensity_120s",
    "mkt_1357_value_intensity_300s",
    "mkt_1357_trade_staleness_seconds",
    "mkt_1357_quote_staleness_seconds",
)
PROXY_1570_1357_DIVERGENCE_STATE_COLUMNS = (
    "proxy_return_divergence_30s",
    "proxy_return_divergence_120s",
    "proxy_return_divergence_300s",
    "proxy_spread_1570_1357_30s",
    "proxy_spread_1570_1357_120s",
    "proxy_spread_1570_1357_300s",
    "bull_bear_proxy_disagreement_30s",
    "bull_bear_proxy_disagreement_120s",
    "bull_bear_proxy_disagreement_300s",
)
STATE_CANDIDATE_1357_ADDED_COLUMNS = (
    PROXY_1357_STATE_COLUMNS + PROXY_1570_1357_DIVERGENCE_STATE_COLUMNS
)
MARKET_STATE_COLUMNS = (
    PRICE_STATE_COLUMNS
    + VOLUME_VALUE_STATE_COLUMNS
    + L1_STATE_COLUMNS
    + TOP5_STATE_COLUMNS
    + STALENESS_STATE_COLUMNS
    + TIME_STATE_COLUMNS
    + PROXY_1570_STATE_COLUMNS
)
INTERNAL_STATE_COLUMNS = (
    "position_side_flat",
    "position_side_long",
    "position_side_short",
    "position_size",
    "holding_time_seconds",
    "entry_price_distance",
    "unrealized_return",
    "cooldown_active",
    "cooldown_remaining_steps_norm",
)
STATE_COLUMNS = MARKET_STATE_COLUMNS + INTERNAL_STATE_COLUMNS
LAG1_AGENT_TRANSITION_CONTEXT_COLUMNS = (
    "prev_decision_position_flat",
    "prev_decision_position_long",
    "prev_decision_position_short",
    "prev_effective_action_hold",
    "prev_effective_action_buy",
    "prev_effective_action_sell",
)
LAG2_AGENT_TRANSITION_CONTEXT_COLUMNS = (
    "prev1_decision_position_flat",
    "prev1_decision_position_long",
    "prev1_decision_position_short",
    "prev1_effective_action_hold",
    "prev1_effective_action_buy",
    "prev1_effective_action_sell",
    "prev2_decision_position_flat",
    "prev2_decision_position_long",
    "prev2_decision_position_short",
    "prev2_effective_action_hold",
    "prev2_effective_action_buy",
    "prev2_effective_action_sell",
)
LAG3_AGENT_TRANSITION_CONTEXT_COLUMNS = (
    *LAG2_AGENT_TRANSITION_CONTEXT_COLUMNS,
    "prev3_decision_position_flat",
    "prev3_decision_position_long",
    "prev3_decision_position_short",
    "prev3_effective_action_hold",
    "prev3_effective_action_buy",
    "prev3_effective_action_sell",
)
STATE_SPEC_VERSION_V01 = "state_vector_v02_cooldown_v01"
STATE_CANDIDATE_ENV_VAR_V01 = "TRADING_STATE_CANDIDATE_NAME"
STATE_CANDIDATE_1357_PROXY_V01 = "state_candidate_v01_1357_proxy"
STATE_CANDIDATE_VOLATILITY_MOVEMENT_V01 = "state_candidate_v02_volatility_movement_regime"
STATE_CANDIDATE_SPREAD_LIQUIDITY_V01 = "state_candidate_v03_spread_liquidity_regime"
STATE_CANDIDATE_900S_WINDOW_V01 = "state_candidate_v04_900s_window"
STATE_CANDIDATE_LAG1_AGENT_TRANSITION_CONTEXT_V01 = (
    "state_candidate_v05_lag1_agent_transition_context"
)
STATE_CANDIDATE_LAG2_AGENT_TRANSITION_CONTEXT_V01 = (
    "state_candidate_v06_lag2_agent_transition_context"
)
STATE_CANDIDATE_LAG3_AGENT_TRANSITION_CONTEXT_V01 = (
    "state_candidate_v07_lag3_agent_transition_context"
)
STATE_DIM_V01 = len(STATE_COLUMNS)


@dataclass
class StateVectorResult:
    df: pd.DataFrame
    state_columns: list[str]
    state_vector: np.ndarray
    qc: dict


def get_state_columns_v01(
    state_candidate_name: str | None = None,
) -> list[str]:
    """Return the fixed cooldown-aware state-vector column order."""

    candidate = get_state_candidate_name_v01(state_candidate_name)
    history_columns = get_agent_transition_history_columns_v01(candidate)
    if history_columns:
        return list(STATE_COLUMNS + history_columns)
    added_columns = tuple(get_state_candidate_added_columns_v01(candidate))
    if added_columns:
        return list(MARKET_STATE_COLUMNS + added_columns + INTERNAL_STATE_COLUMNS)
    return list(STATE_COLUMNS)


def get_state_spec_version_v01(
    state_candidate_name: str | None = None,
) -> str:
    """Return the state spec version for persisted artifacts and manifests."""

    candidate = get_state_candidate_name_v01(state_candidate_name)
    if candidate != "current_state":
        return f"{STATE_SPEC_VERSION_V01}__{candidate}"
    return STATE_SPEC_VERSION_V01


def get_state_candidate_name_v01(
    state_candidate_name: str | None = None,
) -> str:
    explicit = state_candidate_name is not None
    candidate = (
        str(state_candidate_name)
        if explicit
        else os.environ.get(STATE_CANDIDATE_ENV_VAR_V01)
    )
    if candidate == "current_state":
        return "current_state"
    if candidate in {
        STATE_CANDIDATE_1357_PROXY_V01,
        STATE_CANDIDATE_VOLATILITY_MOVEMENT_V01,
        STATE_CANDIDATE_SPREAD_LIQUIDITY_V01,
        STATE_CANDIDATE_900S_WINDOW_V01,
        STATE_CANDIDATE_LAG1_AGENT_TRANSITION_CONTEXT_V01,
        STATE_CANDIDATE_LAG2_AGENT_TRANSITION_CONTEXT_V01,
        STATE_CANDIDATE_LAG3_AGENT_TRANSITION_CONTEXT_V01,
    }:
        return str(candidate)
    if explicit:
        raise ValueError(f"unknown state candidate: {candidate!r}")
    return "current_state"


def get_state_candidate_added_columns_v01(
    state_candidate_name: str | None = None,
) -> list[str]:
    candidate = get_state_candidate_name_v01(state_candidate_name)
    if candidate == STATE_CANDIDATE_1357_PROXY_V01:
        return list(STATE_CANDIDATE_1357_ADDED_COLUMNS)
    if candidate == STATE_CANDIDATE_VOLATILITY_MOVEMENT_V01:
        return list(VOLATILITY_MOVEMENT_STATE_COLUMNS)
    if candidate == STATE_CANDIDATE_SPREAD_LIQUIDITY_V01:
        return list(SPREAD_LIQUIDITY_STATE_COLUMNS)
    if candidate == STATE_CANDIDATE_900S_WINDOW_V01:
        return list(WINDOW_900S_STATE_COLUMNS)
    return []


def state_candidate_enabled_v01(
    state_candidate_name: str | None = None,
) -> bool:
    return get_state_candidate_name_v01(state_candidate_name) != "current_state"


def state_candidate_1357_proxy_enabled_v01() -> bool:
    return get_state_candidate_name_v01() == STATE_CANDIDATE_1357_PROXY_V01


def state_candidate_volatility_movement_enabled_v01() -> bool:
    return get_state_candidate_name_v01() == STATE_CANDIDATE_VOLATILITY_MOVEMENT_V01


def state_candidate_spread_liquidity_enabled_v01() -> bool:
    return get_state_candidate_name_v01() == STATE_CANDIDATE_SPREAD_LIQUIDITY_V01


def state_candidate_900s_window_enabled_v01() -> bool:
    return get_state_candidate_name_v01() == STATE_CANDIDATE_900S_WINDOW_V01


def state_candidate_lag1_agent_transition_context_enabled_v01(
    state_candidate_name: str | None = None,
) -> bool:
    return (
        get_state_candidate_name_v01(state_candidate_name)
        == STATE_CANDIDATE_LAG1_AGENT_TRANSITION_CONTEXT_V01
    )


def get_agent_transition_history_depth_v01(
    state_candidate_name: str | None = None,
) -> int:
    """Return the candidate-owned explicit history depth, or zero."""

    candidate = get_state_candidate_name_v01(state_candidate_name)
    return {
        STATE_CANDIDATE_LAG1_AGENT_TRANSITION_CONTEXT_V01: 1,
        STATE_CANDIDATE_LAG2_AGENT_TRANSITION_CONTEXT_V01: 2,
        STATE_CANDIDATE_LAG3_AGENT_TRANSITION_CONTEXT_V01: 3,
    }.get(candidate, 0)


def get_agent_transition_history_columns_v01(
    state_candidate_name: str | None = None,
) -> tuple[str, ...]:
    """Return the exact schema-bound history columns for one candidate."""

    candidate = get_state_candidate_name_v01(state_candidate_name)
    return {
        STATE_CANDIDATE_LAG1_AGENT_TRANSITION_CONTEXT_V01: (
            LAG1_AGENT_TRANSITION_CONTEXT_COLUMNS
        ),
        STATE_CANDIDATE_LAG2_AGENT_TRANSITION_CONTEXT_V01: (
            LAG2_AGENT_TRANSITION_CONTEXT_COLUMNS
        ),
        STATE_CANDIDATE_LAG3_AGENT_TRANSITION_CONTEXT_V01: (
            LAG3_AGENT_TRANSITION_CONTEXT_COLUMNS
        ),
    }.get(candidate, ())


def build_internal_state_features_v01(internal_state: dict) -> dict:
    """Convert internal state into numeric state-vector features."""

    source = internal_state or {}
    position_side = source.get("position_side")
    side_features = _position_side_features(position_side)
    cooldown_remaining = _nonnegative_float_or_zero(
        source.get("cooldown_remaining_steps")
    )
    cooldown_length = _positive_float_or_zero(source.get("cooldown_length_steps"))
    if cooldown_length > 0.0:
        cooldown_norm = min(cooldown_remaining, cooldown_length) / cooldown_length
    else:
        cooldown_norm = 0.0
    cooldown_active = 1.0 if cooldown_remaining > 0.0 else 0.0

    return {
        **side_features,
        "position_size": _float_or_nan(source.get("position_size")),
        "holding_time_seconds": _float_or_nan(
            source.get("holding_time_seconds")
        ),
        "entry_price_distance": _float_or_nan(source.get("entry_price_distance")),
        "unrealized_return": _float_or_nan(source.get("unrealized_return")),
        "cooldown_active": cooldown_active,
        "cooldown_remaining_steps_norm": cooldown_norm,
    }


def build_lag1_agent_transition_context_features_v01(
    previous_decision_position: str | None,
    previous_effective_action: str | None,
) -> dict[str, float]:
    """Encode an all-valid or all-unavailable same-session Lag-1 context."""

    encoded = _encode_agent_transition_context_v01(
        previous_decision_position,
        previous_effective_action,
        label="Lag-1",
    )
    return dict(zip(LAG1_AGENT_TRANSITION_CONTEXT_COLUMNS, encoded))


def build_agent_transition_history_features_v01(
    history: list[object] | tuple[object, ...] | None,
    *,
    state_candidate_name: str,
) -> dict[str, float]:
    """Encode the shared newest-first history queue for Lag-1/2/3 states."""

    candidate = get_state_candidate_name_v01(state_candidate_name)
    depth = get_agent_transition_history_depth_v01(candidate)
    columns = get_agent_transition_history_columns_v01(candidate)
    if depth <= 0 or len(columns) != 6 * depth:
        raise ValueError("agent transition history candidate is required")
    entries = list(history or ())
    if len(entries) > depth:
        raise ValueError(f"history length must not exceed candidate depth {depth}")
    entries.extend([None] * (depth - len(entries)))
    unavailable_seen = False
    encoded_groups: list[float] = []
    for lag, entry in enumerate(entries, start=1):
        if entry is None:
            position = None
            action = None
        elif isinstance(entry, dict):
            position = entry.get("position_before")
            action = entry.get("effective_action")
        elif isinstance(entry, (tuple, list)) and len(entry) == 2:
            position, action = entry
        else:
            raise TypeError("history entry must be None, mapping, or position/action pair")
        available = position is not None or action is not None
        if unavailable_seen and available:
            raise ValueError("agent transition history availability must be a newest-first prefix")
        group = _encode_agent_transition_context_v01(
            position,
            action,
            label=f"Lag-{lag}",
        )
        unavailable_seen = unavailable_seen or not available
        encoded_groups.extend(group)
    return dict(zip(columns, encoded_groups))


def _encode_agent_transition_context_v01(
    previous_decision_position: object,
    previous_effective_action: object,
    *,
    label: str,
) -> tuple[float, ...]:
    if previous_decision_position is None and previous_effective_action is None:
        return (float("nan"),) * 6
    if previous_decision_position is None or previous_effective_action is None:
        raise ValueError(f"{label} position/action availability must be atomic")
    positions = {
        "Flat": (1.0, 0.0, 0.0),
        "Long": (0.0, 1.0, 0.0),
        "Short": (0.0, 0.0, 1.0),
    }
    actions = {
        "Hold": (1.0, 0.0, 0.0),
        "Buy": (0.0, 1.0, 0.0),
        "Sell": (0.0, 0.0, 1.0),
    }
    if previous_decision_position not in positions:
        raise ValueError(
            "previous_decision_position must be Flat, Long, or Short"
        )
    if previous_effective_action not in actions:
        raise ValueError(
            "previous_effective_action must be Hold, Buy, or Sell"
        )
    return positions[previous_decision_position] + actions[previous_effective_action]


def build_state_vector_v01(
    row: pd.Series,
    internal_state: dict,
    *,
    state_candidate_name: str | None = None,
) -> StateVectorResult:
    """Build a fixed-order state vector from one market row and internal state."""

    market_row = row if isinstance(row, pd.Series) else pd.Series(dtype="float64")
    source_internal_state = internal_state or {}
    internal_features = build_internal_state_features_v01(source_internal_state)
    candidate = get_state_candidate_name_v01(state_candidate_name)

    values: dict[str, float] = {}
    candidate_added_columns = tuple(get_state_candidate_added_columns_v01(candidate))
    market_columns = MARKET_STATE_COLUMNS + candidate_added_columns
    for column in market_columns:
        if column in market_row.index:
            values[column] = _float_or_nan(market_row[column])
        else:
            values[column] = np.nan

    for column in INTERNAL_STATE_COLUMNS:
        values[column] = _float_or_nan(internal_features.get(column))

    history_features: dict[str, float] = {}
    history_depth = get_agent_transition_history_depth_v01(candidate)
    if history_depth:
        history = source_internal_state.get("agent_transition_history")
        if history is None and candidate == STATE_CANDIDATE_LAG1_AGENT_TRANSITION_CONTEXT_V01:
            previous_position = source_internal_state.get("previous_decision_position")
            previous_action = source_internal_state.get("previous_effective_action")
            history = (
                ()
                if previous_position is None and previous_action is None
                else ((previous_position, previous_action),)
            )
        history_features = build_agent_transition_history_features_v01(
            history,
            state_candidate_name=candidate,
        )
        values.update(history_features)

    state_columns = get_state_columns_v01(candidate)
    vector = np.array([values[column] for column in state_columns], dtype="float64")
    valid_state_mask = np.isfinite(vector)
    missing_columns = [
        column for column in state_columns if pd.isna(values[column])
    ]

    output = pd.DataFrame([{column: values[column] for column in state_columns}])
    qc: dict[str, Any] = {
        "state_dim": len(state_columns),
        "missing_count": int(pd.isna(vector).sum()),
        "missing_columns": missing_columns,
        "nonfinite_count": int((~valid_state_mask).sum()),
        "market_feature_count": len(market_columns),
        "internal_feature_count": len(INTERNAL_STATE_COLUMNS)
        + len(history_features),
        "proxy_feature_count": len(PROXY_1570_STATE_COLUMNS)
        + len(candidate_added_columns),
        "state_candidate_name": candidate,
        "state_schema_id": get_state_spec_version_v01(candidate),
        "valid_state_mask": valid_state_mask,
    }
    if history_features:
        history_values = vector[-6 * history_depth:]
        history_mask = valid_state_mask[-6 * history_depth:]
        availability: list[bool] = []
        for lag in range(1, history_depth + 1):
            start = (lag - 1) * 6
            group_values = history_values[start:start + 6]
            group_mask = history_mask[start:start + 6]
            group_available = bool(group_mask.all())
            group_unavailable = bool((~group_mask).all())
            if not (group_available or group_unavailable):
                raise RuntimeError(f"Lag-{lag} history mask must be all-valid or all-false")
            if group_available:
                if not np.isfinite(group_values).all() or not (
                    float(group_values[:3].sum()) == 1.0
                    and float(group_values[3:].sum()) == 1.0
                    and np.isin(group_values, (0.0, 1.0)).all()
                ):
                    raise RuntimeError(
                        f"Lag-{lag} available history must be two finite one-hot groups"
                    )
            elif not np.isnan(group_values).all():
                raise RuntimeError(f"Lag-{lag} unavailable history must be all NaN")
            if availability and not availability[-1] and group_available:
                raise RuntimeError("history availability must be a newest-first prefix")
            availability.append(group_available)
            qc[f"agent_history_prev{lag}_available"] = group_available
            qc[f"agent_history_prev{lag}_unavailable"] = group_unavailable
        qc.update(
            {
                "agent_history_available": availability[0],
                "agent_history_unavailable": not availability[0],
                "agent_history_available_group_count": int(sum(availability)),
                "agent_history_depth": history_depth,
                "agent_history_invariant_violation_count": 0,
            }
        )

    return StateVectorResult(
        df=output,
        state_columns=state_columns,
        state_vector=vector,
        qc=qc,
    )


def _position_side_features(position_side) -> dict[str, float]:
    if position_side == "Flat":
        return {
            "position_side_flat": 1.0,
            "position_side_long": 0.0,
            "position_side_short": 0.0,
        }
    if position_side == "Long":
        return {
            "position_side_flat": 0.0,
            "position_side_long": 1.0,
            "position_side_short": 0.0,
        }
    if position_side == "Short":
        return {
            "position_side_flat": 0.0,
            "position_side_long": 0.0,
            "position_side_short": 1.0,
        }
    return {
        "position_side_flat": np.nan,
        "position_side_long": np.nan,
        "position_side_short": np.nan,
    }


def _float_or_nan(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _nonnegative_float_or_zero(value) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(numeric) or numeric <= 0.0:
        return 0.0
    return numeric


def _positive_float_or_zero(value) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(numeric) or numeric <= 0.0:
        return 0.0
    return numeric
