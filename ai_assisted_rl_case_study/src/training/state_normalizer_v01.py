"""State-vector normalizer utilities v0.1.

This module normalizes already-built cooldown-aware state vectors. It does not
touch raw dataframes, TradingEnv, ReplayBuffer, TensorFlow graph code, or
training-loop wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from features.state_vector_v01 import STATE_DIM_V01
from features.state_vector_v01 import STATE_CANDIDATE_1357_ADDED_COLUMNS
from features.state_vector_v01 import (
    LAG1_AGENT_TRANSITION_CONTEXT_COLUMNS,
    LAG2_AGENT_TRANSITION_CONTEXT_COLUMNS,
    LAG3_AGENT_TRANSITION_CONTEXT_COLUMNS,
)
from features.state_candidate_features_v01 import (
    SPREAD_LIQUIDITY_STATE_COLUMNS,
    VOLATILITY_MOVEMENT_STATE_COLUMNS,
    WINDOW_900S_STATE_COLUMNS,
)


TRANSFORM_IDENTITY = "identity"
TRANSFORM_CLIP = "clip"
TRANSFORM_LOG1P = "log1p"
CLIP_MIN_V01 = -0.2
CLIP_MAX_V01 = 0.2


IDENTITY_FEATURES_V01 = frozenset(
    {
        "position_side_flat",
        "position_side_long",
        "position_side_short",
        "cooldown_active",
        "cooldown_remaining_steps_norm",
        "is_morning_session",
        "is_afternoon_session",
        "l1_book_imbalance",
        "book_imbalance_top3",
        "book_imbalance_top5",
        *LAG1_AGENT_TRANSITION_CONTEXT_COLUMNS,
        *LAG2_AGENT_TRANSITION_CONTEXT_COLUMNS,
        *LAG3_AGENT_TRANSITION_CONTEXT_COLUMNS,
        "buy_qty_top3_ratio",
        "sell_qty_top3_ratio",
        "buy_qty_top5_ratio",
        "sell_qty_top5_ratio",
        "intraday_range_position",
        "low_volatility_flag_30s",
        "low_volatility_flag_120s",
        "low_volatility_flag_300s",
        "high_volatility_flag_30s",
        "high_volatility_flag_120s",
        "high_volatility_flag_300s",
        "volatility_compression_flag",
        "recent_movement_insufficient_flag",
        "spread_regime_tight",
        "spread_regime_normal",
        "spread_regime_wide",
        "depth_regime_low",
        "depth_regime_normal",
        "depth_regime_high",
    }
)
CLIP_FEATURES_V01 = frozenset(
    {
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
        "buy_distance_top3",
        "sell_distance_top3",
        "buy_distance_top5",
        "sell_distance_top5",
        "mkt_1570_return_30s",
        "mkt_1570_return_120s",
        "mkt_1570_return_300s",
        "mkt_1570_gap_from_prev_close",
        "mkt_1570_gap_from_vwap",
        "entry_price_distance",
        "unrealized_return",
        "relative_spread",
    }
    | {
        feature
        for feature in STATE_CANDIDATE_1357_ADDED_COLUMNS
        if (
            "return" in feature
            or "log_return" in feature
            or "divergence" in feature
            or "proxy_spread" in feature
            or "disagreement" in feature
        )
    }
    | {
        feature
        for feature in VOLATILITY_MOVEMENT_STATE_COLUMNS
        if "flag" not in feature
    }
    | {
        feature
        for feature in SPREAD_LIQUIDITY_STATE_COLUMNS
        if feature
        not in {
            "spread_bps",
            "best_bid_ask_spread_abs",
            "spread_regime_tight",
            "spread_regime_normal",
            "spread_regime_wide",
            "depth_regime_low",
            "depth_regime_normal",
            "depth_regime_high",
        }
    }
    | {
        feature
        for feature in WINDOW_900S_STATE_COLUMNS
        if "volume_intensity" not in feature and "value_intensity" not in feature
    }
)
LOG1P_FEATURES_V01 = frozenset(
    {
        "best_sell_qty",
        "best_buy_qty",
        "spread",
        "volume_delta_30s",
        "volume_delta_120s",
        "volume_delta_300s",
        "value_delta_30s",
        "value_delta_120s",
        "value_delta_300s",
        "volume_intensity_30s",
        "value_intensity_30s",
        "trade_staleness_seconds",
        "quote_staleness_seconds",
        "high_update_age_seconds",
        "low_update_age_seconds",
        "minutes_from_session_start",
        "minutes_to_session_end",
        "mkt_1570_volume_delta_120s",
        "mkt_1570_value_delta_120s",
        "mkt_1570_trade_staleness_seconds",
        "holding_time_seconds",
        "position_size",
    }
    | {
        feature
        for feature in STATE_CANDIDATE_1357_ADDED_COLUMNS
        if (
            "volume_delta" in feature
            or "value_delta" in feature
            or "volume_intensity" in feature
            or "value_intensity" in feature
            or "staleness" in feature
        )
    }
    | {
        feature
        for feature in WINDOW_900S_STATE_COLUMNS
        if "volume_intensity" in feature or "value_intensity" in feature
    }
    | {"spread_bps", "best_bid_ask_spread_abs"}
)


@dataclass
class StateNormalizerSpec:
    feature_name: str
    transform_type: str
    clip_min: float | None
    clip_max: float | None


@dataclass
class FittedStateNormalizerV01:
    state_columns: list[str]
    feature_specs: list[StateNormalizerSpec]
    state_dim: int
    qc: dict[str, Any]


@dataclass
class NormalizedStateResult:
    state_vector: np.ndarray
    valid_state_mask: np.ndarray
    qc: dict[str, Any]


@dataclass
class NormalizedStateBatchResult:
    state_matrix: np.ndarray
    valid_state_mask_matrix: np.ndarray
    qc: dict[str, Any]


def get_state_normalizer_specs_v01(
    state_columns: list[str],
) -> list[StateNormalizerSpec]:
    """Build per-feature normalizer specs from fixed state column names."""

    specs: list[StateNormalizerSpec] = []
    unknown_features: list[str] = []
    for feature_name in list(state_columns):
        if feature_name in IDENTITY_FEATURES_V01:
            specs.append(
                StateNormalizerSpec(
                    feature_name=feature_name,
                    transform_type=TRANSFORM_IDENTITY,
                    clip_min=None,
                    clip_max=None,
                )
            )
        elif feature_name in CLIP_FEATURES_V01:
            specs.append(
                StateNormalizerSpec(
                    feature_name=feature_name,
                    transform_type=TRANSFORM_CLIP,
                    clip_min=CLIP_MIN_V01,
                    clip_max=CLIP_MAX_V01,
                )
            )
        elif feature_name in LOG1P_FEATURES_V01:
            specs.append(
                StateNormalizerSpec(
                    feature_name=feature_name,
                    transform_type=TRANSFORM_LOG1P,
                    clip_min=None,
                    clip_max=None,
                )
            )
        else:
            unknown_features.append(str(feature_name))

    if unknown_features:
        raise ValueError(f"unknown state normalizer features: {unknown_features}")
    return specs


def fit_state_normalizer_v01(
    state_matrix: np.ndarray,
    state_columns: list[str],
    valid_state_mask_matrix: np.ndarray | None = None,
) -> FittedStateNormalizerV01:
    """Fit a v0.1 normalizer spec for a training state matrix."""

    columns = list(state_columns)
    matrix = _validate_state_matrix(state_matrix, expected_state_dim=len(columns))
    if len(columns) != matrix.shape[1]:
        raise ValueError(
            f"state_columns length must be {matrix.shape[1]}, got {len(columns)}"
        )
    specs = get_state_normalizer_specs_v01(columns)
    valid_mask = _validate_or_default_mask_matrix(valid_state_mask_matrix, matrix)
    fit_mask = np.logical_and(valid_mask, ~np.isnan(matrix))

    qc: dict[str, Any] = {
        "state_dim": int(matrix.shape[1]),
        "feature_group_counts": _feature_group_counts_v01(specs),
        "fit_input_rows": int(matrix.shape[0]),
        "fit_nan_count": int(np.isnan(matrix).sum()),
        "fit_valid_count": int(fit_mask.sum()),
        "unknown_feature_count": 0,
    }
    return FittedStateNormalizerV01(
        state_columns=columns,
        feature_specs=specs,
        state_dim=int(matrix.shape[1]),
        qc=qc,
    )


def transform_state_vector_v01(
    state_vector: np.ndarray,
    fitted: FittedStateNormalizerV01,
    valid_state_mask: np.ndarray | None = None,
) -> NormalizedStateResult:
    """Transform one state vector using a fitted v0.1 normalizer."""

    vector = _validate_state_vector(state_vector, fitted.state_dim)
    mask = _validate_or_default_mask_vector(valid_state_mask, vector)
    matrix_result = _transform_matrix_v01(vector.reshape(1, -1), mask.reshape(1, -1), fitted)
    output_vector = matrix_result["state_matrix"][0]
    output_mask = mask.copy()
    qc = {
        "state_dim": int(fitted.state_dim),
        "nan_count_before": matrix_result["nan_count_before"],
        "nan_count_after": matrix_result["nan_count_after"],
        "inf_count_before": matrix_result["inf_count_before"],
        "inf_count_after": matrix_result["inf_count_after"],
        "clip_applied_count": matrix_result["clip_applied_count"],
        "log1p_applied_count": matrix_result["log1p_applied_count"],
        "negative_to_zero_count": matrix_result["negative_to_zero_count"],
    }
    return NormalizedStateResult(
        state_vector=output_vector,
        valid_state_mask=output_mask,
        qc=qc,
    )


def transform_state_batch_v01(
    state_matrix: np.ndarray,
    fitted: FittedStateNormalizerV01,
    valid_state_mask_matrix: np.ndarray | None = None,
) -> NormalizedStateBatchResult:
    """Transform a batch of state vectors using a fitted v0.1 normalizer."""

    matrix = _validate_state_matrix(state_matrix, expected_state_dim=fitted.state_dim)
    mask = _validate_or_default_mask_matrix(valid_state_mask_matrix, matrix)
    transform_result = _transform_matrix_v01(matrix, mask, fitted)
    qc = {
        "batch_size": int(matrix.shape[0]),
        "state_dim": int(fitted.state_dim),
        "nan_count_before": transform_result["nan_count_before"],
        "nan_count_after": transform_result["nan_count_after"],
        "inf_count_before": transform_result["inf_count_before"],
        "inf_count_after": transform_result["inf_count_after"],
        "clip_applied_count": transform_result["clip_applied_count"],
        "log1p_applied_count": transform_result["log1p_applied_count"],
        "negative_to_zero_count": transform_result["negative_to_zero_count"],
    }
    return NormalizedStateBatchResult(
        state_matrix=transform_result["state_matrix"],
        valid_state_mask_matrix=mask.copy(),
        qc=qc,
    )


def summarize_normalized_state_batch_v01(
    state_matrix: np.ndarray,
    valid_state_mask_matrix: np.ndarray,
) -> dict:
    """Return lightweight diagnostics for a normalized state batch."""

    matrix = _validate_state_matrix(state_matrix)
    mask = _validate_or_default_mask_matrix(valid_state_mask_matrix, matrix)
    return {
        "batch_size": int(matrix.shape[0]),
        "state_dim": int(matrix.shape[1]),
        "nan_count": int(np.isnan(matrix).sum()),
        "inf_count": int(np.isinf(matrix).sum()),
        "valid_count": int(mask.sum()),
        "finite_valid_count": int(np.logical_and(mask, np.isfinite(matrix)).sum()),
    }


def _transform_matrix_v01(
    matrix: np.ndarray,
    valid_mask: np.ndarray,
    fitted: FittedStateNormalizerV01,
) -> dict[str, Any]:
    if matrix.shape[1] != fitted.state_dim:
        raise ValueError(
            f"state_dim must be {fitted.state_dim}, got {matrix.shape[1]}"
        )
    if len(fitted.feature_specs) != fitted.state_dim:
        raise ValueError("feature_specs length must match fitted.state_dim")

    output = matrix.astype(np.float64, copy=True)
    nan_count_before = int(np.isnan(output).sum())
    inf_count_before = int(np.isinf(output).sum())
    clip_applied_count = 0
    log1p_applied_count = 0
    negative_to_zero_count = 0

    invalid_positions = np.logical_or(~valid_mask, np.isnan(output))
    output[invalid_positions] = np.nan

    for column_index, spec in enumerate(fitted.feature_specs):
        column = output[:, column_index]
        valid_values = np.logical_and(valid_mask[:, column_index], np.isfinite(column))
        if spec.transform_type == TRANSFORM_IDENTITY:
            continue
        if spec.transform_type == TRANSFORM_CLIP:
            before = column.copy()
            clipped = np.clip(column[valid_values], spec.clip_min, spec.clip_max)
            column[valid_values] = clipped
            clip_applied_count += int(np.count_nonzero(before[valid_values] != clipped))
            continue
        if spec.transform_type == TRANSFORM_LOG1P:
            values = column[valid_values]
            negative_mask = values < 0.0
            negative_to_zero_count += int(np.count_nonzero(negative_mask))
            nonnegative_values = np.maximum(values, 0.0)
            column[valid_values] = np.log1p(nonnegative_values)
            log1p_applied_count += int(values.size)
            continue
        raise ValueError(f"invalid transform_type: {spec.transform_type!r}")

    return {
        "state_matrix": output,
        "nan_count_before": nan_count_before,
        "nan_count_after": int(np.isnan(output).sum()),
        "inf_count_before": inf_count_before,
        "inf_count_after": int(np.isinf(output).sum()),
        "clip_applied_count": clip_applied_count,
        "log1p_applied_count": log1p_applied_count,
        "negative_to_zero_count": negative_to_zero_count,
    }


def _feature_group_counts_v01(specs: list[StateNormalizerSpec]) -> dict[str, int]:
    counts = {
        TRANSFORM_IDENTITY: 0,
        TRANSFORM_CLIP: 0,
        TRANSFORM_LOG1P: 0,
    }
    for spec in specs:
        counts[spec.transform_type] = counts.get(spec.transform_type, 0) + 1
    return counts


def _validate_state_matrix(
    state_matrix: np.ndarray,
    expected_state_dim: int | None = None,
) -> np.ndarray:
    matrix = np.asarray(state_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"state_matrix must be rank 2, got {matrix.ndim}")
    state_dim = STATE_DIM_V01 if expected_state_dim is None else int(expected_state_dim)
    if matrix.shape[1] != state_dim:
        raise ValueError(f"state_matrix state_dim must be {state_dim}, got {matrix.shape[1]}")
    return matrix


def _validate_state_vector(
    state_vector: np.ndarray,
    expected_state_dim: int,
) -> np.ndarray:
    vector = np.asarray(state_vector, dtype=np.float64)
    if vector.shape != (int(expected_state_dim),):
        raise ValueError(
            f"state_vector must have shape ({int(expected_state_dim)},), got {vector.shape}"
        )
    return vector


def _validate_or_default_mask_matrix(
    valid_state_mask_matrix: np.ndarray | None,
    state_matrix: np.ndarray,
) -> np.ndarray:
    if valid_state_mask_matrix is None:
        return ~np.isnan(state_matrix)
    mask = np.asarray(valid_state_mask_matrix, dtype=bool)
    if mask.shape != state_matrix.shape:
        raise ValueError(
            "valid_state_mask_matrix shape must match state_matrix, "
            f"got {mask.shape} != {state_matrix.shape}"
        )
    return mask


def _validate_or_default_mask_vector(
    valid_state_mask: np.ndarray | None,
    state_vector: np.ndarray,
) -> np.ndarray:
    if valid_state_mask is None:
        return ~np.isnan(state_vector)
    mask = np.asarray(valid_state_mask, dtype=bool)
    if mask.shape != state_vector.shape:
        raise ValueError(
            f"valid_state_mask shape must match state_vector, got {mask.shape} != {state_vector.shape}"
        )
    return mask
