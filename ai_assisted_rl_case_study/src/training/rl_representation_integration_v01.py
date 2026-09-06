"""Strict, versioned B0/R1/A1 state, Q-input, and readiness contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from features.state_vector_v01 import (
    STATE_CANDIDATE_LAG1_AGENT_TRANSITION_CONTEXT_V01,
    STATE_CANDIDATE_LAG2_AGENT_TRANSITION_CONTEXT_V01,
    STATE_CANDIDATE_LAG3_AGENT_TRANSITION_CONTEXT_V01,
    STATE_SPEC_VERSION_V01,
)


READINESS_SCHEMA_ID = "rl_representation_w300_readiness_v01"
OBSERVABILITY_SCHEMA_ID = "rl_representation_e0_observability_v01"
R1_STATE_SCHEMA_ID = "rl_representation_r1_state_105_v01"
R1_Q_INPUT_SCHEMA_ID = "rl_representation_r1_q_input_114_v01"
A1_STATE_SCHEMA_ID = "rl_representation_a1_state_155_v01"
A1_Q_INPUT_SCHEMA_ID = "rl_representation_a1_q_input_214_v01"
B0_Q_INPUT_MODE = "masked_state_plus_valid_state_mask"
LAG1_STATE_SCHEMA_ID = (
    f"{STATE_SPEC_VERSION_V01}__"
    f"{STATE_CANDIDATE_LAG1_AGENT_TRANSITION_CONTEXT_V01}"
)
LAG1_Q_INPUT_SCHEMA_ID = "lag1_agent_transition_context_q_input_130_v01"
LAG2_STATE_SCHEMA_ID = (
    f"{STATE_SPEC_VERSION_V01}__"
    f"{STATE_CANDIDATE_LAG2_AGENT_TRANSITION_CONTEXT_V01}"
)
LAG2_Q_INPUT_SCHEMA_ID = "lag2_agent_transition_context_q_input_142_v01"
LAG3_STATE_SCHEMA_ID = (
    f"{STATE_SPEC_VERSION_V01}__"
    f"{STATE_CANDIDATE_LAG3_AGENT_TRANSITION_CONTEXT_V01}"
)
LAG3_Q_INPUT_SCHEMA_ID = "lag3_agent_transition_context_q_input_154_v01"

EMBEDDING_DIM = 32
EMBEDDING_STACK_DIM = 96
CURRENT_MARKET_DIM = 50
CURRENT_INTERNAL_DIM = 9
CURRENT_STATE_DIM = 59
LAG1_STATE_DIM = 65
LAG2_STATE_DIM = 71
LAG3_STATE_DIM = 77
ACTION_DIM = 3


class RepresentationVariantV01(str, Enum):
    B0 = "B0"
    LAG1 = "LAG1"
    LAG2 = "LAG2"
    LAG3 = "LAG3"
    R1 = "R1"
    A1 = "A1"


@dataclass(frozen=True)
class RepresentationContractV01:
    variant: RepresentationVariantV01
    state_schema_id: str
    q_input_schema_id: str
    state_dim: int
    valid_mask_dim: int
    q_input_dim: int
    state_layout: tuple[str, ...]
    valid_mask_layout: tuple[str, ...]
    q_input_layout: tuple[str, ...]


CONTRACTS_V01: Mapping[RepresentationVariantV01, RepresentationContractV01] = {
    RepresentationVariantV01.B0: RepresentationContractV01(
        RepresentationVariantV01.B0,
        STATE_SPEC_VERSION_V01,
        B0_Q_INPUT_MODE,
        59,
        59,
        118,
        ("current_market_state[50]", "current_internal_state[9]"),
        ("current_state_valid_mask[59]",),
        ("normalized_current_state[59]", "current_state_valid_mask_float[59]"),
    ),
    RepresentationVariantV01.LAG1: RepresentationContractV01(
        RepresentationVariantV01.LAG1,
        LAG1_STATE_SCHEMA_ID,
        LAG1_Q_INPUT_SCHEMA_ID,
        65,
        65,
        130,
        (
            "current_market_state[50]",
            "current_internal_state[9]",
            "lag1_agent_transition_context[6]",
        ),
        ("candidate_state_valid_mask[65]",),
        (
            "normalized_candidate_state[65]",
            "candidate_state_valid_mask_float[65]",
        ),
    ),
    RepresentationVariantV01.LAG2: RepresentationContractV01(
        RepresentationVariantV01.LAG2,
        LAG2_STATE_SCHEMA_ID,
        LAG2_Q_INPUT_SCHEMA_ID,
        71,
        71,
        142,
        (
            "current_market_state[50]",
            "current_internal_state[9]",
            "lag1_agent_transition_context[6]",
            "lag2_agent_transition_context[6]",
        ),
        ("candidate_state_valid_mask[71]",),
        (
            "normalized_candidate_state[71]",
            "candidate_state_valid_mask_float[71]",
        ),
    ),
    RepresentationVariantV01.LAG3: RepresentationContractV01(
        RepresentationVariantV01.LAG3,
        LAG3_STATE_SCHEMA_ID,
        LAG3_Q_INPUT_SCHEMA_ID,
        77,
        77,
        154,
        (
            "current_market_state[50]",
            "current_internal_state[9]",
            "lag1_agent_transition_context[6]",
            "lag2_agent_transition_context[6]",
            "lag3_agent_transition_context[6]",
        ),
        ("candidate_state_valid_mask[77]",),
        (
            "normalized_candidate_state[77]",
            "candidate_state_valid_mask_float[77]",
        ),
    ),
    RepresentationVariantV01.R1: RepresentationContractV01(
        RepresentationVariantV01.R1,
        R1_STATE_SCHEMA_ID,
        R1_Q_INPUT_SCHEMA_ID,
        105,
        9,
        114,
        (
            "board_embedding[32]",
            "trade_embedding[32]",
            "market_embedding[32]",
            "current_internal_state[9]",
        ),
        ("current_internal_state_valid_mask[9]",),
        (
            "board_embedding[32]",
            "trade_embedding[32]",
            "market_embedding[32]",
            "normalized_current_internal_state[9]",
            "current_internal_state_valid_mask_float[9]",
        ),
    ),
    RepresentationVariantV01.A1: RepresentationContractV01(
        RepresentationVariantV01.A1,
        A1_STATE_SCHEMA_ID,
        A1_Q_INPUT_SCHEMA_ID,
        155,
        59,
        214,
        (
            "current_market_state[50]",
            "board_embedding[32]",
            "trade_embedding[32]",
            "market_embedding[32]",
            "current_internal_state[9]",
        ),
        ("current_state_valid_mask[59]",),
        (
            "normalized_current_state[59]",
            "current_state_valid_mask_float[59]",
            "board_embedding[32]",
            "trade_embedding[32]",
            "market_embedding[32]",
        ),
    ),
}


def contract_for_variant_v01(
    variant: RepresentationVariantV01 | str,
) -> RepresentationContractV01:
    if isinstance(variant, RepresentationVariantV01):
        return CONTRACTS_V01[variant]
    try:
        key = RepresentationVariantV01(str(variant))
    except ValueError as exc:
        raise ValueError(f"unknown representation variant: {variant!r}") from exc
    return CONTRACTS_V01[key]


def build_semantic_state_v01(
    variant: RepresentationVariantV01 | str,
    *,
    current_state: np.ndarray,
    board_embedding: np.ndarray | None = None,
    trade_embedding: np.ndarray | None = None,
    market_embedding: np.ndarray | None = None,
) -> np.ndarray:
    """Construct one semantic state in the exact approved order."""

    contract = contract_for_variant_v01(variant)
    direct_state_variants = {
        RepresentationVariantV01.B0,
        RepresentationVariantV01.LAG1,
        RepresentationVariantV01.LAG2,
        RepresentationVariantV01.LAG3,
    }
    current_dim = (
        contract.state_dim
        if contract.variant in direct_state_variants
        else CURRENT_STATE_DIM
    )
    current = _vector(current_state, current_dim, "current_state")
    if contract.variant in {
        RepresentationVariantV01.B0,
        RepresentationVariantV01.LAG1,
        RepresentationVariantV01.LAG2,
        RepresentationVariantV01.LAG3,
    }:
        return _freeze(current.astype(np.float32, copy=True))
    embeddings = _embedding_stack(
        board_embedding,
        trade_embedding,
        market_embedding,
    )
    market = current[:CURRENT_MARKET_DIM]
    internal = current[CURRENT_MARKET_DIM:]
    if contract.variant is RepresentationVariantV01.R1:
        result = np.concatenate([embeddings, internal])
    else:
        result = np.concatenate([market, embeddings, internal])
    if result.shape != (contract.state_dim,):
        raise RuntimeError("constructed semantic state dimension mismatch")
    return _freeze(result.astype(np.float32, copy=False))


def build_q_input_v01(
    variant: RepresentationVariantV01 | str,
    *,
    semantic_state: np.ndarray,
    valid_state_mask: np.ndarray,
    normalized_current_state: np.ndarray,
) -> np.ndarray:
    """Build the strict approved Q input without padding or truncation."""

    contract = contract_for_variant_v01(variant)
    state = _vector(semantic_state, contract.state_dim, "semantic_state")
    mask = _bool_vector(valid_state_mask, contract.valid_mask_dim, "valid_state_mask")
    normalized_dim = (
        9
        if contract.variant is RepresentationVariantV01.R1
        else contract.state_dim
        if contract.variant in {
            RepresentationVariantV01.LAG1,
            RepresentationVariantV01.LAG2,
            RepresentationVariantV01.LAG3,
        }
        else CURRENT_STATE_DIM
    )
    normalized = _vector(
        normalized_current_state,
        normalized_dim,
        "normalized_current_state",
    )
    effective = mask & np.isfinite(normalized)
    masked_normalized = np.where(effective, normalized, 0.0).astype(np.float32)
    mask_features = effective.astype(np.float32)
    if contract.variant in {
        RepresentationVariantV01.B0,
        RepresentationVariantV01.LAG1,
        RepresentationVariantV01.LAG2,
        RepresentationVariantV01.LAG3,
    }:
        result = np.concatenate([masked_normalized, mask_features])
    elif contract.variant is RepresentationVariantV01.R1:
        embeddings = _require_finite_embedding_stack(state[:EMBEDDING_STACK_DIM])
        result = np.concatenate([embeddings, masked_normalized, mask_features])
    else:
        embeddings = _require_finite_embedding_stack(
            state[CURRENT_MARKET_DIM:CURRENT_MARKET_DIM + EMBEDDING_STACK_DIM]
        )
        result = np.concatenate([masked_normalized, mask_features, embeddings])
    if result.shape != (contract.q_input_dim,):
        raise RuntimeError("constructed Q input dimension mismatch")
    if not np.isfinite(result).all():
        raise RuntimeError("Q input finite gate failed")
    return _freeze(result.astype(np.float32, copy=False))


def build_q_input_batch_v01(
    variant: RepresentationVariantV01 | str,
    *,
    semantic_states: np.ndarray,
    valid_state_masks: np.ndarray,
    normalized_current_states: np.ndarray,
) -> np.ndarray:
    """Batch form of the strict adapter for Replay/online/target parity."""

    contract = contract_for_variant_v01(variant)
    states = np.asarray(semantic_states)
    masks = np.asarray(valid_state_masks)
    normalized = np.asarray(normalized_current_states)
    normalized_dim = (
        9
        if contract.variant is RepresentationVariantV01.R1
        else contract.state_dim
        if contract.variant in {
            RepresentationVariantV01.LAG1,
            RepresentationVariantV01.LAG2,
            RepresentationVariantV01.LAG3,
        }
        else CURRENT_STATE_DIM
    )
    if states.ndim != 2 or states.shape[0] == 0 or states.shape[1] != contract.state_dim:
        raise ValueError(f"semantic_states must be [N,{contract.state_dim}]")
    if masks.shape != (states.shape[0], contract.valid_mask_dim) or masks.dtype != np.bool_:
        raise ValueError(
            f"valid_state_masks must be bool [N,{contract.valid_mask_dim}]"
        )
    if normalized.shape != (states.shape[0], normalized_dim):
        raise ValueError(f"normalized_current_states must be [N,{normalized_dim}]")
    result = np.stack(
        [
            build_q_input_v01(
                contract.variant,
                semantic_state=states[index],
                valid_state_mask=masks[index],
                normalized_current_state=normalized[index],
            )
            for index in range(states.shape[0])
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    result.setflags(write=False)
    return result


def build_variant_q_network_v01(variant: RepresentationVariantV01 | str):
    """Use the active Q builder with only the approved input dimension changed."""

    from training.q_network_v01 import build_q_network_v01

    contract = contract_for_variant_v01(variant)
    return build_q_network_v01(
        input_dim=contract.q_input_dim,
        hidden_dim=128,
        num_actions=ACTION_DIM,
        name=f"q_network_{contract.variant.value.lower()}_v01",
    )


def expected_q_parameter_count_v01(variant: RepresentationVariantV01 | str) -> int:
    contract = contract_for_variant_v01(variant)
    return (contract.q_input_dim + 1) * 128 + (128 + 1) * 128 + (128 + 1) * 3


@dataclass(frozen=True)
class RepresentationReadinessInputV01:
    timestamp: pd.Timestamp | str
    symbol: str
    trading_date: str
    session: str
    session_start: pd.Timestamp | str
    session_end: pd.Timestamp | str
    rl_grid_timestamps: Sequence[pd.Timestamp | str]
    board_history_timestamps: Sequence[pd.Timestamp | str]
    trade_history_timestamps: Sequence[pd.Timestamp | str]
    market_history_timestamps: Sequence[pd.Timestamp | str]
    board_semantic_masks: Sequence[np.ndarray]
    trade_semantic_masks: Sequence[np.ndarray]
    market_semantic_masks: Sequence[np.ndarray]
    target_identity_valid: bool
    market_1570_identity_valid: bool
    board_cache_identity_valid: bool


@dataclass(frozen=True)
class RepresentationReadinessResultV01:
    ready: bool
    timestamp: str
    symbol: str
    trading_date: str
    session: str
    reasons: tuple[str, ...]
    schema_id: str = READINESS_SCHEMA_ID


def evaluate_representation_readiness_v01(
    value: RepresentationReadinessInputV01,
) -> RepresentationReadinessResultV01:
    """Evaluate the one common B0/R1/A1 W300 policy-population predicate."""

    if not isinstance(value, RepresentationReadinessInputV01):
        raise TypeError("value must be RepresentationReadinessInputV01")
    t = _timestamp(value.timestamp, "timestamp")
    start = _timestamp(value.session_start, "session_start")
    end = _timestamp(value.session_end, "session_end")
    reasons: list[str] = []
    if not value.symbol or not value.trading_date or not value.session:
        reasons.append("invalid_symbol_date_session_identity")
    if t.date().isoformat() != str(value.trading_date):
        reasons.append("timestamp_trading_date_mismatch")
    if not start <= t < end:
        reasons.append("timestamp_outside_session")
    grid = {_timestamp(item, "rl_grid_timestamp") for item in value.rl_grid_timestamps}
    if t not in grid:
        reasons.append("timestamp_missing_from_actual_rl_grid")
    if t < start + pd.Timedelta(seconds=300):
        reasons.append("before_session_start_plus_300_seconds")
    _check_exact_history(
        reasons,
        name="board",
        observed=value.board_history_timestamps,
        expected=pd.date_range(end=t, periods=12, freq="5s"),
        start=start,
        end=end,
    )
    _check_exact_history(
        reasons,
        name="trade",
        observed=value.trade_history_timestamps,
        expected=pd.date_range(end=t, periods=60, freq="1s"),
        start=start,
        end=end,
    )
    _check_exact_history(
        reasons,
        name="market",
        observed=value.market_history_timestamps,
        expected=pd.date_range(end=t, periods=60, freq="5s"),
        start=start,
        end=end,
    )
    if not bool(value.target_identity_valid):
        reasons.append("target_identity_invalid")
    if not bool(value.market_1570_identity_valid):
        reasons.append("market_1570_identity_invalid")
    if not bool(value.board_cache_identity_valid):
        reasons.append("board_cache_identity_invalid")
    for family, masks in (
        ("board", value.board_semantic_masks),
        ("trade", value.trade_semantic_masks),
        ("market", value.market_semantic_masks),
    ):
        if not _any_semantically_valid(masks):
            reasons.append(f"{family}_complete_semantic_mask_all_false")
    return RepresentationReadinessResultV01(
        ready=not reasons,
        timestamp=t.isoformat(),
        symbol=str(value.symbol),
        trading_date=str(value.trading_date),
        session=str(value.session),
        reasons=tuple(reasons),
    )


def _embedding_stack(
    board: np.ndarray | None,
    trade: np.ndarray | None,
    market: np.ndarray | None,
) -> np.ndarray:
    values = [
        _vector(board, EMBEDDING_DIM, "board_embedding"),
        _vector(trade, EMBEDDING_DIM, "trade_embedding"),
        _vector(market, EMBEDDING_DIM, "market_embedding"),
    ]
    return _require_finite_embedding_stack(np.concatenate(values))


def _require_finite_embedding_stack(value: np.ndarray) -> np.ndarray:
    array = _vector(value, EMBEDDING_STACK_DIM, "embedding_stack")
    if not np.isfinite(array).all():
        raise RuntimeError("encoder output finite gate failed")
    return array.astype(np.float32, copy=False)


def _vector(value: np.ndarray | None, dim: int, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} is required")
    array = np.asarray(value)
    if array.shape != (dim,):
        raise ValueError(f"{name} shape must be ({dim},), got {array.shape}")
    if array.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"{name} must be numeric")
    return array.astype(np.float32, copy=False)


def _bool_vector(value: np.ndarray, dim: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (dim,) or array.dtype != np.bool_:
        raise ValueError(f"{name} must be bool shape ({dim},), got {array.shape}/{array.dtype}")
    return array


def _timestamp(value: pd.Timestamp | str, name: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tz is None:
        raise ValueError(f"{name} must be timezone-aware")
    return result


def _check_exact_history(
    reasons: list[str],
    *,
    name: str,
    observed: Sequence[pd.Timestamp | str],
    expected: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    actual = tuple(_timestamp(item, f"{name}_history_timestamp") for item in observed)
    expected_tuple = tuple(expected)
    if actual != expected_tuple:
        reasons.append(f"{name}_exact_causal_history_unavailable")
        return
    if any(item < start or item >= end or item.date() != start.date() for item in actual):
        reasons.append(f"{name}_history_crosses_session_or_date")


def _any_semantically_valid(masks: Sequence[np.ndarray]) -> bool:
    if not masks:
        return False
    return any(bool(np.any(np.asarray(mask, dtype=bool))) for mask in masks)


def _freeze(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


__all__ = [
    "A1_Q_INPUT_SCHEMA_ID",
    "A1_STATE_SCHEMA_ID",
    "CONTRACTS_V01",
    "OBSERVABILITY_SCHEMA_ID",
    "READINESS_SCHEMA_ID",
    "LAG1_Q_INPUT_SCHEMA_ID",
    "LAG1_STATE_SCHEMA_ID",
    "LAG2_Q_INPUT_SCHEMA_ID",
    "LAG2_STATE_SCHEMA_ID",
    "LAG3_Q_INPUT_SCHEMA_ID",
    "LAG3_STATE_SCHEMA_ID",
    "R1_Q_INPUT_SCHEMA_ID",
    "R1_STATE_SCHEMA_ID",
    "RepresentationContractV01",
    "RepresentationReadinessInputV01",
    "RepresentationReadinessResultV01",
    "RepresentationVariantV01",
    "build_q_input_v01",
    "build_q_input_batch_v01",
    "build_semantic_state_v01",
    "build_variant_q_network_v01",
    "contract_for_variant_v01",
    "evaluate_representation_readiness_v01",
    "expected_q_parameter_count_v01",
]
