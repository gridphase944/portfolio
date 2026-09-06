"""Current latency1s plus closing-risk execution contract.

Delayed/recovery/auction observations are kept in an execution-only sidecar;
they never become policy-visible episode columns.  This module is the single
canonical primitive used by current sequential, evaluation, and parallel
runtime routes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from features.timestamp_preprocess_v01 import (
    build_row_timestamp_raw,
    filter_timestamp_reverse_rows,
)
from trading_env.execution_latency_v01 import (
    TOP5_COLUMNS,
    TRADING_ENV_KWARGS_ATTR,
    canonical_sha256_v01,
    file_sha256_v01,
    latency_time_band_v01,
    normalize_timestamp_jst_v01,
    session_for_timestamp_v01,
)
from trading_env.reward_v01 import (
    NAN,
    ORDER_QTY,
    OrderBookLevel,
    StepRewardResult,
    compute_reward,
    compute_top5_weighted_price,
    step_reward_v01,
    validate_reference_notional,
)


EXECUTION_CONTRACT_ID = "fixed_latency1s_closingrisk_v02"
ENVIRONMENT_CONTRACT_ID = "trading_env_execution_latency1s_closingrisk_v02"
REWARD_CLOCK_CONTRACT_ID = "liquidation_equity_effective_clock_closingrisk_v02"
TRANSITION_CONTRACT_ID = "rl_transition_execution_latency1s_closingrisk_terminal_v02"
TIMESTAMP_PREPROCESSING_ID = "canonical_reverse_row_latest_prior_v01"
ENTRY_CUTOFF_GATE_ID = "closing_entry_cutoff_gate_v01"
STAGE1_CONTRACT_ID = "forced_liquidation_top5_152001_v01"
STAGE2_CONTRACT_ID = "forced_liquidation_zaraba_first_executable_v01"
STAGE3_CONTRACT_ID = "closing_auction_execution_v01"
CLOSING_AUCTION_ELIGIBILITY_ID = "strict_closing_auction_eligibility_v01"
TARGET_ACTION_MASK_SOURCE = "policy_effective"
SIDECAR_SCHEMA_ID = "execution_only_sidecar_latency1s_closingrisk_v02"
SIDECAR_MANIFEST_SCHEMA_ID = "execution_only_sidecar_manifest_closingrisk_v02"
POPULATION_PREFLIGHT_SCHEMA_ID = "execution_closingrisk_population_preflight_v02"

TOKYO_TZ = "Asia/Tokyo"
LATENCY_SECONDS = 1
ENTRY_CUTOFF_TIME = time(15, 15, 0)
AGENT_TRADING_END_TIME = time(15, 20, 0)
STAGE1_TARGET_TIME = time(15, 20, 1)
ZARABA_END_TIME = time(15, 25, 0)
CLOSING_AUCTION_TIME = time(15, 30, 0)

RAW_REQUIRED_COLUMNS = (
    "Symbol",
    "BidTime",
    "AskTime",
    "CurrentPrice",
    "CurrentPriceTime",
    "TradingVolume",
    "TradingVolumeTime",
) + TOP5_COLUMNS

ROUTING_FAILURE_REASONS = (
    "execution_snapshot_missing",
    "execution_identity_mismatch",
    "execution_session_mismatch",
    "execution_date_mismatch",
    "execution_timestamp_after_target",
)


class ClosingRiskSidecarError(RuntimeError):
    """Structured fail-closed build/selection error."""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}:{detail}")
        self.reason = str(reason)
        self.detail = str(detail)


@dataclass(frozen=True)
class ExecutionClosingRiskSidecarV02:
    rows: pd.DataFrame
    symbol: str
    trading_date: str
    decision_identity_sha256: str
    artifact_identity_sha256: str
    execution_contract_id: str = EXECUTION_CONTRACT_ID
    environment_contract_id: str = ENVIRONMENT_CONTRACT_ID
    reward_clock_contract_id: str = REWARD_CLOCK_CONTRACT_ID
    transition_contract_id: str = TRANSITION_CONTRACT_ID
    timestamp_preprocessing_id: str = TIMESTAMP_PREPROCESSING_ID
    entry_cutoff_gate_id: str = ENTRY_CUTOFF_GATE_ID
    agent_trading_end_jst: str = AGENT_TRADING_END_TIME.isoformat()
    stage1_contract_id: str = STAGE1_CONTRACT_ID
    stage2_contract_id: str = STAGE2_CONTRACT_ID
    stage3_contract_id: str = STAGE3_CONTRACT_ID
    closing_auction_eligibility_id: str = CLOSING_AUCTION_ELIGIBILITY_ID
    target_action_mask_source: str = TARGET_ACTION_MASK_SOURCE
    schema_id: str = SIDECAR_SCHEMA_ID

    def __len__(self) -> int:
        return len(self.rows)

    def payload_at(self, index: int) -> dict[str, Any]:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("execution sidecar index must be an integer")
        index = int(index)
        if index < 0 or index >= len(self.rows):
            raise IndexError("execution sidecar index out of range")
        payload = self.rows.iloc[index].to_dict()
        payload.update(
            {
                "schema_id": self.schema_id,
                "execution_contract_id": self.execution_contract_id,
                "environment_contract_id": self.environment_contract_id,
                "reward_clock_contract_id": self.reward_clock_contract_id,
                "transition_contract_id": self.transition_contract_id,
                "timestamp_preprocessing_id": self.timestamp_preprocessing_id,
                "entry_cutoff_gate_id": self.entry_cutoff_gate_id,
                "agent_trading_end_jst": self.agent_trading_end_jst,
                "stage1_contract_id": self.stage1_contract_id,
                "stage2_contract_id": self.stage2_contract_id,
                "stage3_contract_id": self.stage3_contract_id,
                "closing_auction_eligibility_id": (
                    self.closing_auction_eligibility_id
                ),
                "target_action_mask_source": self.target_action_mask_source,
                "sidecar_symbol": self.symbol,
                "sidecar_trading_date": self.trading_date,
                "sidecar_decision_identity_sha256": self.decision_identity_sha256,
                "sidecar_artifact_identity_sha256": self.artifact_identity_sha256,
            }
        )
        return payload


@dataclass(frozen=True)
class ExecutionClosingRiskBuildResultV02:
    decision_rows: pd.DataFrame
    sidecar: ExecutionClosingRiskSidecarV02
    qc: Mapping[str, Any]


def prepare_closingrisk_decision_rows_v02(
    decision_rows: pd.DataFrame,
) -> pd.DataFrame:
    """End the agent grid at 15:20 and mark exactly one terminal risk row."""

    required = {
        "grid_timestamp",
        "row_timestamp_raw",
        "source_row_order",
        "session",
        "is_forced_exit",
    } | set(TOP5_COLUMNS)
    missing = required - set(decision_rows.columns)
    if missing:
        raise ValueError(f"decision closing-risk columns missing: {sorted(missing)}")
    if decision_rows.empty:
        raise ValueError("decision rows must not be empty")
    timestamps = decision_rows["grid_timestamp"].map(normalize_timestamp_jst_v01)
    dates = {value.date().isoformat() for value in timestamps}
    if len(dates) != 1:
        raise ClosingRiskSidecarError(
            "execution_date_mismatch", f"decision dates={sorted(dates)}"
        )
    keep = timestamps.map(lambda value: value.time() <= AGENT_TRADING_END_TIME)
    result = decision_rows.loc[keep].copy().reset_index(drop=True)
    if result.empty:
        raise ValueError("agent-trading decision population is empty")
    result_times = result["grid_timestamp"].map(normalize_timestamp_jst_v01)
    terminal = result_times.map(lambda value: value.time() == AGENT_TRADING_END_TIME)
    if int(terminal.sum()) != 1 or not bool(terminal.iloc[-1]):
        raise ValueError("current contract requires exactly one final 15:20:00 row")
    result.loc[:, "is_forced_exit"] = False
    result.loc[result.index[-1], "is_forced_exit"] = True
    return result


def decision_population_identity_v02(decision_rows: pd.DataFrame) -> str:
    columns = (
        "grid_timestamp",
        "row_timestamp_raw",
        "source_row_order",
        "session",
        "is_forced_exit",
    )
    missing = set(columns) - set(decision_rows.columns)
    if missing:
        raise ValueError(f"decision identity columns missing: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for row in decision_rows.loc[:, list(columns)].to_dict("records"):
        rows.append(
            {
                "grid_timestamp": normalize_timestamp_jst_v01(
                    row["grid_timestamp"]
                ).isoformat(),
                "row_timestamp_raw": normalize_timestamp_jst_v01(
                    row["row_timestamp_raw"]
                ).isoformat(),
                "source_row_order": int(row["source_row_order"]),
                "session": str(row["session"]),
                "is_forced_exit": bool(row["is_forced_exit"]),
            }
        )
    return canonical_sha256_v01(rows)


def build_execution_closingrisk_sidecar_v02(
    *,
    decision_rows: pd.DataFrame,
    raw_push_rows: pd.DataFrame,
    symbol: str,
    trading_date: str,
) -> ExecutionClosingRiskBuildResultV02:
    """Build normal latency snapshots and the three-stage terminal payload."""

    current_rows = prepare_closingrisk_decision_rows_v02(decision_rows)
    expected_symbol = _canonical_symbol(symbol)
    expected_date = str(trading_date)
    _validate_decision_identity(current_rows, expected_symbol, expected_date)
    missing = set(RAW_REQUIRED_COLUMNS) - set(raw_push_rows.columns)
    if missing:
        raise ValueError(f"raw closing-risk columns missing: {sorted(missing)}")
    _validate_raw_symbol_identity(raw_push_rows, expected_symbol)

    parsed = build_row_timestamp_raw(raw_push_rows)
    prepared, reverse_qc = filter_timestamp_reverse_rows(parsed)
    prepared = prepared.loc[prepared["row_timestamp_raw"].notna()].copy()
    if prepared.empty:
        raise ClosingRiskSidecarError(
            "execution_snapshot_missing", "raw source has no surviving timestamp"
        )
    prepared = prepared.sort_values(
        ["row_timestamp_raw", "row_order"], kind="mergesort"
    ).reset_index(drop=True)
    raw_dates = set(prepared["row_timestamp_raw"].dt.date.astype(str).unique())
    if raw_dates != {expected_date}:
        raise ClosingRiskSidecarError(
            "execution_date_mismatch",
            f"expected={expected_date},observed={sorted(raw_dates)}",
        )
    seconds = (
        prepared["row_timestamp_raw"].dt.hour * 3600
        + prepared["row_timestamp_raw"].dt.minute * 60
        + prepared["row_timestamp_raw"].dt.second
    )
    prepared["execution_session"] = np.select(
        [
            (seconds >= 9 * 3600) & (seconds < 11 * 3600 + 30 * 60),
            (seconds >= 12 * 3600 + 30 * 60) & (seconds < 15 * 3600 + 25 * 60),
        ],
        ["morning", "afternoon"],
        default=None,
    )
    session_sources: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    for session in ("morning", "afternoon"):
        source = prepared.loc[prepared["execution_session"] == session].copy()
        session_sources[session] = (
            source.reset_index(drop=True),
            source["row_timestamp_raw"].astype("int64").to_numpy(
                dtype=np.int64, copy=False
            ),
        )

    closing = identify_strict_closing_auction_v02(
        prepared, trading_date=expected_date
    )
    output_rows: list[dict[str, Any]] = []
    for decision_index, decision in current_rows.iterrows():
        decision_timestamp = normalize_timestamp_jst_v01(
            decision["grid_timestamp"]
        )
        target_timestamp = decision_timestamp + pd.Timedelta(seconds=1)
        session = str(decision["session"])
        if session_for_timestamp_v01(decision_timestamp) != session:
            raise ClosingRiskSidecarError(
                "execution_session_mismatch",
                f"decision index {decision_index} session mismatch",
            )
        if session_for_timestamp_v01(target_timestamp) != session:
            raise ClosingRiskSidecarError(
                "execution_session_mismatch",
                f"decision index {decision_index} target crosses session",
            )
        source_rows, source_ns = session_sources[session]
        selected_position = int(
            np.searchsorted(
                source_ns, np.int64(target_timestamp.value), side="right"
            )
            - 1
        )
        if selected_position < 0:
            raise ClosingRiskSidecarError(
                "execution_snapshot_missing",
                f"decision index {decision_index} has no latest-prior row",
            )
        selected = source_rows.iloc[selected_position]
        source_timestamp = normalize_timestamp_jst_v01(
            selected["row_timestamp_raw"]
        )
        if source_timestamp > target_timestamp:
            raise ClosingRiskSidecarError(
                "execution_timestamp_after_target", f"decision index {decision_index}"
            )
        if source_timestamp.date().isoformat() != expected_date:
            raise ClosingRiskSidecarError(
                "execution_date_mismatch", f"decision index {decision_index}"
            )
        decision_source_timestamp = normalize_timestamp_jst_v01(
            decision["row_timestamp_raw"]
        )
        decision_source_order = int(decision["source_row_order"])
        execution_source_order = int(selected["row_order"])
        same = (
            source_timestamp == decision_source_timestamp
            and execution_source_order == decision_source_order
        )
        row: dict[str, Any] = {
            "decision_index": int(decision_index),
            "decision_timestamp": decision_timestamp,
            "decision_source_timestamp": decision_source_timestamp,
            "decision_source_row_order": decision_source_order,
            "execution_target_timestamp": target_timestamp,
            "execution_source_timestamp": source_timestamp,
            "execution_source_row_order": execution_source_order,
            "execution_snapshot_age_seconds": float(
                (target_timestamp - source_timestamp).total_seconds()
            ),
            "execution_snapshot_advanced": bool(not same),
            "execution_snapshot_same_as_decision": bool(same),
            "execution_symbol": expected_symbol,
            "execution_trading_date": expected_date,
            "execution_session": session,
            "latency_time_band": latency_time_band_v01(decision_timestamp),
            "is_forced_exit": bool(decision["is_forced_exit"]),
        }
        for column in TOP5_COLUMNS:
            row[column] = selected[column]
        row.update(_policy_independent_price_diagnostics(decision, selected))
        _initialize_recovery_fields(row)
        if bool(decision["is_forced_exit"]):
            if decision_timestamp.time() != AGENT_TRADING_END_TIME:
                raise ValueError("forced row is not 15:20:00")
            recovery_source = prepared.loc[
                (prepared["row_timestamp_raw"] > target_timestamp)
                & (
                    prepared["row_timestamp_raw"]
                    < normalize_timestamp_jst_v01(f"{expected_date} 15:25:00")
                )
            ]
            _add_stage2_side(row, recovery_source, "long", "Buy")
            _add_stage2_side(row, recovery_source, "short", "Sell")
            row.update(closing)
        output_rows.append(row)

    sidecar_rows = pd.DataFrame(output_rows)
    decision_identity = decision_population_identity_v02(current_rows)
    artifact_identity = execution_sidecar_artifact_identity_v02(
        rows=sidecar_rows,
        symbol=expected_symbol,
        trading_date=expected_date,
        decision_identity_sha256=decision_identity,
    )
    sidecar = ExecutionClosingRiskSidecarV02(
        rows=sidecar_rows,
        symbol=expected_symbol,
        trading_date=expected_date,
        decision_identity_sha256=decision_identity,
        artifact_identity_sha256=artifact_identity,
    )
    qc = summarize_execution_closingrisk_sidecar_v02(sidecar_rows)
    qc.update(
        {
            "raw_input_row_count": int(len(raw_push_rows)),
            "raw_surviving_row_count": int(len(prepared)),
            "timestamp_reverse_rows_removed": int(
                reverse_qc["timestamp_reverse_rows_removed"]
            ),
            "source_last_timestamp": normalize_timestamp_jst_v01(
                prepared["row_timestamp_raw"].iloc[-1]
            ).isoformat(),
            "decision_identity_sha256": decision_identity,
            "sidecar_artifact_identity_sha256": artifact_identity,
            "closing_auction": closing,
        }
    )
    return ExecutionClosingRiskBuildResultV02(
        decision_rows=current_rows, sidecar=sidecar, qc=qc
    )


def identify_strict_closing_auction_v02(
    prepared_rows: pd.DataFrame, *, trading_date: str
) -> dict[str, Any]:
    """Fail-closed strict 15:30 print extraction for the current contract."""

    closing_timestamp = normalize_timestamp_jst_v01(f"{trading_date} 15:30:00")
    before = prepared_rows.loc[
        prepared_rows["row_timestamp_raw"] < closing_timestamp
    ]
    exact = prepared_rows.loc[
        prepared_rows["row_timestamp_raw"] == closing_timestamp
    ]
    output: dict[str, Any] = {
        "closing_auction_eligible": False,
        "closing_auction_failure_reason": None,
        "closing_auction_price": np.nan,
        "closing_auction_volume_delta": np.nan,
        "closing_auction_matching_row_count": int(len(exact)),
        "closing_auction_eligible_row_count": 0,
        "closing_auction_unique_price_count": 0,
        "closing_auction_source_timestamp": closing_timestamp,
        "closing_auction_source_row_order": np.nan,
        "preclosing_last_trading_volume": np.nan,
    }
    if before.empty:
        output["closing_auction_failure_reason"] = "preclosing_volume_missing"
        return output
    pre_volume = _finite_nonnegative(before.iloc[-1].get("TradingVolume"))
    if pre_volume is None:
        output["closing_auction_failure_reason"] = "preclosing_volume_invalid"
        return output
    output["preclosing_last_trading_volume"] = pre_volume
    if exact.empty:
        output["closing_auction_failure_reason"] = "closing_row_missing"
        return output
    eligible: list[tuple[pd.Series, float, float]] = []
    for _, row in exact.iterrows():
        try:
            price_time = normalize_timestamp_jst_v01(row["CurrentPriceTime"])
            volume_time = normalize_timestamp_jst_v01(row["TradingVolumeTime"])
        except (TypeError, ValueError):
            continue
        price = _finite_positive(row.get("CurrentPrice"))
        volume = _finite_nonnegative(row.get("TradingVolume"))
        if (
            price_time == closing_timestamp
            and volume_time == closing_timestamp
            and price is not None
            and volume is not None
            and volume - pre_volume >= ORDER_QTY
        ):
            eligible.append((row, price, volume - pre_volume))
    output["closing_auction_eligible_row_count"] = len(eligible)
    if not eligible:
        output["closing_auction_failure_reason"] = "closing_strict_conditions_not_met"
        return output
    prices = sorted({float(item[1]) for item in eligible})
    output["closing_auction_unique_price_count"] = len(prices)
    if len(prices) != 1:
        output["closing_auction_failure_reason"] = "closing_price_not_unique"
        return output
    eligible.sort(key=lambda item: int(item[0]["row_order"]))
    selected, price, volume_delta = eligible[-1]
    output.update(
        {
            "closing_auction_eligible": True,
            "closing_auction_failure_reason": None,
            "closing_auction_price": float(price),
            "closing_auction_volume_delta": float(volume_delta),
            "closing_auction_source_row_order": int(selected["row_order"]),
        }
    )
    return output


def validate_execution_payload_v02(
    *, payload: Mapping[str, Any] | None, decision_row: Mapping[str, Any]
) -> str | None:
    if not isinstance(payload, Mapping):
        return "execution_snapshot_missing"
    for field, expected in (
        ("schema_id", SIDECAR_SCHEMA_ID),
        ("execution_contract_id", EXECUTION_CONTRACT_ID),
        ("environment_contract_id", ENVIRONMENT_CONTRACT_ID),
        ("reward_clock_contract_id", REWARD_CLOCK_CONTRACT_ID),
        ("transition_contract_id", TRANSITION_CONTRACT_ID),
        ("timestamp_preprocessing_id", TIMESTAMP_PREPROCESSING_ID),
        ("closing_auction_eligibility_id", CLOSING_AUCTION_ELIGIBILITY_ID),
    ):
        if payload.get(field) != expected:
            return "execution_identity_mismatch"
    try:
        decision = normalize_timestamp_jst_v01(
            decision_row.get("grid_timestamp", decision_row.get("timestamp"))
        )
        payload_decision = normalize_timestamp_jst_v01(
            payload["decision_timestamp"]
        )
        target = normalize_timestamp_jst_v01(
            payload["execution_target_timestamp"]
        )
        source = normalize_timestamp_jst_v01(
            payload["execution_source_timestamp"]
        )
    except (KeyError, TypeError, ValueError):
        return "execution_snapshot_missing"
    if payload_decision != decision or target != decision + pd.Timedelta(seconds=1):
        return "execution_identity_mismatch"
    symbol = _canonical_symbol(decision_row.get("symbol", decision_row.get("Symbol")))
    if (
        _canonical_symbol(payload.get("execution_symbol")) != symbol
        or _canonical_symbol(payload.get("sidecar_symbol")) != symbol
    ):
        return "execution_identity_mismatch"
    date = decision.date().isoformat()
    if any(
        str(value) != date
        for value in (
            payload.get("execution_trading_date"),
            payload.get("sidecar_trading_date"),
            target.date().isoformat(),
            source.date().isoformat(),
        )
    ):
        return "execution_date_mismatch"
    session = str(decision_row.get("session"))
    if (
        str(payload.get("execution_session")) != session
        or session_for_timestamp_v01(decision) != session
        or session_for_timestamp_v01(target) != session
        or session_for_timestamp_v01(source) != session
    ):
        return "execution_session_mismatch"
    if source > target:
        return "execution_timestamp_after_target"
    if any(column not in payload for column in TOP5_COLUMNS):
        return "execution_snapshot_missing"
    return None


def policy_effective_action_mask_v02(
    *,
    env_action_mask: np.ndarray,
    position: str,
    decision_timestamp: Any,
    is_forced_liquidation_event: bool = False,
) -> dict[str, Any]:
    """Apply the outer entry cutoff/risk gate without mutating Env ownership."""

    env_mask = np.asarray(env_action_mask, dtype=bool).reshape(-1)
    if env_mask.shape != (3,):
        raise ValueError("env action mask must use Q order Hold/Buy/Sell")
    timestamp = normalize_timestamp_jst_v01(decision_timestamp)
    clock = timestamp.time()
    effective = env_mask.copy()
    reason: str | None = None
    cutoff_active = ENTRY_CUTOFF_TIME <= clock < AGENT_TRADING_END_TIME
    if is_forced_liquidation_event:
        if clock != AGENT_TRADING_END_TIME:
            raise ValueError("forced-liquidation event must be at 15:20:00")
        effective[:] = False
        effective[{"Flat": 0, "Short": 1, "Long": 2}[str(position)]] = True
        reason = "risk_management_forced_liquidation"
    elif cutoff_active and str(position) == "Flat":
        effective[1:] = False
        reason = "new_entry_cutoff_flat"
    blocked = env_mask & ~effective
    if not np.any(effective):
        raise ValueError("policy-effective action mask must contain a valid action")
    return {
        "env_action_mask": env_mask,
        "policy_effective_action_mask": effective,
        "entry_cutoff_active": bool(cutoff_active),
        "entry_cutoff_gate_id": ENTRY_CUTOFF_GATE_ID,
        "policy_effective_mask_reason": reason,
        "blocked_action_slot_count": int(blocked.sum()),
        "env_valid_action_slot_count": int(env_mask.sum()),
        "is_forced_liquidation_event": bool(is_forced_liquidation_event),
    }


def risk_management_action_v02(position: str) -> str:
    return {"Flat": "Hold", "Long": "Sell", "Short": "Buy"}[str(position)]


def resolve_forced_liquidation_v02(
    *,
    position_before: str,
    cash_pnl_before: float,
    equity_before: float,
    previous_close: Any,
    payload: Mapping[str, Any],
) -> tuple[StepRewardResult, dict[str, Any]]:
    """Resolve one terminal risk event without intermediate agent transitions."""

    base = {
        "agent_trading_end_timestamp": normalize_timestamp_jst_v01(
            payload["decision_timestamp"]
        ).isoformat(),
        "entry_cutoff_active": False,
        "forced_liquidation_started": bool(position_before in {"Long", "Short"}),
        "forced_liquidation_stage": "flat_no_action",
        "forced_liquidation_resolution_timestamp": normalize_timestamp_jst_v01(
            payload["decision_timestamp"]
        ).isoformat(),
        "forced_liquidation_delay_seconds": 0.0,
        "stage1_failure_reason": None,
        "stage2_attempted": False,
        "stage2_snapshot_count": 0,
        "closing_auction_used": False,
        "closing_auction_price": None,
        "closing_auction_volume_delta": None,
        "execution_price_semantic": None,
    }
    stage1_book = _book_from_payload(payload)
    if position_before == "Flat":
        result = step_reward_v01(
            position_before="Flat",
            cash_pnl_before=cash_pnl_before,
            equity_before=equity_before,
            action="Hold",
            order_book=stage1_book,
            previous_close=previous_close,
            is_forced_exit=True,
        )
        return result, base

    result = step_reward_v01(
        position_before=position_before,
        cash_pnl_before=cash_pnl_before,
        equity_before=equity_before,
        action=risk_management_action_v02(position_before),
        order_book=stage1_book,
        previous_close=previous_close,
        is_forced_exit=True,
    )
    if result.executed and result.position_after == "Flat":
        resolution = normalize_timestamp_jst_v01(
            payload["execution_target_timestamp"]
        )
        base.update(
            {
                "forced_liquidation_stage": "stage1_top5",
                "forced_liquidation_resolution_timestamp": resolution.isoformat(),
                "forced_liquidation_delay_seconds": float(
                    (
                        resolution
                        - normalize_timestamp_jst_v01(payload["decision_timestamp"])
                    ).total_seconds()
                ),
                "execution_price_semantic": "top5_weighted_execution_price",
            }
        )
        return result, base

    stage1_reason = str(result.failure_reason or "stage1_top5_failure")
    side = "long" if position_before == "Long" else "short"
    base["stage1_failure_reason"] = stage1_reason
    base["stage2_attempted"] = True
    base["stage2_snapshot_count"] = int(
        _optional_int(payload.get(f"stage2_{side}_snapshot_count")) or 0
    )
    if bool(payload.get(f"stage2_{side}_available", False)):
        stage2_book = _stage2_book(payload, side)
        stage2 = step_reward_v01(
            position_before=position_before,
            cash_pnl_before=cash_pnl_before,
            equity_before=equity_before,
            action=risk_management_action_v02(position_before),
            order_book=stage2_book,
            previous_close=previous_close,
            is_forced_exit=True,
        )
        if stage2.executed and stage2.position_after == "Flat":
            resolution = normalize_timestamp_jst_v01(
                payload[f"stage2_{side}_source_timestamp"]
            )
            base.update(
                {
                    "forced_liquidation_stage": "stage2_zaraba_recovery",
                    "forced_liquidation_resolution_timestamp": resolution.isoformat(),
                    "forced_liquidation_delay_seconds": float(
                        (
                            resolution
                            - normalize_timestamp_jst_v01(
                                payload["decision_timestamp"]
                            )
                        ).total_seconds()
                    ),
                    "execution_price_semantic": "top5_weighted_execution_price",
                }
            )
            return stage2, base

    if bool(payload.get("closing_auction_eligible", False)):
        auction = _closing_auction_reward_result(
            position_before=position_before,
            cash_pnl_before=cash_pnl_before,
            equity_before=equity_before,
            previous_close=previous_close,
            price=payload.get("closing_auction_price"),
        )
        if auction.executed and auction.position_after == "Flat":
            resolution = normalize_timestamp_jst_v01(
                payload["closing_auction_source_timestamp"]
            )
            base.update(
                {
                    "forced_liquidation_stage": "stage3_closing_auction",
                    "forced_liquidation_resolution_timestamp": resolution.isoformat(),
                    "forced_liquidation_delay_seconds": float(
                        (
                            resolution
                            - normalize_timestamp_jst_v01(
                                payload["decision_timestamp"]
                            )
                        ).total_seconds()
                    ),
                    "closing_auction_used": True,
                    "closing_auction_price": float(
                        payload["closing_auction_price"]
                    ),
                    "closing_auction_volume_delta": float(
                        payload["closing_auction_volume_delta"]
                    ),
                    "execution_price_semantic": "closing_auction_execution_price",
                }
            )
            return auction, base

    failure = str(
        payload.get("closing_auction_failure_reason")
        or f"stage2_{side}_not_executable"
    )
    base.update(
        {
            "forced_liquidation_stage": "unresolved",
            "forced_liquidation_resolution_timestamp": None,
            "forced_liquidation_delay_seconds": None,
        }
    )
    return (
        _unresolved_forced_liquidation_result(
            position_before=position_before,
            cash_pnl_before=cash_pnl_before,
            previous_close=previous_close,
            reason=failure,
        ),
        base,
    )


def execution_info_v02(
    *,
    payload: Mapping[str, Any] | None,
    result: StepRewardResult,
    routing_failure: str | None,
    closing_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "execution_contract_id": EXECUTION_CONTRACT_ID,
        "environment_contract_id": ENVIRONMENT_CONTRACT_ID,
        "reward_clock_contract_id": REWARD_CLOCK_CONTRACT_ID,
        "transition_contract_id": TRANSITION_CONTRACT_ID,
        "timestamp_preprocessing_id": TIMESTAMP_PREPROCESSING_ID,
        "entry_cutoff_gate_id": ENTRY_CUTOFF_GATE_ID,
        "target_action_mask_source": TARGET_ACTION_MASK_SOURCE,
        "execution_only_payload_schema_id": SIDECAR_SCHEMA_ID,
        "execution_routing_failed": routing_failure is not None,
        "execution_routing_failure_reason": routing_failure,
        "execution_snapshot_count": int(routing_failure is None),
        "latency_policy_selected_execution": bool(
            result.executed and not result.is_forced_exit
        ),
        "latency_policy_selected_action_side": None,
        "delayed_vs_decision_execution_price_delta_bps": None,
        "delayed_vs_decision_execution_price_delta_sign": None,
    }
    if isinstance(payload, Mapping):
        for field in (
            "execution_target_timestamp",
            "execution_source_timestamp",
        ):
            try:
                info[field] = normalize_timestamp_jst_v01(payload[field]).isoformat()
            except (KeyError, TypeError, ValueError):
                info[field] = None
        info.update(
            {
                "execution_source_row_order": _optional_int(
                    payload.get("execution_source_row_order")
                ),
                "execution_snapshot_age_seconds": _optional_float(
                    payload.get("execution_snapshot_age_seconds")
                ),
                "execution_snapshot_advanced": _optional_bool(
                    payload.get("execution_snapshot_advanced")
                ),
                "execution_snapshot_same_as_decision": _optional_bool(
                    payload.get("execution_snapshot_same_as_decision")
                ),
                "latency_time_band": payload.get("latency_time_band"),
            }
        )
        if result.executed and not result.is_forced_exit:
            if result.execution_used_book_side == "SellTop5":
                prefix, side = "latency_buy", "Buy"
            elif result.execution_used_book_side == "BuyTop5":
                prefix, side = "latency_sell", "Sell"
            else:
                prefix, side = None, None
            if prefix is not None:
                delta = _optional_float(payload.get(f"{prefix}_price_delta_bps"))
                info["latency_policy_selected_action_side"] = side
                info["delayed_vs_decision_execution_price_delta_bps"] = delta
                info["delayed_vs_decision_execution_price_delta_sign"] = (
                    None
                    if delta is None
                    else "positive"
                    if delta > 0
                    else "negative"
                    if delta < 0
                    else "zero"
                )
    else:
        info.update(
            {
                "execution_target_timestamp": None,
                "execution_source_timestamp": None,
                "execution_source_row_order": None,
                "execution_snapshot_age_seconds": None,
                "execution_snapshot_advanced": None,
                "execution_snapshot_same_as_decision": None,
                "latency_time_band": None,
            }
        )
    info.update(dict(closing_diagnostics or {}))
    info.setdefault("forced_liquidation_started", False)
    info.setdefault("forced_liquidation_stage", None)
    info.setdefault("forced_liquidation_resolution_timestamp", None)
    info.setdefault("forced_liquidation_delay_seconds", None)
    info.setdefault("stage1_failure_reason", None)
    info.setdefault("stage2_attempted", False)
    info.setdefault("stage2_snapshot_count", 0)
    info.setdefault("closing_auction_used", False)
    info.setdefault("closing_auction_price", None)
    info.setdefault("closing_auction_volume_delta", None)
    info.setdefault("forced_liquidation_unresolved", False)
    info["forced_liquidation_unresolved"] = (
        info.get("forced_liquidation_stage") == "unresolved"
    )
    return info


def persist_execution_closingrisk_sidecar_v02(
    *,
    result: ExecutionClosingRiskBuildResultV02,
    directory: Path,
    raw_source_identity: Mapping[str, Any],
    decision_artifact_identity: Mapping[str, Any],
) -> dict[str, Any]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sidecar_path = directory / "sidecar.pkl"
    temporary = directory / ".sidecar.pkl.tmp"
    result.sidecar.rows.to_pickle(temporary)
    temporary.replace(sidecar_path)
    manifest = {
        "schema_id": SIDECAR_MANIFEST_SCHEMA_ID,
        "status": "PASS",
        "execution_contract_id": EXECUTION_CONTRACT_ID,
        "environment_contract_id": ENVIRONMENT_CONTRACT_ID,
        "reward_clock_contract_id": REWARD_CLOCK_CONTRACT_ID,
        "transition_contract_id": TRANSITION_CONTRACT_ID,
        "timestamp_preprocessing_id": TIMESTAMP_PREPROCESSING_ID,
        "entry_cutoff_gate_id": ENTRY_CUTOFF_GATE_ID,
        "agent_trading_end_jst": AGENT_TRADING_END_TIME.isoformat(),
        "stage1_contract_id": STAGE1_CONTRACT_ID,
        "stage2_contract_id": STAGE2_CONTRACT_ID,
        "stage3_contract_id": STAGE3_CONTRACT_ID,
        "closing_auction_eligibility_id": CLOSING_AUCTION_ELIGIBILITY_ID,
        "target_action_mask_source": TARGET_ACTION_MASK_SOURCE,
        "symbol": result.sidecar.symbol,
        "trading_date": result.sidecar.trading_date,
        "row_count": len(result.sidecar),
        "decision_identity_sha256": result.sidecar.decision_identity_sha256,
        "sidecar_artifact_identity_sha256": result.sidecar.artifact_identity_sha256,
        "sidecar_file_sha256": file_sha256_v01(sidecar_path),
        "raw_source_identity": dict(raw_source_identity),
        "decision_artifact_identity": dict(decision_artifact_identity),
        "qc": dict(result.qc),
    }
    _write_json(directory / "manifest.json", manifest)
    return manifest


def load_execution_closingrisk_sidecar_v02(
    *, directory: Path, decision_rows: pd.DataFrame
) -> tuple[ExecutionClosingRiskSidecarV02, dict[str, Any]]:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    sidecar_path = directory / "sidecar.pkl"
    if manifest.get("schema_id") != SIDECAR_MANIFEST_SCHEMA_ID:
        raise RuntimeError("closing-risk sidecar manifest schema mismatch")
    for field, expected in (
        ("execution_contract_id", EXECUTION_CONTRACT_ID),
        ("environment_contract_id", ENVIRONMENT_CONTRACT_ID),
        ("reward_clock_contract_id", REWARD_CLOCK_CONTRACT_ID),
        ("transition_contract_id", TRANSITION_CONTRACT_ID),
        ("timestamp_preprocessing_id", TIMESTAMP_PREPROCESSING_ID),
        ("entry_cutoff_gate_id", ENTRY_CUTOFF_GATE_ID),
        ("agent_trading_end_jst", AGENT_TRADING_END_TIME.isoformat()),
        ("stage1_contract_id", STAGE1_CONTRACT_ID),
        ("stage2_contract_id", STAGE2_CONTRACT_ID),
        ("stage3_contract_id", STAGE3_CONTRACT_ID),
        ("closing_auction_eligibility_id", CLOSING_AUCTION_ELIGIBILITY_ID),
        ("target_action_mask_source", TARGET_ACTION_MASK_SOURCE),
    ):
        if manifest.get(field) != expected:
            raise RuntimeError(f"closing-risk sidecar {field} mismatch")
    if file_sha256_v01(sidecar_path) != manifest.get("sidecar_file_sha256"):
        raise RuntimeError("closing-risk sidecar file hash mismatch")
    identity = decision_population_identity_v02(decision_rows)
    if identity != manifest.get("decision_identity_sha256"):
        raise RuntimeError("closing-risk decision identity mismatch")
    rows = pd.read_pickle(sidecar_path)
    if len(rows) != len(decision_rows) or len(rows) != int(manifest["row_count"]):
        raise RuntimeError("closing-risk sidecar row count mismatch")
    observed = execution_sidecar_artifact_identity_v02(
        rows=rows,
        symbol=str(manifest["symbol"]),
        trading_date=str(manifest["trading_date"]),
        decision_identity_sha256=identity,
    )
    if observed != manifest.get("sidecar_artifact_identity_sha256"):
        raise RuntimeError("closing-risk sidecar semantic identity mismatch")
    return (
        ExecutionClosingRiskSidecarV02(
            rows=rows,
            symbol=str(manifest["symbol"]),
            trading_date=str(manifest["trading_date"]),
            decision_identity_sha256=identity,
            artifact_identity_sha256=observed,
        ),
        manifest,
    )


def attach_execution_closingrisk_sidecar_v02(
    decision_rows: pd.DataFrame, sidecar: ExecutionClosingRiskSidecarV02
) -> pd.DataFrame:
    validate_execution_closingrisk_sidecar_v02(decision_rows, sidecar)
    result = decision_rows.copy(deep=False)
    result.attrs = dict(getattr(decision_rows, "attrs", {}))
    result.attrs[TRADING_ENV_KWARGS_ATTR] = {
        "execution_contract_id": EXECUTION_CONTRACT_ID,
        "execution_sidecar": sidecar,
    }
    return result


def execution_sidecar_artifact_identity_v02(
    *,
    rows: pd.DataFrame,
    symbol: str,
    trading_date: str,
    decision_identity_sha256: str,
) -> str:
    """Bind every current execution semantic to the sidecar row payload."""

    return canonical_sha256_v01(
        {
            "schema_id": SIDECAR_SCHEMA_ID,
            "execution_contract_id": EXECUTION_CONTRACT_ID,
            "environment_contract_id": ENVIRONMENT_CONTRACT_ID,
            "reward_clock_contract_id": REWARD_CLOCK_CONTRACT_ID,
            "transition_contract_id": TRANSITION_CONTRACT_ID,
            "timestamp_preprocessing_id": TIMESTAMP_PREPROCESSING_ID,
            "entry_cutoff_gate_id": ENTRY_CUTOFF_GATE_ID,
            "agent_trading_end_jst": AGENT_TRADING_END_TIME.isoformat(),
            "stage1_contract_id": STAGE1_CONTRACT_ID,
            "stage2_contract_id": STAGE2_CONTRACT_ID,
            "stage3_contract_id": STAGE3_CONTRACT_ID,
            "closing_auction_eligibility_id": CLOSING_AUCTION_ELIGIBILITY_ID,
            "target_action_mask_source": TARGET_ACTION_MASK_SOURCE,
            "symbol": str(symbol),
            "trading_date": str(trading_date),
            "decision_identity_sha256": str(decision_identity_sha256),
            "rows": _identity_rows(rows),
        }
    )


def validate_execution_closingrisk_sidecar_v02(
    decision_rows: pd.DataFrame,
    sidecar: ExecutionClosingRiskSidecarV02,
) -> None:
    """Fail closed on missing, stale, shape-only, or mutated sidecars."""

    if not isinstance(sidecar, ExecutionClosingRiskSidecarV02):
        raise ValueError("current execution sidecar type mismatch")
    expected_fields = {
        "schema_id": SIDECAR_SCHEMA_ID,
        "execution_contract_id": EXECUTION_CONTRACT_ID,
        "environment_contract_id": ENVIRONMENT_CONTRACT_ID,
        "reward_clock_contract_id": REWARD_CLOCK_CONTRACT_ID,
        "transition_contract_id": TRANSITION_CONTRACT_ID,
        "timestamp_preprocessing_id": TIMESTAMP_PREPROCESSING_ID,
        "entry_cutoff_gate_id": ENTRY_CUTOFF_GATE_ID,
        "agent_trading_end_jst": AGENT_TRADING_END_TIME.isoformat(),
        "stage1_contract_id": STAGE1_CONTRACT_ID,
        "stage2_contract_id": STAGE2_CONTRACT_ID,
        "stage3_contract_id": STAGE3_CONTRACT_ID,
        "closing_auction_eligibility_id": CLOSING_AUCTION_ELIGIBILITY_ID,
        "target_action_mask_source": TARGET_ACTION_MASK_SOURCE,
    }
    for field, expected in expected_fields.items():
        if getattr(sidecar, field, None) != expected:
            raise ValueError(f"current execution sidecar {field} mismatch")
    if len(decision_rows) != len(sidecar):
        raise ValueError("decision/closing-risk sidecar row count mismatch")
    decision_identity = decision_population_identity_v02(decision_rows)
    if decision_identity != sidecar.decision_identity_sha256:
        raise ValueError("decision/closing-risk sidecar identity mismatch")
    observed = execution_sidecar_artifact_identity_v02(
        rows=sidecar.rows,
        symbol=sidecar.symbol,
        trading_date=sidecar.trading_date,
        decision_identity_sha256=decision_identity,
    )
    if observed != sidecar.artifact_identity_sha256:
        raise ValueError("current execution sidecar artifact identity mismatch")


def summarize_execution_closingrisk_sidecar_v02(
    rows: pd.DataFrame,
) -> dict[str, Any]:
    ages = pd.to_numeric(rows["execution_snapshot_age_seconds"], errors="coerce")
    if ages.isna().any() or (ages < 0).any():
        raise ValueError("closing-risk snapshot ages are invalid")
    advanced = rows["execution_snapshot_advanced"].astype(bool)
    same = rows["execution_snapshot_same_as_decision"].astype(bool)
    if not (advanced ^ same).all():
        raise ValueError("advanced/same snapshot flags must be complementary")
    forced = rows.loc[rows["is_forced_exit"].astype(bool)]
    if len(forced) != 1:
        raise ValueError("sidecar requires exactly one forced-liquidation row")
    final = forced.iloc[0]
    result: dict[str, Any] = {
        "schema_id": "execution_closingrisk_episode_summary_v02",
        "execution_snapshot_count": len(rows),
        "execution_snapshot_advanced_count": int(advanced.sum()),
        "execution_snapshot_same_as_decision_count": int(same.sum()),
        "execution_snapshot_age_seconds": _distribution(ages.to_numpy()),
        "execution_snapshot_age_eq_0_count": int((ages == 0).sum()),
        "execution_snapshot_age_le_1s_count": int((ages <= 1).sum()),
        "execution_snapshot_age_gt_5s_count": int((ages > 5).sum()),
        "execution_snapshot_age_gt_10s_count": int((ages > 10).sum()),
        "stage1_long_executable": _top5_result(final, "Buy").executable,
        "stage1_short_executable": _top5_result(final, "Sell").executable,
        "stage2_long_available": bool(final["stage2_long_available"]),
        "stage2_short_available": bool(final["stage2_short_available"]),
        "closing_auction_eligible": bool(final["closing_auction_eligible"]),
        "top5": {},
    }
    for side in ("buy", "sell"):
        values = pd.to_numeric(
            rows[f"latency_{side}_price_delta_bps"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        result["top5"][side] = {
            "valid_count": int(finite.sum()),
            "failure_count": int((~finite).sum()),
            "price_delta_bps": _distribution(values[finite]),
        }
    return result


def _initialize_recovery_fields(row: dict[str, Any]) -> None:
    for side, book in (("long", "Buy"), ("short", "Sell")):
        row[f"stage2_{side}_available"] = False
        row[f"stage2_{side}_source_timestamp"] = pd.NaT
        row[f"stage2_{side}_source_row_order"] = np.nan
        row[f"stage2_{side}_snapshot_count"] = 0
        row[f"stage2_{side}_failure_reason"] = None
        for level in range(1, 6):
            for field in ("Price", "Qty"):
                row[f"stage2_{side}_{book}{level}_{field}"] = np.nan
    row.update(
        {
            "closing_auction_eligible": False,
            "closing_auction_failure_reason": None,
            "closing_auction_price": np.nan,
            "closing_auction_volume_delta": np.nan,
            "closing_auction_matching_row_count": 0,
            "closing_auction_eligible_row_count": 0,
            "closing_auction_unique_price_count": 0,
            "closing_auction_source_timestamp": pd.NaT,
            "closing_auction_source_row_order": np.nan,
            "preclosing_last_trading_volume": np.nan,
        }
    )


def _add_stage2_side(
    output: dict[str, Any], source: pd.DataFrame, side: str, book: str
) -> None:
    observed = 0
    last_reason: str | None = "no_recovery_snapshot"
    for _, row in source.iterrows():
        observed += 1
        execution = _top5_result(row, book)
        last_reason = execution.failure_reason
        if execution.executable:
            output[f"stage2_{side}_available"] = True
            output[f"stage2_{side}_source_timestamp"] = row["row_timestamp_raw"]
            output[f"stage2_{side}_source_row_order"] = int(row["row_order"])
            output[f"stage2_{side}_snapshot_count"] = observed
            output[f"stage2_{side}_failure_reason"] = None
            for level in range(1, 6):
                for field in ("Price", "Qty"):
                    output[f"stage2_{side}_{book}{level}_{field}"] = row[
                        f"{book}{level}_{field}"
                    ]
            return
    output[f"stage2_{side}_snapshot_count"] = observed
    output[f"stage2_{side}_failure_reason"] = last_reason


def _policy_independent_price_diagnostics(
    decision: Mapping[str, Any], execution: Mapping[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for side, book in (("buy", "Sell"), ("sell", "Buy")):
        at_decision = _top5_result(decision, book)
        delayed = _top5_result(execution, book)
        delta = np.nan
        if (
            at_decision.executable
            and delayed.executable
            and at_decision.price is not None
            and delayed.price is not None
        ):
            delta = (
                (float(delayed.price) - float(at_decision.price))
                / float(at_decision.price)
                * 10_000.0
            )
        output.update(
            {
                f"latency_{side}_decision_price": at_decision.price,
                f"latency_{side}_delayed_price": delayed.price,
                f"latency_{side}_price_delta_bps": delta,
                f"latency_{side}_top5_failure_reason": (
                    at_decision.failure_reason or delayed.failure_reason
                ),
            }
        )
    return output


def _top5_result(mapping: Mapping[str, Any], side: str):
    return compute_top5_weighted_price(
        [
            OrderBookLevel(
                mapping.get(f"{side}{level}_Price"),
                mapping.get(f"{side}{level}_Qty"),
            )
            for level in range(1, 6)
        ],
        qty=ORDER_QTY,
        used_book_side=f"{side}Top5",
    )


def _book_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {column: payload.get(column) for column in TOP5_COLUMNS}


def _stage2_book(payload: Mapping[str, Any], side: str) -> dict[str, Any]:
    required = "Buy" if side == "long" else "Sell"
    book: dict[str, Any] = {}
    for book_side in ("Buy", "Sell"):
        for level in range(1, 6):
            for field in ("Price", "Qty"):
                if book_side == required:
                    value = payload.get(
                        f"stage2_{side}_{required}{level}_{field}"
                    )
                else:
                    # The forced close ends Flat; the unused side is never valued.
                    value = 1.0 if field == "Price" else float(ORDER_QTY)
                book[f"{book_side}{level}_{field}"] = value
    return book


def _closing_auction_reward_result(
    *,
    position_before: str,
    cash_pnl_before: float,
    equity_before: float,
    previous_close: Any,
    price: Any,
) -> StepRewardResult:
    reference = validate_reference_notional(previous_close)
    execution_price = _finite_positive(price)
    if execution_price is None:
        return _unresolved_forced_liquidation_result(
            position_before=position_before,
            cash_pnl_before=cash_pnl_before,
            previous_close=previous_close,
            reason="closing_price_invalid",
        )
    cash_after = float(cash_pnl_before) + (
        ORDER_QTY * execution_price
        if position_before == "Long"
        else -ORDER_QTY * execution_price
    )
    if not reference.valid or reference.reference_notional is None:
        return StepRewardResult(
            position_after="Flat",
            cash_pnl_after=cash_after,
            equity_after=cash_after,
            reward_raw_yen=None,
            reward=NAN,
            reward_valid=False,
            training_valid=False,
            executed=True,
            execution_failed=False,
            execution_price=execution_price,
            execution_qty=ORDER_QTY,
            invalid_action=False,
            valuation_failed=False,
            is_forced_exit=True,
            forced_exit_failed=False,
            execution_used_book_side="none",
            valuation_used_book_side="none",
            failure_reason=reference.failure_reason,
            reference_notional=None,
            review_required=True,
        )
    reward = compute_reward(
        equity_before, cash_after, reference.reference_notional
    )
    return StepRewardResult(
        position_after="Flat",
        cash_pnl_after=cash_after,
        equity_after=cash_after,
        reward_raw_yen=reward.reward_raw_yen,
        reward=reward.reward,
        reward_valid=reward.reward_valid,
        training_valid=reward.reward_valid,
        executed=True,
        execution_failed=False,
        execution_price=execution_price,
        execution_qty=ORDER_QTY,
        invalid_action=False,
        valuation_failed=False,
        is_forced_exit=True,
        forced_exit_failed=False,
        execution_used_book_side="none",
        valuation_used_book_side="none",
        failure_reason=reward.failure_reason,
        reference_notional=reference.reference_notional,
        review_required=not reward.reward_valid,
    )


def _unresolved_forced_liquidation_result(
    *, position_before: str, cash_pnl_before: float, previous_close: Any, reason: str
) -> StepRewardResult:
    reference = validate_reference_notional(previous_close)
    return StepRewardResult(
        position_after=position_before,
        cash_pnl_after=float(cash_pnl_before),
        equity_after=None,
        reward_raw_yen=None,
        reward=NAN,
        reward_valid=False,
        training_valid=False,
        executed=False,
        execution_failed=True,
        execution_price=None,
        execution_qty=0,
        invalid_action=False,
        valuation_failed=False,
        is_forced_exit=True,
        forced_exit_failed=True,
        execution_used_book_side=(
            "BuyTop5" if position_before == "Long" else "SellTop5"
        ),
        valuation_used_book_side="none",
        failure_reason=f"forced_liquidation_unresolved:{reason}",
        reference_notional=reference.reference_notional,
        review_required=True,
    )


def _validate_decision_identity(
    rows: pd.DataFrame, symbol: str, trading_date: str
) -> None:
    symbols = {
        _canonical_symbol(value)
        for value in rows.get("Symbol", pd.Series([symbol])).dropna()
    }
    if symbols and symbols != {symbol}:
        raise ClosingRiskSidecarError(
            "execution_identity_mismatch", f"decision symbols={sorted(symbols)}"
        )
    dates = {
        normalize_timestamp_jst_v01(value).date().isoformat()
        for value in rows["grid_timestamp"]
    }
    if dates != {trading_date}:
        raise ClosingRiskSidecarError(
            "execution_date_mismatch", f"decision dates={sorted(dates)}"
        )


def _validate_raw_symbol_identity(rows: pd.DataFrame, symbol: str) -> None:
    symbols = {
        _canonical_symbol(value)
        for value in rows["Symbol"].dropna().unique().tolist()
    }
    if symbols != {symbol}:
        raise ClosingRiskSidecarError(
            "execution_identity_mismatch", f"raw symbols={sorted(symbols)}"
        )


def _identity_rows(rows: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {key: _canonical_value(value) for key, value in row.items()}
        for row in rows.to_dict("records")
    ]


def _canonical_value(value: Any) -> Any:
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return normalize_timestamp_jst_v01(value).isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else str(numeric)
    return str(value)


def _canonical_symbol(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _finite_positive(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) and numeric > 0 else None


def _finite_nonnegative(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) and numeric >= 0 else None


def _optional_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _optional_int(value: Any) -> int | None:
    numeric = _optional_float(value)
    return None if numeric is None else int(numeric)


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return None


def _distribution(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array):
        return {
            key: 0 if key == "count" else None
            for key in ("count", "mean", "p01", "p05", "p50", "p90", "p95", "p99", "max")
        }
    if not np.isfinite(array).all():
        raise ValueError("distribution input contains nonfinite values")
    return {
        "count": len(array),
        "mean": float(array.mean()),
        "p01": float(np.percentile(array, 1)),
        "p05": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def _write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_canonical_value,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "AGENT_TRADING_END_TIME",
    "CLOSING_AUCTION_TIME",
    "CLOSING_AUCTION_ELIGIBILITY_ID",
    "ENTRY_CUTOFF_GATE_ID",
    "ENTRY_CUTOFF_TIME",
    "ENVIRONMENT_CONTRACT_ID",
    "EXECUTION_CONTRACT_ID",
    "ExecutionClosingRiskBuildResultV02",
    "ExecutionClosingRiskSidecarV02",
    "LATENCY_SECONDS",
    "POPULATION_PREFLIGHT_SCHEMA_ID",
    "RAW_REQUIRED_COLUMNS",
    "REWARD_CLOCK_CONTRACT_ID",
    "SIDECAR_MANIFEST_SCHEMA_ID",
    "SIDECAR_SCHEMA_ID",
    "STAGE1_CONTRACT_ID",
    "STAGE2_CONTRACT_ID",
    "STAGE3_CONTRACT_ID",
    "TARGET_ACTION_MASK_SOURCE",
    "TIMESTAMP_PREPROCESSING_ID",
    "TRANSITION_CONTRACT_ID",
    "attach_execution_closingrisk_sidecar_v02",
    "build_execution_closingrisk_sidecar_v02",
    "decision_population_identity_v02",
    "execution_info_v02",
    "identify_strict_closing_auction_v02",
    "load_execution_closingrisk_sidecar_v02",
    "persist_execution_closingrisk_sidecar_v02",
    "policy_effective_action_mask_v02",
    "prepare_closingrisk_decision_rows_v02",
    "resolve_forced_liquidation_v02",
    "risk_management_action_v02",
    "summarize_execution_closingrisk_sidecar_v02",
    "validate_execution_payload_v02",
]
