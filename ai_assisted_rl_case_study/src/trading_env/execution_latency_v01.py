"""Experimental fixed one-second execution-latency sidecar contract.

The current TradingEnv default continues to execute and value on the decision
row.  This module builds a separate, explicitly opt-in execution-only payload
from raw PUSH snapshots.  It intentionally does not add delayed values to the
policy-visible episode dataframe.
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

from features.timestamp_preprocess_v01 import build_row_timestamp_raw
from trading_env.reward_v01 import OrderBookLevel, compute_top5_weighted_price


EXECUTION_CONTRACT_ID = "fixed_latency_latest_prior_1s_v01"
ENVIRONMENT_CONTRACT_ID = "trading_env_execution_latency1s_v01"
REWARD_CLOCK_CONTRACT_ID = "liquidation_equity_effective_clock_1s_v01"
TRANSITION_CONTRACT_ID = "rl_transition_execution_latency1s_v01"
SIDECAR_SCHEMA_ID = "execution_only_sidecar_fixed_latency1s_v01"
SIDECAR_MANIFEST_SCHEMA_ID = "execution_only_sidecar_manifest_latency1s_v01"
TRADING_ENV_KWARGS_ATTR = "_trading_env_explicit_kwargs_v01"
TOKYO_TZ = "Asia/Tokyo"
LATENCY_SECONDS = 1

TOP5_COLUMNS = tuple(
    f"{side}{level}_{field}"
    for side in ("Sell", "Buy")
    for level in range(1, 6)
    for field in ("Price", "Qty")
)
RAW_REQUIRED_COLUMNS = (
    "Symbol",
    "BidTime",
    "AskTime",
    "CurrentPriceTime",
    "TradingVolumeTime",
) + TOP5_COLUMNS

ROUTING_FAILURE_REASONS = (
    "execution_snapshot_missing",
    "execution_identity_mismatch",
    "execution_session_mismatch",
    "execution_date_mismatch",
    "execution_timestamp_after_target",
)


class ExecutionSidecarSelectionError(RuntimeError):
    """A structured fail-closed error raised while building a sidecar."""

    def __init__(self, reason: str, detail: str):
        if reason not in ROUTING_FAILURE_REASONS:
            raise ValueError(f"unsupported execution sidecar reason: {reason}")
        super().__init__(f"{reason}:{detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class ExecutionLatencySidecarV01:
    """Runtime execution-only rows aligned one-to-one with decision rows."""

    rows: pd.DataFrame
    symbol: str
    trading_date: str
    decision_identity_sha256: str
    artifact_identity_sha256: str
    execution_contract_id: str = EXECUTION_CONTRACT_ID
    environment_contract_id: str = ENVIRONMENT_CONTRACT_ID
    reward_clock_contract_id: str = REWARD_CLOCK_CONTRACT_ID
    transition_contract_id: str = TRANSITION_CONTRACT_ID
    schema_id: str = SIDECAR_SCHEMA_ID

    def __len__(self) -> int:
        return len(self.rows)

    def payload_at(self, index: int) -> dict[str, Any]:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("execution sidecar index must be an integer")
        position = int(index)
        if position < 0 or position >= len(self.rows):
            raise IndexError("execution sidecar index out of range")
        payload = self.rows.iloc[position].to_dict()
        payload.update(
            {
                "schema_id": self.schema_id,
                "execution_contract_id": self.execution_contract_id,
                "environment_contract_id": self.environment_contract_id,
                "reward_clock_contract_id": self.reward_clock_contract_id,
                "transition_contract_id": self.transition_contract_id,
                "sidecar_symbol": self.symbol,
                "sidecar_trading_date": self.trading_date,
                "sidecar_decision_identity_sha256": self.decision_identity_sha256,
                "sidecar_artifact_identity_sha256": self.artifact_identity_sha256,
            }
        )
        return payload


@dataclass(frozen=True)
class ExecutionSidecarBuildResultV01:
    sidecar: ExecutionLatencySidecarV01
    qc: Mapping[str, Any]


def normalize_timestamp_jst_v01(value: Any) -> pd.Timestamp:
    """Return a timestamp in Asia/Tokyo, treating naive input as JST."""

    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("timestamp is missing")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(TOKYO_TZ)
    return timestamp.tz_convert(TOKYO_TZ)


def session_for_timestamp_v01(value: Any) -> str | None:
    timestamp = normalize_timestamp_jst_v01(value)
    clock = timestamp.time()
    if time(9, 0) <= clock < time(11, 30):
        return "morning"
    if time(12, 30) <= clock < time(15, 25):
        return "afternoon"
    return None


def latency_time_band_v01(value: Any) -> str:
    timestamp = normalize_timestamp_jst_v01(value)
    clock = timestamp.time()
    if time(9, 0) <= clock < time(9, 30):
        return "morning_open_0900_0930"
    if time(9, 30) <= clock < time(11, 30):
        return "morning_core_0930_1130"
    if time(12, 30) <= clock < time(13, 0):
        return "afternoon_start_1230_1300"
    if time(13, 0) <= clock < time(15, 20):
        return "afternoon_core_1300_1520"
    if time(15, 20) <= clock < time(15, 25):
        return "forced_exit_region_1520_1525"
    return "outside_candidate_session"


def decision_population_identity_v01(decision_rows: pd.DataFrame) -> str:
    ordered_columns = (
        "grid_timestamp",
        "row_timestamp_raw",
        "source_row_order",
        "session",
        "is_forced_exit",
    )
    missing = set(ordered_columns) - set(decision_rows.columns)
    if missing:
        raise ValueError(f"decision identity columns missing: {sorted(missing)}")
    rows = []
    for row in decision_rows.loc[:, list(ordered_columns)].itertuples(index=False):
        values = row._asdict()
        rows.append(
            {
                "grid_timestamp": normalize_timestamp_jst_v01(
                    values["grid_timestamp"]
                ).isoformat(),
                "row_timestamp_raw": normalize_timestamp_jst_v01(
                    values["row_timestamp_raw"]
                ).isoformat(),
                "source_row_order": int(values["source_row_order"]),
                "session": str(values["session"]),
                "is_forced_exit": bool(values["is_forced_exit"]),
            }
        )
    return canonical_sha256_v01(rows)


def build_execution_latency_sidecar_v01(
    *,
    decision_rows: pd.DataFrame,
    raw_push_rows: pd.DataFrame,
    symbol: str,
    trading_date: str,
    latency_seconds: int = LATENCY_SECONDS,
) -> ExecutionSidecarBuildResultV01:
    """Select same-symbol/date/session latest-prior raw rows at ``t+1s``."""

    if latency_seconds != LATENCY_SECONDS:
        raise ValueError("this candidate supports exactly one second latency")
    if decision_rows.empty:
        raise ValueError("decision rows must not be empty")
    missing_raw = set(RAW_REQUIRED_COLUMNS) - set(raw_push_rows.columns)
    if missing_raw:
        raise ValueError(f"raw execution columns missing: {sorted(missing_raw)}")
    expected_symbol = _canonical_symbol(symbol)
    expected_date = str(trading_date)
    _validate_decision_identity_v01(decision_rows, expected_symbol, expected_date)
    _validate_raw_symbol_identity_v01(raw_push_rows, expected_symbol)

    prepared = build_row_timestamp_raw(raw_push_rows)
    valid_timestamp = prepared["row_timestamp_raw"].notna()
    prepared = prepared.loc[valid_timestamp].copy()
    if prepared.empty:
        raise ExecutionSidecarSelectionError(
            "execution_snapshot_missing", "raw source has no parseable row timestamp"
        )
    timestamp_series = prepared["row_timestamp_raw"]
    seconds = (
        timestamp_series.dt.hour * 3600
        + timestamp_series.dt.minute * 60
        + timestamp_series.dt.second
    )
    prepared["execution_session"] = np.select(
        [
            (seconds >= 9 * 3600) & (seconds < 11 * 3600 + 30 * 60),
            (seconds >= 12 * 3600 + 30 * 60) & (seconds < 15 * 3600 + 25 * 60),
        ],
        ["morning", "afternoon"],
        default=None,
    )
    raw_dates = set(timestamp_series.dt.date.astype(str).unique().tolist())
    if raw_dates != {expected_date}:
        raise ExecutionSidecarSelectionError(
            "execution_date_mismatch",
            f"expected={expected_date},observed={sorted(raw_dates)}",
        )

    coverage_target = normalize_timestamp_jst_v01(
        f"{expected_date} 15:24:56"
    )
    source_last_timestamp = max(
        normalize_timestamp_jst_v01(value)
        for value in prepared["row_timestamp_raw"]
    )
    if source_last_timestamp < coverage_target:
        raise ExecutionSidecarSelectionError(
            "execution_snapshot_missing",
            "forced-exit coverage requires source last timestamp >= 15:24:56",
        )

    session_sources: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    for session in ("morning", "afternoon"):
        source = prepared.loc[
            prepared["execution_session"] == session
        ].sort_values(["row_timestamp_raw", "row_order"], kind="mergesort")
        timestamps_ns = source["row_timestamp_raw"].astype("int64").to_numpy(
            dtype=np.int64, copy=False
        )
        session_sources[session] = (source.reset_index(drop=True), timestamps_ns)

    output_rows: list[dict[str, Any]] = []
    for decision_index, decision in decision_rows.reset_index(drop=True).iterrows():
        decision_timestamp = normalize_timestamp_jst_v01(
            decision["grid_timestamp"]
        )
        target_timestamp = decision_timestamp + pd.Timedelta(
            seconds=latency_seconds
        )
        decision_session = str(decision["session"])
        if session_for_timestamp_v01(decision_timestamp) != decision_session:
            raise ExecutionSidecarSelectionError(
                "execution_session_mismatch",
                f"decision index {decision_index} has invalid session identity",
            )
        if session_for_timestamp_v01(target_timestamp) != decision_session:
            raise ExecutionSidecarSelectionError(
                "execution_session_mismatch",
                f"decision index {decision_index} target crosses session",
            )
        if target_timestamp.date().isoformat() != expected_date:
            raise ExecutionSidecarSelectionError(
                "execution_date_mismatch",
                f"decision index {decision_index} target crosses date",
            )
        source_rows, source_timestamps_ns = session_sources[decision_session]
        selected_position = int(
            np.searchsorted(
                source_timestamps_ns,
                np.int64(target_timestamp.value),
                side="right",
            )
            - 1
        )
        if selected_position < 0:
            raise ExecutionSidecarSelectionError(
                "execution_snapshot_missing",
                f"decision index {decision_index} has no same-session latest-prior row",
            )
        selected = source_rows.iloc[selected_position]
        source_timestamp = normalize_timestamp_jst_v01(
            selected["row_timestamp_raw"]
        )
        if source_timestamp > target_timestamp:
            raise ExecutionSidecarSelectionError(
                "execution_timestamp_after_target",
                f"decision index {decision_index} selected a future row",
            )
        if session_for_timestamp_v01(source_timestamp) != decision_session:
            raise ExecutionSidecarSelectionError(
                "execution_session_mismatch",
                f"decision index {decision_index} source session mismatch",
            )
        if source_timestamp.date().isoformat() != expected_date:
            raise ExecutionSidecarSelectionError(
                "execution_date_mismatch",
                f"decision index {decision_index} source date mismatch",
            )

        decision_source_timestamp = normalize_timestamp_jst_v01(
            decision["row_timestamp_raw"]
        )
        decision_source_order = int(decision["source_row_order"])
        execution_source_order = int(selected["row_order"])
        same_snapshot = (
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
            "execution_snapshot_advanced": bool(not same_snapshot),
            "execution_snapshot_same_as_decision": bool(same_snapshot),
            "execution_symbol": expected_symbol,
            "execution_trading_date": expected_date,
            "execution_session": decision_session,
            "latency_time_band": latency_time_band_v01(decision_timestamp),
            "is_forced_exit": bool(decision["is_forced_exit"]),
        }
        for column in TOP5_COLUMNS:
            row[column] = selected[column]
        row.update(_policy_independent_price_diagnostics_v01(decision, selected))
        output_rows.append(row)

    sidecar_rows = pd.DataFrame(output_rows)
    decision_identity = decision_population_identity_v01(decision_rows)
    artifact_identity = canonical_sha256_v01(
        {
            "schema_id": SIDECAR_SCHEMA_ID,
            "execution_contract_id": EXECUTION_CONTRACT_ID,
            "environment_contract_id": ENVIRONMENT_CONTRACT_ID,
            "reward_clock_contract_id": REWARD_CLOCK_CONTRACT_ID,
            "transition_contract_id": TRANSITION_CONTRACT_ID,
            "symbol": expected_symbol,
            "trading_date": expected_date,
            "decision_identity_sha256": decision_identity,
            "rows": _sidecar_identity_rows_v01(sidecar_rows),
        }
    )
    sidecar = ExecutionLatencySidecarV01(
        rows=sidecar_rows,
        symbol=expected_symbol,
        trading_date=expected_date,
        decision_identity_sha256=decision_identity,
        artifact_identity_sha256=artifact_identity,
    )
    qc = summarize_execution_sidecar_v01(sidecar_rows)
    qc.update(
        {
            "source_last_timestamp": source_last_timestamp.isoformat(),
            "forced_exit_coverage_target": coverage_target.isoformat(),
            "forced_exit_coverage_pass": True,
            "decision_identity_sha256": decision_identity,
            "sidecar_artifact_identity_sha256": artifact_identity,
        }
    )
    return ExecutionSidecarBuildResultV01(sidecar=sidecar, qc=qc)


def validate_execution_payload_v01(
    *, payload: Mapping[str, Any] | None, decision_row: Mapping[str, Any]
) -> str | None:
    """Return a structured routing reason or ``None`` for a valid payload."""

    if not isinstance(payload, Mapping):
        return "execution_snapshot_missing"
    if (
        payload.get("schema_id") != SIDECAR_SCHEMA_ID
        or payload.get("execution_contract_id") != EXECUTION_CONTRACT_ID
        or payload.get("environment_contract_id") != ENVIRONMENT_CONTRACT_ID
        or payload.get("reward_clock_contract_id") != REWARD_CLOCK_CONTRACT_ID
        or payload.get("transition_contract_id") != TRANSITION_CONTRACT_ID
    ):
        return "execution_identity_mismatch"
    try:
        decision_timestamp = normalize_timestamp_jst_v01(
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
    if payload_decision != decision_timestamp:
        return "execution_identity_mismatch"
    expected_symbol = _canonical_symbol(
        decision_row.get("symbol", decision_row.get("Symbol"))
    )
    if (
        _canonical_symbol(payload.get("execution_symbol")) != expected_symbol
        or _canonical_symbol(payload.get("sidecar_symbol")) != expected_symbol
    ):
        return "execution_identity_mismatch"
    expected_date = decision_timestamp.date().isoformat()
    if (
        str(payload.get("execution_trading_date")) != expected_date
        or str(payload.get("sidecar_trading_date")) != expected_date
        or target.date().isoformat() != expected_date
        or source.date().isoformat() != expected_date
    ):
        return "execution_date_mismatch"
    decision_session = str(decision_row.get("session"))
    if (
        str(payload.get("execution_session")) != decision_session
        or session_for_timestamp_v01(decision_timestamp) != decision_session
        or session_for_timestamp_v01(target) != decision_session
        or session_for_timestamp_v01(source) != decision_session
    ):
        return "execution_session_mismatch"
    if target != decision_timestamp + pd.Timedelta(seconds=LATENCY_SECONDS):
        return "execution_identity_mismatch"
    if source > target:
        return "execution_timestamp_after_target"
    if any(column not in payload for column in TOP5_COLUMNS):
        return "execution_snapshot_missing"
    return None


def summarize_execution_sidecar_v01(rows: pd.DataFrame) -> dict[str, Any]:
    ages = pd.to_numeric(rows["execution_snapshot_age_seconds"], errors="coerce")
    if ages.isna().any() or not np.isfinite(ages.to_numpy(dtype=np.float64)).all():
        raise ValueError("execution snapshot age must be finite")
    if (ages < 0).any():
        raise ValueError("execution snapshot age must be nonnegative")
    advanced = rows["execution_snapshot_advanced"].astype(bool)
    same = rows["execution_snapshot_same_as_decision"].astype(bool)
    if not (advanced ^ same).all():
        raise ValueError("advanced/same snapshot flags must be complementary")
    count = len(rows)
    result: dict[str, Any] = {
        "schema_id": "execution_latency1s_episode_summary_v01",
        "execution_snapshot_count": count,
        "execution_snapshot_advanced_count": int(advanced.sum()),
        "execution_snapshot_advanced_rate": _ratio(int(advanced.sum()), count),
        "execution_snapshot_same_as_decision_count": int(same.sum()),
        "execution_snapshot_same_as_decision_rate": _ratio(int(same.sum()), count),
        "execution_snapshot_age_seconds": _distribution(ages.to_numpy()),
        "execution_snapshot_age_eq_0_count": int((ages == 0).sum()),
        "execution_snapshot_age_le_1s_count": int((ages <= 1).sum()),
        "execution_snapshot_age_gt_5s_count": int((ages > 5).sum()),
        "execution_snapshot_age_gt_10s_count": int((ages > 10).sum()),
        "forced_exit_snapshot_count": int(rows["is_forced_exit"].astype(bool).sum()),
        "session_counts": {
            str(key): int(value)
            for key, value in rows["execution_session"].value_counts().sort_index().items()
        },
        "time_band_counts": {
            str(key): int(value)
            for key, value in rows["latency_time_band"].value_counts().sort_index().items()
        },
        "latency_top5": {},
    }
    for side in ("buy", "sell"):
        delta = pd.to_numeric(rows[f"latency_{side}_price_delta_bps"], errors="coerce")
        valid = delta.notna() & np.isfinite(delta.fillna(0).to_numpy(dtype=np.float64))
        values = delta.loc[valid].to_numpy(dtype=np.float64)
        failure = rows[f"latency_{side}_top5_failure_reason"].notna()
        result["latency_top5"][side] = {
            "valid_count": int(valid.sum()),
            "failure_count": int(failure.sum()),
            "delta_bps": _distribution(values),
            "positive_count": int((values > 0).sum()),
            "negative_count": int((values < 0).sum()),
            "zero_count": int((values == 0).sum()),
        }
    return result


def persist_execution_sidecar_v01(
    *,
    result: ExecutionSidecarBuildResultV01,
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
    _write_json_atomic(directory / "manifest.json", manifest)
    return manifest


def load_execution_sidecar_v01(
    *, directory: Path, decision_rows: pd.DataFrame
) -> tuple[ExecutionLatencySidecarV01, dict[str, Any]]:
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    sidecar_path = directory / "sidecar.pkl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_id") != SIDECAR_MANIFEST_SCHEMA_ID:
        raise RuntimeError("execution sidecar manifest schema mismatch")
    if manifest.get("status") != "PASS":
        raise RuntimeError("execution sidecar manifest is not successful")
    for field, expected in (
        ("execution_contract_id", EXECUTION_CONTRACT_ID),
        ("environment_contract_id", ENVIRONMENT_CONTRACT_ID),
        ("reward_clock_contract_id", REWARD_CLOCK_CONTRACT_ID),
        ("transition_contract_id", TRANSITION_CONTRACT_ID),
    ):
        if manifest.get(field) != expected:
            raise RuntimeError(f"execution sidecar {field} mismatch")
    if file_sha256_v01(sidecar_path) != manifest.get("sidecar_file_sha256"):
        raise RuntimeError("execution sidecar file hash mismatch")
    decision_identity = decision_population_identity_v01(decision_rows)
    if decision_identity != manifest.get("decision_identity_sha256"):
        raise RuntimeError("execution sidecar decision identity mismatch")
    rows = pd.read_pickle(sidecar_path)
    if len(rows) != len(decision_rows) or len(rows) != int(manifest["row_count"]):
        raise RuntimeError("execution sidecar row count mismatch")
    observed_identity = canonical_sha256_v01(
        {
            "schema_id": SIDECAR_SCHEMA_ID,
            "execution_contract_id": EXECUTION_CONTRACT_ID,
            "environment_contract_id": ENVIRONMENT_CONTRACT_ID,
            "reward_clock_contract_id": REWARD_CLOCK_CONTRACT_ID,
            "transition_contract_id": TRANSITION_CONTRACT_ID,
            "symbol": str(manifest["symbol"]),
            "trading_date": str(manifest["trading_date"]),
            "decision_identity_sha256": decision_identity,
            "rows": _sidecar_identity_rows_v01(rows),
        }
    )
    if observed_identity != manifest.get("sidecar_artifact_identity_sha256"):
        raise RuntimeError("execution sidecar semantic identity mismatch")
    sidecar = ExecutionLatencySidecarV01(
        rows=rows,
        symbol=str(manifest["symbol"]),
        trading_date=str(manifest["trading_date"]),
        decision_identity_sha256=decision_identity,
        artifact_identity_sha256=observed_identity,
    )
    return sidecar, manifest


def attach_execution_sidecar_v01(
    decision_rows: pd.DataFrame, sidecar: ExecutionLatencySidecarV01
) -> pd.DataFrame:
    """Attach explicit Env kwargs via DataFrame attrs, never future columns."""

    if len(decision_rows) != len(sidecar):
        raise ValueError("decision/sidecar row count mismatch")
    if decision_population_identity_v01(decision_rows) != sidecar.decision_identity_sha256:
        raise ValueError("decision/sidecar identity mismatch")
    result = decision_rows.copy(deep=False)
    result.attrs = dict(getattr(decision_rows, "attrs", {}))
    result.attrs[TRADING_ENV_KWARGS_ATTR] = {
        "execution_contract_id": EXECUTION_CONTRACT_ID,
        "execution_sidecar": sidecar,
    }
    return result


def sidecar_env_kwargs_v01(decision_rows: pd.DataFrame) -> dict[str, Any]:
    value = getattr(decision_rows, "attrs", {}).get(TRADING_ENV_KWARGS_ATTR)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("TradingEnv explicit kwargs attr must be a mapping")
    if set(value) != {"execution_contract_id", "execution_sidecar"}:
        raise ValueError("TradingEnv explicit kwargs attr has unexpected fields")
    return dict(value)


def _policy_independent_price_diagnostics_v01(
    decision: Mapping[str, Any], execution: Mapping[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for side, book_side in (("buy", "Sell"), ("sell", "Buy")):
        decision_result = _top5_result_from_mapping(decision, book_side)
        delayed_result = _top5_result_from_mapping(execution, book_side)
        failure = decision_result.failure_reason or delayed_result.failure_reason
        if (
            decision_result.executable
            and delayed_result.executable
            and decision_result.price is not None
            and delayed_result.price is not None
        ):
            delta_bps = (
                (float(delayed_result.price) - float(decision_result.price))
                / float(decision_result.price)
                * 10_000.0
            )
        else:
            delta_bps = np.nan
        output.update(
            {
                f"latency_{side}_decision_price": decision_result.price,
                f"latency_{side}_delayed_price": delayed_result.price,
                f"latency_{side}_price_delta_bps": delta_bps,
                f"latency_{side}_top5_failure_reason": failure,
            }
        )
    return output


def _top5_result_from_mapping(mapping: Mapping[str, Any], side: str):
    return compute_top5_weighted_price(
        [
            OrderBookLevel(
                mapping.get(f"{side}{level}_Price"),
                mapping.get(f"{side}{level}_Qty"),
            )
            for level in range(1, 6)
        ],
        used_book_side=f"{side}Top5",
    )


def _validate_decision_identity_v01(
    decision_rows: pd.DataFrame, symbol: str, trading_date: str
) -> None:
    required = {
        "grid_timestamp",
        "row_timestamp_raw",
        "source_row_order",
        "session",
        "is_forced_exit",
    } | set(TOP5_COLUMNS)
    missing = required - set(decision_rows.columns)
    if missing:
        raise ValueError(f"decision execution columns missing: {sorted(missing)}")
    observed_symbols = {
        _canonical_symbol(value)
        for value in decision_rows.get(
            "Symbol", pd.Series([symbol], dtype=object)
        ).dropna()
    }
    if observed_symbols and observed_symbols != {symbol}:
        raise ExecutionSidecarSelectionError(
            "execution_identity_mismatch",
            f"decision symbols={sorted(observed_symbols)},expected={symbol}",
        )
    observed_dates = {
        normalize_timestamp_jst_v01(value).date().isoformat()
        for value in decision_rows["grid_timestamp"]
    }
    if observed_dates != {trading_date}:
        raise ExecutionSidecarSelectionError(
            "execution_date_mismatch",
            f"decision dates={sorted(observed_dates)},expected={trading_date}",
        )


def _validate_raw_symbol_identity_v01(
    raw_push_rows: pd.DataFrame, expected_symbol: str
) -> None:
    observed = {
        _canonical_symbol(value)
        for value in raw_push_rows["Symbol"].dropna().unique().tolist()
    }
    if observed != {expected_symbol}:
        raise ExecutionSidecarSelectionError(
            "execution_identity_mismatch",
            f"raw symbols={sorted(observed)},expected={expected_symbol}",
        )


def _canonical_symbol(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _sidecar_identity_rows_v01(rows: pd.DataFrame) -> list[dict[str, Any]]:
    identity_rows: list[dict[str, Any]] = []
    for row in rows.to_dict("records"):
        identity = {
            "decision_index": int(row["decision_index"]),
            "decision_timestamp": normalize_timestamp_jst_v01(
                row["decision_timestamp"]
            ).isoformat(),
            "execution_target_timestamp": normalize_timestamp_jst_v01(
                row["execution_target_timestamp"]
            ).isoformat(),
            "execution_source_timestamp": normalize_timestamp_jst_v01(
                row["execution_source_timestamp"]
            ).isoformat(),
            "execution_source_row_order": int(row["execution_source_row_order"]),
            "execution_snapshot_advanced": bool(row["execution_snapshot_advanced"]),
            "execution_snapshot_same_as_decision": bool(
                row["execution_snapshot_same_as_decision"]
            ),
        }
        for column in TOP5_COLUMNS:
            identity[column] = _canonical_scalar(row[column])
        identity_rows.append(identity)
    return identity_rows


def _canonical_scalar(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else str(numeric)
    return str(value)


def canonical_sha256_v01(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256_v01(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array):
        return {
            key: (0 if key == "count" else None)
            for key in ("count", "mean", "std", "min", "p01", "p05", "p50", "p90", "p95", "p99", "max")
        }
    if not np.isfinite(array).all():
        raise ValueError("distribution values must be finite")
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p01": float(np.percentile(array, 1)),
        "p05": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def _ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "aggregation_unit": "decision_step",
        "value": numerator / denominator if denominator else None,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "ENVIRONMENT_CONTRACT_ID",
    "EXECUTION_CONTRACT_ID",
    "ExecutionLatencySidecarV01",
    "ExecutionSidecarBuildResultV01",
    "ExecutionSidecarSelectionError",
    "LATENCY_SECONDS",
    "RAW_REQUIRED_COLUMNS",
    "REWARD_CLOCK_CONTRACT_ID",
    "ROUTING_FAILURE_REASONS",
    "SIDECAR_MANIFEST_SCHEMA_ID",
    "SIDECAR_SCHEMA_ID",
    "TOP5_COLUMNS",
    "TRADING_ENV_KWARGS_ATTR",
    "TRANSITION_CONTRACT_ID",
    "attach_execution_sidecar_v01",
    "build_execution_latency_sidecar_v01",
    "decision_population_identity_v01",
    "file_sha256_v01",
    "latency_time_band_v01",
    "load_execution_sidecar_v01",
    "normalize_timestamp_jst_v01",
    "persist_execution_sidecar_v01",
    "session_for_timestamp_v01",
    "sidecar_env_kwargs_v01",
    "summarize_execution_sidecar_v01",
    "validate_execution_payload_v01",
]
