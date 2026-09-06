"""Bounded durable E0 health, learning, policy, and graph-source metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import resource
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from training.rl_representation_integration_v01 import OBSERVABILITY_SCHEMA_ID


NEAR_CONSTANT_STD_TOLERANCE = 1.0e-8
GRAPH_SOURCE_SCHEMA_ID = "rl_representation_e0_graph_sources_v01"


def metric_schema_manifest_v01() -> dict[str, Any]:
    """Return denominator/aggregation declarations for every persisted family."""

    fields = [
        _field("run_identity", "health", "identity", "run", None),
        _field("replay_staged_count", "health", "transition", "run", None),
        _field("replay_committed_count", "health", "transition", "run", None),
        _field("replay_skipped_count", "health", "transition", "run", None),
        _field("replay_discarded_count", "health", "transition", "run", None),
        _field("training_invalid_episode_count", "health", "episode", "run", None),
        _field("review_required_count", "health", "step", "run", None),
        _field("nonfinite_count", "health", "value", "finite_gate", "checked_value_count"),
        _field("fatal_finite_gate_count", "health", "event", "run", None),
        _field("encoder_finite_rate", "health", "ratio", "family_run", "encoder_output_value_count"),
        _field("gradient_global_norm_before_clip", "learning", "L2_norm", "optimizer_update", None),
        _field("gradient_global_norm_after_clip", "learning", "L2_norm", "optimizer_update", None),
        _field("learning_rate", "learning", "optimizer_rate", "optimizer_update", None),
        _field("replay_size", "health", "transition", "optimizer_update", None),
        _field("train_credit_added", "health", "transition_credit", "run", None),
        _field("train_credit_consumed", "health", "transition_credit", "run", None),
        _field("train_credit_pending", "health", "transition_credit", "run", None),
        _field("optimizer_update_count", "health", "update", "run", None),
        _field("target_network_update_count", "health", "update", "run", None),
        _field("q_by_action", "learning", "Q_value", "optimizer_update_action", "sample_count"),
        _field("chosen_q", "learning", "Q_value", "optimizer_update", "sample_count"),
        _field("valid_max_q", "learning", "Q_value", "optimizer_update", "valid_row_count"),
        _field("q_margin", "learning", "Q_value", "optimizer_update", "rows_with_two_valid_actions"),
        _field("td_error", "learning", "reward_unit", "optimizer_update", "sample_count"),
        _field("bellman_target", "learning", "reward_unit", "optimizer_update", "sample_count"),
        _field("loss", "learning", "loss", "optimizer_update", "loss_batch_denominator"),
        _field("epsilon", "learning", "probability", "optimizer_update", None),
        _field("greedy_decision_count", "learning", "decision", "optimizer_update_interval", "policy_decision_count"),
        _field("random_valid_decision_count", "learning", "decision", "optimizer_update_interval", "policy_decision_count"),
        _field("action_by_position_before_counts", "policy", "decision", "episode", "policy_eligible_decision_count"),
        _field("policy_action_rate", "policy", "ratio", "episode", "policy_eligible_decision_count"),
        _field("position_occupancy_step_counts", "policy", "step", "episode", "env_step_count"),
        _field("holding_time_seconds", "policy", "second", "completed_holding", "completed_holding_count"),
        _field("full_episode_normalized_reward", "economic", "normalized_reward", "episode", "env_step_count"),
        _field("full_episode_raw_yen_reward", "economic", "JPY", "episode", "env_step_count"),
        _field("warmup_raw_yen_reward", "warmup", "JPY", "episode_session", "warmup_step_count"),
        _field("policy_phase_raw_yen_reward", "economic", "JPY", "episode", "policy_eligible_decision_count"),
        _field("failure_by_root_reason", "policy", "event", "episode", "env_step_count"),
        _field("warmup_step_count", "warmup", "step", "episode_session", None),
        _field("warmup_replay_excluded_count", "warmup", "transition", "episode_session", "warmup_step_count"),
        _field("warmup_carried_position_steps", "warmup", "step", "episode_session_position", "warmup_step_count"),
        _field("first_common_policy_eligible_timestamp", "warmup", "timestamp", "episode_session", None),
        _field("no_ready_episode_count", "warmup", "episode", "run", None),
        _field("no_ready_session_count", "warmup", "session", "run", None),
        _field("throughput", "health", "event_per_second", "run", "elapsed_seconds"),
        _field("process_memory", "health", "byte", "run_snapshot", None),
        _field("input_wait_seconds", "health", "second", "run", "env_step_count"),
        _field("embedding_dimension_mean_std", "embedding", "encoder_output", "family_run", "finite_embedding_count"),
        _field("embedding_l2_norm", "embedding", "encoder_output_L2", "family_run", "embedding_count"),
        _field("concentration", "policy", "share", "grouping", "episode_count"),
    ]
    graphs = {
        "loss_td_error_vs_optimizer_update": ["learning_updates.optimizer_update_index", "learning_updates.loss", "learning_updates.td_error"],
        "action_wise_q_distributions": ["learning_updates.optimizer_update_index", "learning_updates.q_by_action"],
        "q_margin_distribution": ["learning_updates.optimizer_update_index", "learning_updates.q_margin"],
        "diagnostic_pnl_vs_checkpoint": ["episodes.checkpoint_id", "episodes.full_episode_raw_yen_reward"],
        "episode_pnl_distribution": ["episodes.full_episode_raw_yen_reward"],
        "action_by_position": ["episodes.action_by_position_before_counts", "episodes.policy_eligible_decision_count"],
        "holding_time_distribution": ["episodes.completed_holding_time_seconds"],
        "symbol_date_session_concentration": ["episodes.symbol", "episodes.trading_date", "episodes.session_aggregations", "concentration"],
        "encoder_norm_std": ["embedding_diagnostics.family", "embedding_diagnostics.dimension_std", "embedding_diagnostics.l2_norm"],
        "warmup_policy_economic_decomposition": ["episodes.warmup_raw_yen_reward", "episodes.policy_phase_raw_yen_reward", "episodes.full_episode_raw_yen_reward"],
        "throughput_optimizer_performance": ["runtime.elapsed_seconds", "runtime.env_steps_per_second", "runtime.optimizer_updates_per_second"],
    }
    return {
        "schema_id": OBSERVABILITY_SCHEMA_ID,
        "graph_source_schema_id": GRAPH_SOURCE_SCHEMA_ID,
        "near_constant_dimension_std_tolerance": NEAR_CONSTANT_STD_TOLERANCE,
        "near_constant_tolerance_affects_model_input": False,
        "fields": fields,
        "graphs": graphs,
        "unbounded_per_transition_or_full_q_dump": False,
    }


class E0ObservabilityRecorderV01:
    """One bounded row per optimizer update and one summary per episode."""

    def __init__(self, *, run_identity: Mapping[str, Any]):
        if not isinstance(run_identity, Mapping) or not run_identity.get("run_id"):
            raise ValueError("run_identity with run_id is required")
        self.run_identity = dict(run_identity)
        self.health: dict[str, Any] = {
            "replay_staged_count": 0,
            "replay_committed_count": 0,
            "replay_skipped_count": 0,
            "replay_discarded_count": 0,
            "training_invalid_episode_count": 0,
            "no_ready_episode_count": 0,
            "no_ready_session_count": 0,
            "review_required_count": 0,
            "nonfinite_count": 0,
            "fatal_finite_gate_count": 0,
            "optimizer_update_count": 0,
            "target_network_update_count": 0,
            "train_credit_added": 0,
            "train_credit_consumed": 0,
            "train_credit_pending": 0,
        }
        self.learning_updates: list[dict[str, Any]] = []
        self.episodes: list[dict[str, Any]] = []
        self.embedding_diagnostics: list[dict[str, Any]] = []
        self.runtime: dict[str, Any] = {}

    def update_health(self, **values: Any) -> None:
        unknown = set(values) - set(self.health)
        if unknown:
            raise ValueError(f"unknown E0 health fields: {sorted(unknown)}")
        self.health.update(values)

    def record_learning_update(
        self,
        *,
        optimizer_update_index: int,
        q_values: np.ndarray,
        chosen_action_ids: np.ndarray,
        valid_action_masks: np.ndarray,
        td_errors: np.ndarray,
        bellman_targets: np.ndarray,
        loss: float,
        epsilon: float,
        gradient_global_norm_before_clip: float,
        gradient_global_norm_after_clip: float,
        learning_rate: float,
        replay_size: int,
        greedy_decision_count: int,
        random_valid_decision_count: int,
        fallback_source_counts: Mapping[str, int],
        action_by_position_before_counts: Mapping[str, int],
    ) -> None:
        q = np.asarray(q_values, dtype=np.float64)
        actions = np.asarray(chosen_action_ids)
        masks = np.asarray(valid_action_masks)
        if q.ndim != 2 or q.shape[1] != 3 or actions.shape != (q.shape[0],):
            raise ValueError("Q/action batch shape mismatch")
        if masks.shape != q.shape or masks.dtype != np.bool_:
            raise ValueError("valid action mask shape/dtype mismatch")
        if np.any(actions < 0) or np.any(actions >= 3):
            raise ValueError("chosen action id invalid")
        chosen = q[np.arange(q.shape[0]), actions.astype(int)]
        valid_max: list[float] = []
        margins: list[float] = []
        for row, mask in zip(q, masks, strict=True):
            valid = row[mask]
            if valid.size:
                valid_max.append(float(np.max(valid)))
            if valid.size >= 2:
                ordered = np.sort(valid)
                margins.append(float(ordered[-1] - ordered[-2]))
        row = {
            "optimizer_update_index": int(optimizer_update_index),
            "q_by_action": {
                name: _distribution(q[:, index])
                for index, name in enumerate(("Hold", "Buy", "Sell"))
            },
            "chosen_q": _distribution(chosen),
            "valid_max_q": _distribution(valid_max),
            "q_margin": _distribution(margins),
            "td_error": _td_distribution(td_errors),
            "bellman_target": _distribution(bellman_targets),
            "loss": _finite(loss, "loss"),
            "epsilon": _finite(epsilon, "epsilon"),
            "gradient_global_norm_before_clip": _finite(
                gradient_global_norm_before_clip, "gradient_global_norm_before_clip"
            ),
            "gradient_global_norm_after_clip": _finite(
                gradient_global_norm_after_clip, "gradient_global_norm_after_clip"
            ),
            "learning_rate": _finite(learning_rate, "learning_rate"),
            "replay_size": int(replay_size),
            "greedy_decision_count": int(greedy_decision_count),
            "random_valid_decision_count": int(random_valid_decision_count),
            "policy_decision_count": int(greedy_decision_count) + int(random_valid_decision_count),
            "fallback_source_counts": _nonnegative_counts(fallback_source_counts),
            "action_by_position_before_counts": _nonnegative_counts(
                action_by_position_before_counts
            ),
        }
        self.learning_updates.append(row)
        self.health["optimizer_update_count"] = max(
            int(self.health["optimizer_update_count"]), int(optimizer_update_index)
        )

    def record_embedding_batch(self, *, family: str, embeddings: np.ndarray) -> None:
        if family not in {"Board", "Trade", "Market"}:
            raise ValueError("unknown embedding family")
        values = np.asarray(embeddings, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 32:
            raise ValueError("embedding batch must be [N,32]")
        finite = np.isfinite(values)
        finite_count = int(finite.sum())
        total = int(values.size)
        if not finite.all():
            self.health["nonfinite_count"] += total - finite_count
            self.health["fatal_finite_gate_count"] += 1
            raise RuntimeError(f"{family} encoder output finite gate failed")
        dimension_mean = values.mean(axis=0)
        dimension_std = values.std(axis=0)
        self.embedding_diagnostics.append(
            {
                "family": family,
                "embedding_count": int(values.shape[0]),
                "finite_count": finite_count,
                "finite_rate": {
                    "numerator": finite_count,
                    "denominator": total,
                    "aggregation_unit": "encoder_output_value",
                    "value": finite_count / total if total else None,
                },
                "dimension_mean": dimension_mean.tolist(),
                "dimension_std": dimension_std.tolist(),
                "l2_norm": _distribution(np.linalg.norm(values, axis=1)),
                "min": float(values.min()),
                "max": float(values.max()),
                "near_constant_dimension_count": int(
                    np.count_nonzero(dimension_std <= NEAR_CONSTANT_STD_TOLERANCE)
                ),
                "near_constant_std_tolerance": NEAR_CONSTANT_STD_TOLERANCE,
                "diagnostic_only_no_input_change": True,
            }
        )

    def record_episode(self, payload: Mapping[str, Any]) -> None:
        required = {
            "episode_id",
            "symbol",
            "trading_date",
            "policy_eligible_decision_count",
            "full_episode_raw_yen_reward",
            "warmup_raw_yen_reward",
            "policy_phase_raw_yen_reward",
        }
        if not required <= set(payload):
            raise ValueError(f"episode metrics missing: {sorted(required-set(payload))}")
        self.episodes.append(dict(payload))
        self.health["no_ready_episode_count"] += int(
            str(payload.get("policy_status", "")).endswith(
                "no_policy_eligible_decision_v01"
            )
        )
        self.health["no_ready_session_count"] += int(
            payload.get("no_ready_session_count", 0)
        )

    def record_runtime(
        self,
        *,
        elapsed_seconds: float,
        env_step_count: int,
        optimizer_update_count: int,
        actual_device: str,
        input_wait_seconds: float | None,
        input_wait_unavailable_reason: str | None = None,
    ) -> None:
        elapsed = _finite(elapsed_seconds, "elapsed_seconds")
        if elapsed <= 0:
            raise ValueError("elapsed_seconds must be positive")
        memory = process_memory_snapshot_v01()
        self.runtime = {
            "elapsed_seconds": elapsed,
            "env_step_count": int(env_step_count),
            "optimizer_update_count": int(optimizer_update_count),
            "env_steps_per_second": int(env_step_count) / elapsed,
            "optimizer_updates_per_second": int(optimizer_update_count) / elapsed,
            "actual_device": str(actual_device),
            "process_memory": memory,
            "input_wait_seconds": (
                _finite(input_wait_seconds, "input_wait_seconds")
                if input_wait_seconds is not None
                else None
            ),
            "input_wait_available": input_wait_seconds is not None,
            "input_wait_unavailable_reason": (
                None if input_wait_seconds is not None else str(input_wait_unavailable_reason or "not_measured")
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": OBSERVABILITY_SCHEMA_ID,
            "run_identity": dict(self.run_identity),
            "metric_schema": metric_schema_manifest_v01(),
            "health": dict(self.health),
            "learning_updates": list(self.learning_updates),
            "episodes": list(self.episodes),
            "embedding_diagnostics": list(self.embedding_diagnostics),
            "concentration": _concentration(self.episodes),
            "runtime": dict(self.runtime),
        }

    def write_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def process_memory_snapshot_v01() -> dict[str, Any]:
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (OSError, ValueError) as exc:
        return {"available": False, "reason": f"getrusage_failed:{type(exc).__name__}"}
    if sys.platform.startswith("linux"):
        return {"available": True, "metric": "process_max_rss", "bytes": value * 1024}
    if sys.platform == "darwin":
        return {"available": True, "metric": "process_max_rss", "bytes": value}
    return {"available": False, "reason": f"unsupported_ru_maxrss_unit:{sys.platform}"}


def _field(name: str, group: str, unit: str, aggregation_unit: str, denominator: str | None) -> dict[str, Any]:
    return {
        "name": name,
        "group": group,
        "unit": unit,
        "aggregation_unit": aggregation_unit,
        "denominator": denominator,
    }


def _distribution(values: Sequence[float] | np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size and not np.isfinite(array).all():
        raise RuntimeError("distribution contains nonfinite values")
    if not array.size:
        return {key: (0 if key == "count" else None) for key in ("count", "mean", "std", "min", "p50", "p95", "max")}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _td_distribution(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    result = _distribution(values)
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    result["abs_mean"] = float(np.abs(array).mean()) if array.size else None
    result["abs_max"] = float(np.abs(array).max()) if array.size else None
    return result


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_counts(values: Mapping[str, int]) -> dict[str, int]:
    result = {str(key): int(value) for key, value in values.items()}
    if any(value < 0 for value in result.values()):
        raise ValueError("counts must be nonnegative")
    return result


def _concentration(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in ("symbol", "trading_date"):
        sums: dict[str, float] = defaultdict(float)
        for episode in episodes:
            sums[str(episode.get(field, "unknown"))] += float(
                episode.get("full_episode_raw_yen_reward", 0.0)
            )
        absolute_total = sum(abs(value) for value in sums.values())
        ordered = sorted(sums.items(), key=lambda item: abs(item[1]), reverse=True)
        result[field] = {
            "group_count": len(ordered),
            "raw_yen_by_group": dict(ordered),
            "top_absolute_pnl_share": (
                abs(ordered[0][1]) / absolute_total if ordered and absolute_total else None
            ),
            "denominator_absolute_raw_yen": absolute_total,
            "aggregation_unit": f"episode_grouped_by_{field}",
        }
    session_sums: dict[str, float] = defaultdict(float)
    for episode in episodes:
        sessions = episode.get("session_aggregations", {})
        if isinstance(sessions, Mapping):
            for session, values in sessions.items():
                if isinstance(values, Mapping):
                    session_sums[str(session)] += float(
                        values.get("full_raw_yen_reward", 0.0)
                    )
    absolute_total = sum(abs(value) for value in session_sums.values())
    ordered = sorted(session_sums.items(), key=lambda item: abs(item[1]), reverse=True)
    result["session"] = {
        "group_count": len(ordered),
        "raw_yen_by_group": dict(ordered),
        "top_absolute_pnl_share": (
            abs(ordered[0][1]) / absolute_total if ordered and absolute_total else None
        ),
        "denominator_absolute_raw_yen": absolute_total,
        "aggregation_unit": "episode_session",
    }
    return result


__all__ = [
    "E0ObservabilityRecorderV01",
    "GRAPH_SOURCE_SCHEMA_ID",
    "NEAR_CONSTANT_STD_TOLERANCE",
    "metric_schema_manifest_v01",
    "process_memory_snapshot_v01",
]
