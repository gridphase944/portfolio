"""Versioned DQN / Double-DQN one-step backup selection helpers.

The helper changes only next-action selection.  Both algorithms evaluate the
chosen action with the target network, use the same valid-action mask, and
detach every next-state value from the optimization graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


BACKUP_ALGORITHMS_V01 = ("dqn", "double_dqn")


def validate_backup_algorithm_v01(value: Any) -> str:
    if not isinstance(value, str) or value not in BACKUP_ALGORITHMS_V01:
        raise ValueError(
            f"backup_algorithm must be one of {list(BACKUP_ALGORITHMS_V01)}"
        )
    return value


@dataclass(frozen=True)
class NextBackupComparisonV01:
    dqn_action_ids: Any
    ddqn_action_ids: Any
    dqn_bootstrap: Any
    ddqn_bootstrap: Any
    selected_bootstrap: Any
    backup_gap: Any


def compute_next_backup_comparison_v01(
    *,
    online_next_q_values: Any,
    target_next_q_values: Any,
    dones: Any,
    next_action_masks: Any,
    backup_algorithm: str,
    nonnegative_tolerance: float = 1e-6,
) -> NextBackupComparisonV01:
    """Compute standard-DQN and DDQN bootstraps on one identical batch.

    Terminal rows may have an all-false mask.  Their action IDs are an
    implementation placeholder only; both returned bootstraps are exactly
    zero and therefore independent of that placeholder.
    """

    import tensorflow as tf

    algorithm = validate_backup_algorithm_v01(backup_algorithm)
    online = tf.stop_gradient(
        tf.convert_to_tensor(online_next_q_values, dtype=tf.float32)
    )
    target = tf.stop_gradient(
        tf.convert_to_tensor(target_next_q_values, dtype=tf.float32)
    )
    done_values = tf.convert_to_tensor(dones, dtype=tf.bool)
    masks = tf.convert_to_tensor(next_action_masks, dtype=tf.bool)

    if online.shape.rank != 2 or target.shape.rank != 2:
        raise ValueError("next Q values must be rank-2")
    if online.shape != target.shape:
        raise ValueError("online/target next Q shape mismatch")
    if masks.shape != online.shape:
        raise ValueError("next action mask shape mismatch")
    if done_values.shape.rank != 1 or done_values.shape[0] != online.shape[0]:
        raise ValueError("done shape mismatch")
    tf.debugging.assert_all_finite(online, "online next Q nonfinite")
    tf.debugging.assert_all_finite(target, "target next Q nonfinite")

    any_valid = tf.reduce_any(masks, axis=1)
    invalid_nonterminal = tf.logical_and(tf.logical_not(done_values), tf.logical_not(any_valid))
    tf.debugging.assert_equal(
        tf.reduce_any(invalid_nonterminal),
        False,
        message="nonterminal next action mask all false",
    )

    negative_inf = tf.constant(-np.inf, dtype=tf.float32)
    masked_target = tf.where(masks, target, negative_inf)
    masked_online = tf.where(masks, online, negative_inf)
    dqn_actions = tf.argmax(masked_target, axis=1, output_type=tf.int32)
    ddqn_actions = tf.argmax(masked_online, axis=1, output_type=tf.int32)
    dqn_values = tf.gather(target, dqn_actions, axis=1, batch_dims=1)
    ddqn_values = tf.gather(target, ddqn_actions, axis=1, batch_dims=1)
    zeros = tf.zeros_like(dqn_values)
    dqn_bootstrap = tf.where(done_values, zeros, dqn_values)
    ddqn_bootstrap = tf.where(done_values, zeros, ddqn_values)
    gap = dqn_bootstrap - ddqn_bootstrap
    nonterminal_gap = tf.boolean_mask(gap, tf.logical_not(done_values))
    if int(tf.size(nonterminal_gap).numpy()) > 0:
        tf.debugging.assert_greater_equal(
            tf.reduce_min(nonterminal_gap),
            tf.constant(-abs(float(nonnegative_tolerance)), tf.float32),
            message="DQN-DDQN backup gap is negative beyond tolerance",
        )
    selected = dqn_bootstrap if algorithm == "dqn" else ddqn_bootstrap
    return NextBackupComparisonV01(
        dqn_action_ids=tf.stop_gradient(dqn_actions),
        ddqn_action_ids=tf.stop_gradient(ddqn_actions),
        dqn_bootstrap=tf.stop_gradient(dqn_bootstrap),
        ddqn_bootstrap=tf.stop_gradient(ddqn_bootstrap),
        selected_bootstrap=tf.stop_gradient(selected),
        backup_gap=tf.stop_gradient(gap),
    )


def _distribution(prefix: str, values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array):
        return {
            f"{prefix}_count": 0,
            **{
                f"{prefix}_{name}": None
                for name in (
                    "mean", "median", "p90", "p95", "p99", "min", "max"
                )
            },
        }
    if not np.isfinite(array).all():
        raise ValueError(f"{prefix} distribution nonfinite")
    return {
        f"{prefix}_count": int(len(array)),
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_p90": float(np.percentile(array, 90)),
        f"{prefix}_p95": float(np.percentile(array, 95)),
        f"{prefix}_p99": float(np.percentile(array, 99)),
        f"{prefix}_min": float(array.min()),
        f"{prefix}_max": float(array.max()),
    }


def _valid_margin_rows(q_values: np.ndarray, masks: np.ndarray) -> np.ndarray:
    margins: list[float] = []
    for q_row, mask_row in zip(q_values, masks, strict=True):
        valid = np.asarray(q_row, dtype=np.float64)[np.asarray(mask_row, dtype=bool)]
        if len(valid) < 2:
            margins.append(0.0)
        else:
            ordered = np.sort(valid)
            margins.append(float(ordered[-1] - ordered[-2]))
    return np.asarray(margins, dtype=np.float64)


def mechanism_telemetry_row_v01(
    *,
    comparison: NextBackupComparisonV01,
    online_next_q_values: Any,
    target_next_q_values: Any,
    dones: Any,
    next_action_masks: Any,
    backup_algorithm: str,
    gamma: float,
    pre_update_index: int,
    last_target_sync_update: int,
    target_sync_interval: int,
    train_loss: float,
    abs_td_values: Any,
    gradient_global_norm_before_clip: float,
    gradient_global_norm_after_clip: float,
    learning_rate: float,
) -> Mapping[str, Any]:
    """Build one compact, pre-optimizer sampled-batch telemetry row."""

    algorithm = validate_backup_algorithm_v01(backup_algorithm)
    online = np.asarray(online_next_q_values, dtype=np.float64)
    target = np.asarray(target_next_q_values, dtype=np.float64)
    done_values = np.asarray(dones, dtype=bool).reshape(-1)
    masks = np.asarray(next_action_masks, dtype=bool)
    dqn_actions = np.asarray(comparison.dqn_action_ids.numpy(), dtype=np.int32)
    ddqn_actions = np.asarray(comparison.ddqn_action_ids.numpy(), dtype=np.int32)
    gap_all = np.asarray(comparison.backup_gap.numpy(), dtype=np.float64)
    nonterminal = ~done_values
    online_nt = online[nonterminal]
    target_nt = target[nonterminal]
    masks_nt = masks[nonterminal]
    gap = gap_all[nonterminal]
    denominator = int(nonterminal.sum())
    disagreements = dqn_actions[nonterminal] != ddqn_actions[nonterminal]
    if denominator and np.min(gap) < -1e-6:
        raise ValueError("DQN-DDQN backup gap QC failure")

    per_transition_q_difference = np.asarray(
        [
            float(np.mean(np.abs(o[m] - t[m])))
            for o, t, m in zip(online_nt, target_nt, masks_nt, strict=True)
        ],
        dtype=np.float64,
    )
    online_margins = _valid_margin_rows(online_nt, masks_nt)
    target_margins = _valid_margin_rows(target_nt, masks_nt)
    tie_count = 0
    for q_row, mask_row in zip(online_nt, masks_nt, strict=True):
        valid = q_row[mask_row]
        tie_count += int(np.count_nonzero(valid == np.max(valid)) > 1)

    confusion = np.zeros((3, 3), dtype=np.int64)
    for dqn_action, ddqn_action in zip(
        dqn_actions[nonterminal], ddqn_actions[nonterminal], strict=True
    ):
        confusion[int(dqn_action), int(ddqn_action)] += 1

    target_gap = float(gamma) * gap
    row: dict[str, Any] = {
        "backup_algorithm": algorithm,
        "pre_update_index": int(pre_update_index),
        "completed_update_index": int(pre_update_index) + 1,
        "last_target_sync_update": int(last_target_sync_update),
        "sync_age": int(pre_update_index) - int(last_target_sync_update),
        "target_sync_event_flag": bool(
            (int(pre_update_index) + 1) % int(target_sync_interval) == 0
        ),
        "batch_count": int(len(done_values)),
        "nonterminal_count": denominator,
        "terminal_count": int(done_values.sum()),
        "argmax_disagreement_count": int(disagreements.sum()),
        "argmax_disagreement_rate": (
            float(disagreements.mean()) if denominator else None
        ),
        "backup_gap_fraction_gt_zero": (
            float(np.mean(gap > 0.0)) if denominator else None
        ),
        "backup_gap_fraction_eq_zero": (
            float(np.mean(gap == 0.0)) if denominator else None
        ),
        "exact_online_tie_count": int(tie_count),
        "train_loss": float(train_loss),
        "abs_td_mean": float(np.mean(np.abs(np.asarray(abs_td_values, dtype=np.float64)))),
        "gradient_global_norm_before_clip": float(gradient_global_norm_before_clip),
        "gradient_global_norm_after_clip": float(gradient_global_norm_after_clip),
        "learning_rate": float(learning_rate),
    }
    row.update(_distribution("backup_gap", gap))
    row.update(_distribution("target_gap", target_gap))
    row.update(_distribution("online_q_margin", online_margins))
    row.update(_distribution("target_q_margin", target_margins))
    row.update(_distribution("online_target_abs_q_difference", per_transition_q_difference))
    for dqn_action in range(3):
        for ddqn_action in range(3):
            row[f"confusion_{dqn_action}_{ddqn_action}"] = int(
                confusion[dqn_action, ddqn_action]
            )
    return row
