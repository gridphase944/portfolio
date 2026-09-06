"""DQN TD-loss helpers v0.1.

This module computes Q(s, a), one-step DQN targets, TD error, and a scalar
loss. It intentionally does not implement gradient tape, optimizers, target
network updates, Double DQN, or a train step.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import tensorflow as tf

from features.state_vector_v01 import STATE_DIM_V01
from training.close_auxiliary_v01 import extract_q_values_from_model_output_v01
from training.dqn_target_helper_v01 import (
    NUM_ACTIONS_V01,
    compute_dqn_targets_v01,
    validate_q_network_output_v01,
)
from training.q_input_adapter_v01 import build_q_network_input_v01


@dataclass
class DQNLossBatch:
    q_values: tf.Tensor
    chosen_q: tf.Tensor
    target_network_q_values: tf.Tensor
    target_q: tf.Tensor
    td_error: tf.Tensor
    per_sample_loss: tf.Tensor
    loss: tf.Tensor
    valid_loss_mask: tf.Tensor
    qc: dict


def gather_q_by_action_v01(
    q_values: tf.Tensor,
    actions: tf.Tensor,
    *,
    num_actions: int = NUM_ACTIONS_V01,
) -> tf.Tensor:
    """Gather Q-values for each row's selected discrete action."""

    q_values = tf.convert_to_tensor(q_values, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)

    _require_rank(q_values, 2, "q_values")
    _require_rank(actions, 1, "actions")
    batch_size = _dim0(q_values, "q_values")
    _require_batch_axis(actions, batch_size, "actions")

    action_dim = _dim1(q_values, "q_values")
    action_count = _validate_num_actions(num_actions)
    if action_dim != action_count:
        raise ValueError(
            f"q_values action dimension must be {action_count}, got {action_dim}"
        )

    gather_indices = tf.stack([tf.range(batch_size, dtype=tf.int32), actions], axis=1)
    return tf.gather_nd(q_values, gather_indices)


def compute_td_error_v01(
    chosen_q: tf.Tensor,
    target_q: tf.Tensor,
) -> tf.Tensor:
    """Compute per-row TD error as target_q - chosen_q."""

    chosen_q = tf.convert_to_tensor(chosen_q, dtype=tf.float32)
    target_q = tf.convert_to_tensor(target_q, dtype=tf.float32)
    _require_rank(chosen_q, 1, "chosen_q")
    _require_rank(target_q, 1, "target_q")
    _require_batch_axis(target_q, _dim0(chosen_q, "chosen_q"), "target_q")
    return target_q - chosen_q


def compute_per_sample_dqn_loss_v01(
    td_error: tf.Tensor,
    loss_type: str = "huber",
) -> tf.Tensor:
    """Compute unreduced per-sample DQN TD loss."""

    td_error = tf.convert_to_tensor(td_error, dtype=tf.float32)
    _require_rank(td_error, 1, "td_error")

    if loss_type == "huber":
        abs_error = tf.abs(td_error)
        quadratic = tf.minimum(abs_error, 1.0)
        linear = abs_error - quadratic
        return 0.5 * tf.square(quadratic) + linear
    if loss_type == "mse":
        return tf.square(td_error)
    raise ValueError(f"invalid loss_type: {loss_type!r}")


def compute_dqn_loss_v01(
    q_network,
    target_network,
    batch,
    gamma: float = 0.99,
    include_mask_features: bool = True,
    loss_type: str = "huber",
    expected_state_dim: int = STATE_DIM_V01,
    target_action_mask_source: str = "policy_effective",
    num_actions: int = NUM_ACTIONS_V01,
) -> DQNLossBatch:
    """Compute DQN one-step TD loss for a tensor replay batch."""

    current_input = build_q_network_input_v01(
        states=batch.states,
        valid_masks=batch.state_valid_masks,
        include_mask_features=include_mask_features,
        expected_state_dim=expected_state_dim,
    )
    q_values = extract_q_values_from_model_output_v01(
        q_network(current_input.network_input, training=True)
    )
    q_values = tf.convert_to_tensor(q_values, dtype=tf.float32)
    batch_size = _dim0(current_input.network_input, "current_input.network_input")
    validate_q_network_output_v01(
        q_values,
        batch_size=batch_size,
        num_actions=num_actions,
    )

    actions = tf.convert_to_tensor(batch.actions, dtype=tf.int32)
    chosen_q = gather_q_by_action_v01(q_values, actions, num_actions=num_actions)

    def adapted_target_network(next_states, training=False):
        next_input = build_q_network_input_v01(
            states=next_states,
            valid_masks=batch.next_state_valid_masks,
            include_mask_features=include_mask_features,
            expected_state_dim=expected_state_dim,
        )
        return target_network(next_input.network_input, training=training)

    target_batch = compute_dqn_targets_v01(
        target_network=adapted_target_network,
        batch=batch,
        gamma=gamma,
        target_action_mask_source=target_action_mask_source,
        num_actions=num_actions,
    )
    target_q = target_batch.target_q
    _require_rank(target_q, 1, "target_q")
    _require_batch_axis(target_q, batch_size, "target_q")

    valid_loss_mask = tf.logical_and(
        tf.math.is_finite(chosen_q),
        tf.logical_and(
            tf.math.is_finite(target_q),
            tf.ones_like(chosen_q, dtype=tf.bool),
        ),
    )
    td_error = compute_td_error_v01(chosen_q, target_q)
    safe_td_error = tf.where(valid_loss_mask, td_error, tf.zeros_like(td_error))
    per_sample_loss = compute_per_sample_dqn_loss_v01(
        safe_td_error,
        loss_type=loss_type,
    )
    valid_loss_mask = tf.logical_and(
        valid_loss_mask,
        tf.math.is_finite(per_sample_loss),
    )
    safe_per_sample_loss = tf.where(
        valid_loss_mask,
        per_sample_loss,
        tf.zeros_like(per_sample_loss),
    )
    sample_weights = getattr(batch, "is_weights", None)
    safe_sample_weights = None
    if sample_weights is not None:
        safe_sample_weights = tf.convert_to_tensor(sample_weights, dtype=tf.float32)
        _require_rank(safe_sample_weights, 1, "is_weights")
        _require_batch_axis(safe_sample_weights, batch_size, "is_weights")
        weight_valid_mask = tf.logical_and(
            tf.math.is_finite(safe_sample_weights),
            safe_sample_weights >= 0.0,
        )
        valid_loss_mask = tf.logical_and(valid_loss_mask, weight_valid_mask)
        safe_sample_weights = tf.where(
            valid_loss_mask,
            safe_sample_weights,
            tf.zeros_like(safe_sample_weights),
        )
        loss = tf.math.divide_no_nan(
            tf.reduce_sum(safe_per_sample_loss * safe_sample_weights),
            tf.reduce_sum(safe_sample_weights),
        )
        valid_loss_count = int(tf.math.count_nonzero(valid_loss_mask).numpy())
    else:
        valid_losses = tf.boolean_mask(safe_per_sample_loss, valid_loss_mask)
        valid_loss_count = _dim0(valid_losses, "valid_losses")
        if valid_loss_count > 0:
            loss = tf.reduce_mean(valid_losses)
        else:
            loss = tf.constant(0.0, dtype=tf.float32)

    qc = {
        "batch_size": batch_size,
        "gamma": float(gamma),
        "target_action_mask_source": str(target_action_mask_source),
        "loss_type": loss_type,
        "input_dim": current_input.qc["input_dim"],
        "valid_loss_count": valid_loss_count,
        "invalid_loss_count": batch_size - valid_loss_count,
        "nan_count_q_values": _nan_count(q_values),
        "inf_count_q_values": _inf_count(q_values),
        "nan_count_target_q": _nan_count(target_q),
        "inf_count_target_q": _inf_count(target_q),
        "nan_count_td_error": _nan_count(td_error),
        "nan_count_per_sample_loss": _nan_count(per_sample_loss),
        "done_count": target_batch.qc.get("done_count"),
        "mean_loss_on_valid_rows": float(loss.numpy()),
        "is_weighted_loss": bool(sample_weights is not None),
        "is_weight_nan_count": (
            0 if safe_sample_weights is None else _nan_count(safe_sample_weights)
        ),
        "is_weight_inf_count": (
            0 if safe_sample_weights is None else _inf_count(safe_sample_weights)
        ),
        "is_weight_mean": (
            None if safe_sample_weights is None else _finite_stat(safe_sample_weights, "mean")
        ),
        "is_weight_p90": (
            None if safe_sample_weights is None else _finite_percentile(safe_sample_weights, 90.0)
        ),
        "max_abs_td_error": _finite_stat(tf.abs(td_error), "max"),
        "mean_abs_td_error": _finite_stat(tf.abs(td_error), "mean"),
        "p95_abs_td_error": _finite_percentile(tf.abs(td_error), 95.0),
        "mean_td_error": _finite_stat(td_error, "mean"),
        "max_chosen_q": _finite_stat(chosen_q, "max"),
        "min_chosen_q": _finite_stat(chosen_q, "min"),
        "mean_chosen_q": _finite_stat(chosen_q, "mean"),
        "mean_abs_chosen_q": _finite_stat(tf.abs(chosen_q), "mean"),
        "p95_abs_chosen_q": _finite_percentile(tf.abs(chosen_q), 95.0),
        "max_target_q": _finite_stat(target_q, "max"),
        "min_target_q": _finite_stat(target_q, "min"),
        "mean_target_q": _finite_stat(target_q, "mean"),
        "mean_abs_target_q": _finite_stat(tf.abs(target_q), "mean"),
        "p95_abs_target_q": _finite_percentile(tf.abs(target_q), 95.0),
        "sample_count_by_action_id": _id_counts(actions),
        "td_error_abs_by_action_id": _group_abs_stats_by_id(
            tf.abs(td_error),
            actions,
        ),
    }
    for attr_name in (
        "position_before_ids",
        "position_after_ids",
        "selected_head_ids",
        "local_action_ids",
        "semantic_action_ids",
        "legacy_action_ids",
    ):
        ids = getattr(batch, attr_name, None)
        if ids is None:
            continue
        short_name = attr_name.removesuffix("s")
        qc[f"sample_count_by_{short_name}"] = _id_counts(ids)
        qc[f"td_error_abs_by_{short_name}"] = _group_abs_stats_by_id(
            tf.abs(td_error),
            ids,
        )

    return DQNLossBatch(
        q_values=q_values,
        chosen_q=chosen_q,
        target_network_q_values=target_batch.next_q_values,
        target_q=target_q,
        td_error=td_error,
        per_sample_loss=per_sample_loss,
        loss=loss,
        valid_loss_mask=valid_loss_mask,
        qc=qc,
    )


def _require_rank(tensor: tf.Tensor, rank: int, name: str) -> None:
    actual_rank = tensor.shape.rank
    if actual_rank is None:
        actual_rank = int(tf.rank(tensor).numpy())
    if actual_rank != rank:
        raise ValueError(f"{name} rank must be {rank}, got {actual_rank}")


def _require_batch_axis(tensor: tf.Tensor, batch_size: int, name: str) -> None:
    actual = _dim0(tensor, name)
    if actual != batch_size:
        raise ValueError(f"{name} batch axis must be {batch_size}, got {actual}")


def _dim0(tensor: tf.Tensor, name: str) -> int:
    dim = tensor.shape[0]
    if dim is None:
        dim = int(tf.shape(tensor)[0].numpy())
    return int(dim)


def _validate_num_actions(value: int) -> int:
    if isinstance(value, bool):
        raise ValueError("num_actions must be 3, 5, or 7")
    normalized = int(value)
    if normalized not in {3, 5, 7}:
        raise ValueError("num_actions must be 3, 5, or 7")
    return normalized


def _dim1(tensor: tf.Tensor, name: str) -> int:
    dim = tensor.shape[1]
    if dim is None:
        dim = int(tf.shape(tensor)[1].numpy())
    return int(dim)


def _nan_count(tensor: tf.Tensor) -> int:
    return int(tf.math.count_nonzero(tf.math.is_nan(tf.cast(tensor, tf.float32))).numpy())


def _inf_count(tensor: tf.Tensor) -> int:
    values = tf.cast(tensor, tf.float32)
    return int(tf.math.count_nonzero(tf.math.is_inf(values)).numpy())


def _finite_stat(tensor: tf.Tensor, stat: str) -> float | None:
    values = tf.cast(tensor, tf.float32)
    finite_values = tf.boolean_mask(values, tf.math.is_finite(values))
    if int(tf.size(finite_values).numpy()) == 0:
        return None
    if stat == "max":
        return float(tf.reduce_max(finite_values).numpy())
    if stat == "min":
        return float(tf.reduce_min(finite_values).numpy())
    if stat == "mean":
        return float(tf.reduce_mean(finite_values).numpy())
    raise ValueError(f"unknown stat: {stat!r}")


def _finite_percentile(tensor: tf.Tensor, percentile: float) -> float | None:
    values = tf.cast(tensor, tf.float32)
    finite_values = tf.boolean_mask(values, tf.math.is_finite(values))
    count = int(tf.size(finite_values).numpy())
    if count == 0:
        return None
    ordered = sorted(float(value) for value in finite_values.numpy().tolist())
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(percentile) / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _id_counts(ids: tf.Tensor) -> dict[str, int]:
    values = tf.convert_to_tensor(ids, dtype=tf.int32).numpy().tolist()
    counts: dict[str, int] = {}
    for value in values:
        normalized = int(value)
        if normalized < 0:
            continue
        key = str(normalized)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _group_abs_stats_by_id(values: tf.Tensor, ids: tf.Tensor) -> dict[str, dict[str, float]]:
    value_list = tf.cast(values, tf.float32).numpy().tolist()
    id_list = tf.convert_to_tensor(ids, dtype=tf.int32).numpy().tolist()
    groups: dict[str, dict[str, float]] = {}
    for value, id_value in zip(value_list, id_list):
        normalized = int(id_value)
        numeric = float(value)
        if normalized < 0 or not math.isfinite(numeric):
            continue
        key = str(normalized)
        slot = groups.setdefault(key, {"count": 0, "sum": 0.0, "mean": 0.0})
        slot["count"] = int(slot["count"]) + 1
        slot["sum"] = float(slot["sum"]) + abs(numeric)
    for slot in groups.values():
        count = int(slot["count"])
        slot["mean"] = (float(slot["sum"]) / count) if count > 0 else None
    return groups
