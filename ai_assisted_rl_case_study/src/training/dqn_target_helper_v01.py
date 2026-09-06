"""DQN one-step target helpers v0.1.

This module computes standard DQN targets from already tensorized replay
batches. It does not implement loss calculation, gradient tape, optimizer
steps, Double DQN, or target-network updates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import tensorflow as tf

from training.close_auxiliary_v01 import extract_q_values_from_model_output_v01


NUM_ACTIONS_V01 = 3
POSITION_SIDE_FLAT_INDEX_V01 = 50
POSITION_SIDE_LONG_INDEX_V01 = 51
POSITION_SIDE_SHORT_INDEX_V01 = 52
TARGET_ACTION_MASK_SOURCES_V01 = {"env", "policy_effective"}


@dataclass
class DQNTargetBatch:
    target_q: tf.Tensor
    next_q_values: tf.Tensor
    max_next_q: tf.Tensor
    reward_component: tf.Tensor
    bootstrap_component: tf.Tensor
    qc: dict


def compute_max_next_q_v01(
    target_network,
    batch,
    target_action_mask_source: str = "policy_effective",
    num_actions: int = NUM_ACTIONS_V01,
) -> tf.Tensor:
    """Compute max_a Q_target(next_state, a) for each batch row."""

    next_states = tf.convert_to_tensor(batch.next_states, dtype=tf.float32)
    _require_rank(next_states, 2, "batch.next_states")
    batch_size = _dim0(next_states, "batch.next_states")

    q_values = extract_q_values_from_model_output_v01(
        target_network(next_states, training=False)
    )
    q_values = tf.convert_to_tensor(q_values, dtype=tf.float32)
    validate_q_network_output_v01(q_values, batch_size=batch_size, num_actions=num_actions)
    mask_source = _validate_target_action_mask_source(target_action_mask_source)
    next_action_masks = _resolve_next_action_masks(
        batch,
        next_states,
        target_action_mask_source=mask_source,
        num_actions=num_actions,
    )

    return masked_reduce_max_next_q_v01(q_values, next_action_masks, num_actions=num_actions)


def compute_dqn_targets_v01(
    target_network,
    batch,
    gamma: float = 0.99,
    target_action_mask_source: str = "policy_effective",
    num_actions: int = NUM_ACTIONS_V01,
) -> DQNTargetBatch:
    """Compute standard one-step DQN targets for a TensorReplayBatch."""

    gamma_value = _validate_gamma(gamma)
    mask_source = _validate_target_action_mask_source(target_action_mask_source)
    rewards = tf.convert_to_tensor(batch.rewards, dtype=tf.float32)
    dones = tf.convert_to_tensor(batch.dones, dtype=tf.bool)
    next_states = tf.convert_to_tensor(batch.next_states, dtype=tf.float32)

    _require_rank(next_states, 2, "batch.next_states")
    _require_rank(rewards, 1, "batch.rewards")
    _require_rank(dones, 1, "batch.dones")

    batch_size = _dim0(next_states, "batch.next_states")
    _require_batch_axis(rewards, batch_size, "batch.rewards")
    _require_batch_axis(dones, batch_size, "batch.dones")

    q_values = extract_q_values_from_model_output_v01(
        target_network(next_states, training=False)
    )
    q_values = tf.convert_to_tensor(q_values, dtype=tf.float32)
    validate_q_network_output_v01(q_values, batch_size=batch_size, num_actions=num_actions)
    next_action_masks = _resolve_next_action_masks(
        batch,
        next_states,
        target_action_mask_source=mask_source,
        num_actions=num_actions,
    )
    _require_batch_axis(next_action_masks, batch_size, "next_action_masks")
    _require_nonterminal_has_valid_next_action(
        next_action_masks,
        dones,
        name="next_action_masks",
    )

    max_next_q = masked_reduce_max_next_q_v01(
        q_values,
        next_action_masks,
        num_actions=num_actions,
    )
    reward_component = rewards
    raw_bootstrap = tf.cast(gamma_value, tf.float32) * max_next_q
    bootstrap_component = tf.where(
        dones,
        tf.zeros_like(raw_bootstrap),
        raw_bootstrap,
    )
    target_q = reward_component + bootstrap_component

    qc = {
        "batch_size": batch_size,
        "gamma": gamma_value,
        "target_action_mask_source": mask_source,
        "nan_count_rewards": _nan_count(rewards),
        "nan_count_next_states": _nan_count(next_states),
        "nan_count_target_q": _nan_count(target_q),
        "nan_count_network_output": _nan_count(q_values),
        "inf_count_network_output": _inf_count(q_values),
        "done_count": _true_count(dones),
        "next_action_mask_false_count": int(
            tf.math.count_nonzero(tf.logical_not(next_action_masks)).numpy()
        ),
        "next_action_mask_all_invalid_count": int(
            tf.math.count_nonzero(
                tf.logical_not(tf.reduce_any(next_action_masks, axis=1))
            ).numpy()
        ),
        "terminal_bootstrap_zero_count": _terminal_bootstrap_zero_count(
            dones,
            bootstrap_component,
        ),
        "max_target_q": _finite_stat(target_q, "max"),
        "min_target_q": _finite_stat(target_q, "min"),
        "mean_target_q": _finite_stat(target_q, "mean"),
        "max_max_next_q": _finite_stat(max_next_q, "max"),
        "min_max_next_q": _finite_stat(max_next_q, "min"),
    }

    return DQNTargetBatch(
        target_q=target_q,
        next_q_values=q_values,
        max_next_q=max_next_q,
        reward_component=reward_component,
        bootstrap_component=bootstrap_component,
        qc=qc,
    )


def masked_reduce_max_next_q_v01(
    q_values: tf.Tensor,
    next_action_masks: tf.Tensor,
    *,
    num_actions: int = NUM_ACTIONS_V01,
) -> tf.Tensor:
    """Reduce max over train/replay-order Q values using only valid next actions."""

    q_values = tf.convert_to_tensor(q_values, dtype=tf.float32)
    next_action_masks = tf.convert_to_tensor(next_action_masks, dtype=tf.bool)
    _require_rank(q_values, 2, "q_values")
    _require_rank(next_action_masks, 2, "next_action_masks")
    batch_size = _dim0(q_values, "q_values")
    _require_batch_axis(next_action_masks, batch_size, "next_action_masks")
    action_count = _validate_num_actions(num_actions)
    if _dim1(q_values, "q_values") != action_count:
        raise ValueError(
            f"q_values action dimension must be {action_count}, "
            f"got {_dim1(q_values, 'q_values')}"
        )
    if _dim1(next_action_masks, "next_action_masks") != action_count:
        raise ValueError(
            "next_action_masks action dimension must be "
            f"{action_count}, got {_dim1(next_action_masks, 'next_action_masks')}"
        )
    masked_q_values = tf.where(
        next_action_masks,
        q_values,
        tf.fill(tf.shape(q_values), tf.constant(float("-inf"), dtype=tf.float32)),
    )
    return tf.reduce_max(masked_q_values, axis=1)


def _resolve_next_action_masks(
    batch,
    next_states: tf.Tensor,
    *,
    target_action_mask_source: str = "policy_effective",
    num_actions: int = NUM_ACTIONS_V01,
) -> tf.Tensor:
    mask_source = _validate_target_action_mask_source(target_action_mask_source)
    if mask_source == "policy_effective":
        explicit_policy = getattr(batch, "next_policy_effective_action_masks", None)
        if explicit_policy is None:
            raise ValueError(
                "target_action_mask_source='policy_effective' requires "
                "batch.next_policy_effective_action_masks"
            )
        return tf.convert_to_tensor(explicit_policy, dtype=tf.bool)
    explicit = getattr(batch, "next_action_masks", None)
    if explicit is not None:
        return tf.convert_to_tensor(explicit, dtype=tf.bool)
    next_state_valid_masks = getattr(batch, "next_state_valid_masks", None)
    return infer_train_action_masks_from_states_v01(
        next_states,
        next_state_valid_masks,
        num_actions=num_actions,
    )


def _validate_target_action_mask_source(value: str) -> str:
    source = str(value)
    if source not in TARGET_ACTION_MASK_SOURCES_V01:
        raise ValueError(
            "target_action_mask_source must be one of "
            f"{sorted(TARGET_ACTION_MASK_SOURCES_V01)}, got {value!r}"
        )
    return source


def _require_nonterminal_has_valid_next_action(
    next_action_masks: tf.Tensor,
    dones: tf.Tensor,
    *,
    name: str,
) -> None:
    any_valid = tf.reduce_any(next_action_masks, axis=1)
    nonterminal_all_invalid = tf.logical_and(tf.logical_not(dones), tf.logical_not(any_valid))
    if bool(tf.reduce_any(nonterminal_all_invalid).numpy()):
        raise ValueError(f"{name} must contain at least one valid nonterminal action")


def infer_train_action_masks_from_states_v01(
    states: tf.Tensor,
    valid_state_masks: tf.Tensor | None = None,
    *,
    num_actions: int = NUM_ACTIONS_V01,
) -> tf.Tensor:
    """Infer train/replay-order [Hold, Buy, Sell] masks from position one-hot state."""

    states = tf.convert_to_tensor(states, dtype=tf.float32)
    action_count = _validate_num_actions(num_actions)
    _require_rank(states, 2, "states")
    batch_size = _dim0(states, "states")
    state_dim = _dim1(states, "states")
    if state_dim <= POSITION_SIDE_SHORT_INDEX_V01:
        return tf.ones((batch_size, action_count), dtype=tf.bool)

    if valid_state_masks is None:
        masks = tf.ones_like(states, dtype=tf.bool)
    else:
        masks = tf.convert_to_tensor(valid_state_masks, dtype=tf.bool)
        _require_rank(masks, 2, "valid_state_masks")
        _require_batch_axis(masks, batch_size, "valid_state_masks")
        if _dim1(masks, "valid_state_masks") != state_dim:
            raise ValueError(
                f"valid_state_masks state_dim must be {state_dim}, "
                f"got {_dim1(masks, 'valid_state_masks')}"
            )

    flat = states[:, POSITION_SIDE_FLAT_INDEX_V01]
    long = states[:, POSITION_SIDE_LONG_INDEX_V01]
    short = states[:, POSITION_SIDE_SHORT_INDEX_V01]
    flat_valid = masks[:, POSITION_SIDE_FLAT_INDEX_V01]
    long_valid = masks[:, POSITION_SIDE_LONG_INDEX_V01]
    short_valid = masks[:, POSITION_SIDE_SHORT_INDEX_V01]
    position_valid = tf.logical_and(flat_valid, tf.logical_and(long_valid, short_valid))
    finite = tf.logical_and(
        tf.math.is_finite(flat),
        tf.logical_and(tf.math.is_finite(long), tf.math.is_finite(short)),
    )
    recognizable = tf.logical_and(position_valid, finite)
    is_long = tf.logical_and(
        recognizable,
        tf.logical_and(long > 0.5, tf.logical_and(flat <= 0.5, short <= 0.5)),
    )
    is_short = tf.logical_and(
        recognizable,
        tf.logical_and(short > 0.5, tf.logical_and(flat <= 0.5, long <= 0.5)),
    )
    is_flat = tf.logical_and(
        recognizable,
        tf.logical_and(flat > 0.5, tf.logical_and(long <= 0.5, short <= 0.5)),
    )
    hold_valid = tf.ones((batch_size,), dtype=tf.bool)
    buy_valid = tf.logical_not(is_long)
    sell_valid = tf.logical_not(is_short)
    if action_count == 5:
        open_long_valid = is_flat
        close_long_valid = is_long
        open_short_valid = is_flat
        close_short_valid = is_short
        return tf.stack(
            [
                hold_valid,
                open_long_valid,
                close_long_valid,
                open_short_valid,
                close_short_valid,
            ],
            axis=1,
        )
    if action_count == 7:
        return tf.stack(
            [
                is_flat,
                is_flat,
                is_flat,
                is_long,
                is_long,
                is_short,
                is_short,
            ],
            axis=1,
        )
    return tf.stack([hold_valid, buy_valid, sell_valid], axis=1)


def validate_q_network_output_v01(
    q_values: tf.Tensor,
    batch_size: int,
    *,
    num_actions: int = NUM_ACTIONS_V01,
) -> None:
    """Validate target-network output shape [batch_size, num_actions]."""

    q_values = tf.convert_to_tensor(q_values)
    action_count = _validate_num_actions(num_actions)
    _require_rank(q_values, 2, "q_values")
    _require_batch_axis(q_values, batch_size, "q_values")
    action_dim = _dim1(q_values, "q_values")
    if action_dim != action_count:
        raise ValueError(
            f"q_values action dimension must be {action_count}, got {action_dim}"
        )


def _validate_num_actions(value: int) -> int:
    if isinstance(value, bool):
        raise ValueError("num_actions must be 3, 5, or 7")
    normalized = int(value)
    if normalized not in {3, 5, 7}:
        raise ValueError("num_actions must be 3, 5, or 7")
    return normalized


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


def _dim1(tensor: tf.Tensor, name: str) -> int:
    dim = tensor.shape[1]
    if dim is None:
        dim = int(tf.shape(tensor)[1].numpy())
    return int(dim)


def _validate_gamma(gamma: float) -> float:
    try:
        value = float(gamma)
    except (TypeError, ValueError) as exc:
        raise ValueError("gamma must be a finite scalar float") from exc
    if not math.isfinite(value):
        raise ValueError("gamma must be a finite scalar float")
    return value


def _nan_count(tensor: tf.Tensor) -> int:
    return int(tf.math.count_nonzero(tf.math.is_nan(tf.cast(tensor, tf.float32))).numpy())


def _inf_count(tensor: tf.Tensor) -> int:
    values = tf.cast(tensor, tf.float32)
    return int(tf.math.count_nonzero(tf.math.is_inf(values)).numpy())


def _true_count(tensor: tf.Tensor) -> int:
    return int(tf.math.count_nonzero(tensor).numpy())


def _terminal_bootstrap_zero_count(
    dones: tf.Tensor,
    bootstrap_component: tf.Tensor,
) -> int:
    terminal_bootstrap = tf.boolean_mask(bootstrap_component, dones)
    if int(tf.size(terminal_bootstrap).numpy()) == 0:
        return 0
    zero_mask = tf.equal(terminal_bootstrap, tf.zeros_like(terminal_bootstrap))
    return int(tf.math.count_nonzero(zero_mask).numpy())


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
