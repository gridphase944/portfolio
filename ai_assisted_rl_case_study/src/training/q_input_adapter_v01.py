"""Q-network input masking adapter v0.1.

This module builds finite TensorFlow inputs for Q-networks from tensor replay
batches. It does not define networks, fit normalizers, compute losses, or touch
the NumPy replay buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from features.state_vector_v01 import STATE_DIM_V01


@dataclass
class QInputBatch:
    masked_states: tf.Tensor
    effective_valid_masks: tf.Tensor
    network_input: tf.Tensor
    qc: dict


def build_q_network_input_v01(
    states: tf.Tensor,
    valid_masks: tf.Tensor,
    include_mask_features: bool = True,
    expected_state_dim: int = STATE_DIM_V01,
) -> QInputBatch:
    """Build finite Q-network inputs from state tensors and validity masks."""

    state_dim = _validate_positive_int(expected_state_dim, "expected_state_dim")
    states = tf.convert_to_tensor(states, dtype=tf.float32)
    valid_masks = tf.convert_to_tensor(valid_masks, dtype=tf.bool)

    _validate_state_and_mask_shapes(states, valid_masks, state_dim)

    finite_mask = tf.math.is_finite(states)
    effective_valid_masks = tf.logical_and(valid_masks, finite_mask)
    masked_states = tf.where(
        effective_valid_masks,
        states,
        tf.zeros_like(states),
    )

    if include_mask_features:
        network_input = tf.concat(
            [masked_states, tf.cast(effective_valid_masks, tf.float32)],
            axis=-1,
        )
    else:
        network_input = masked_states

    batch_size = _dim0(states, "states")
    input_dim = _dim1(network_input, "network_input")
    raw_finite = finite_mask
    raw_nan_mask = tf.math.is_nan(states)
    raw_nonfinite_mask = tf.logical_not(raw_finite)
    mask_false = tf.logical_not(valid_masks)
    effective_mask_false = tf.logical_not(effective_valid_masks)
    all_false_rows = tf.reduce_all(effective_mask_false, axis=1)

    qc = {
        "batch_size": batch_size,
        "state_dim": state_dim,
        "input_dim": input_dim,
        "input_nonfinite_count": _nonfinite_count(network_input),
        "raw_nonfinite_count": _true_count(raw_nonfinite_mask),
        "raw_nan_count": _true_count(raw_nan_mask),
        "mask_false_count": _true_count(mask_false),
        "effective_mask_false_count": _true_count(effective_mask_false),
        "zero_filled_count": _true_count(effective_mask_false),
        "include_mask_features": bool(include_mask_features),
        "all_false_mask_row_count": _true_count(all_false_rows),
        "terminal_like_next_state_row_count": _true_count(all_false_rows),
    }

    return QInputBatch(
        masked_states=masked_states,
        effective_valid_masks=effective_valid_masks,
        network_input=network_input,
        qc=qc,
    )


def build_q_input_from_batch_states_v01(
    batch,
    include_mask_features: bool = True,
    expected_state_dim: int = STATE_DIM_V01,
) -> QInputBatch:
    """Build Q-network input from TensorReplayBatch states."""

    return build_q_network_input_v01(
        states=batch.states,
        valid_masks=batch.state_valid_masks,
        include_mask_features=include_mask_features,
        expected_state_dim=expected_state_dim,
    )


def build_q_input_from_batch_next_states_v01(
    batch,
    include_mask_features: bool = True,
    expected_state_dim: int = STATE_DIM_V01,
) -> QInputBatch:
    """Build target-network input from TensorReplayBatch next_states."""

    return build_q_network_input_v01(
        states=batch.next_states,
        valid_masks=batch.next_state_valid_masks,
        include_mask_features=include_mask_features,
        expected_state_dim=expected_state_dim,
    )


def _validate_state_and_mask_shapes(
    states: tf.Tensor,
    valid_masks: tf.Tensor,
    expected_state_dim: int,
) -> None:
    _require_rank(states, 2, "states")
    _require_rank(valid_masks, 2, "valid_masks")

    batch_size = _dim0(states, "states")
    mask_batch_size = _dim0(valid_masks, "valid_masks")
    if mask_batch_size != batch_size:
        raise ValueError(
            f"valid_masks batch axis must be {batch_size}, got {mask_batch_size}"
        )

    state_dim = _dim1(states, "states")
    mask_state_dim = _dim1(valid_masks, "valid_masks")
    if state_dim != expected_state_dim:
        raise ValueError(
            f"states state_dim must be {expected_state_dim}, got {state_dim}"
        )
    if mask_state_dim != expected_state_dim:
        raise ValueError(
            f"valid_masks state_dim must be {expected_state_dim}, got {mask_state_dim}"
        )


def _require_rank(tensor: tf.Tensor, rank: int, name: str) -> None:
    actual_rank = tensor.shape.rank
    if actual_rank is None:
        actual_rank = int(tf.rank(tensor).numpy())
    if actual_rank != rank:
        raise ValueError(f"{name} rank must be {rank}, got {actual_rank}")


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


def _true_count(tensor: tf.Tensor) -> int:
    return int(tf.math.count_nonzero(tensor).numpy())


def _nonfinite_count(tensor: tf.Tensor) -> int:
    return _true_count(tf.logical_not(tf.math.is_finite(tensor)))


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        normalized_value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if normalized_value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return normalized_value
