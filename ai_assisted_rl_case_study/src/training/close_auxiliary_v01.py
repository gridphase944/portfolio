"""Close auxiliary head/loss helpers for 3-action DQN v0.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tensorflow as tf


CLOSE_AUX_LABEL_CONTINUE_V01 = 0.0
CLOSE_AUX_LABEL_CLOSE_V01 = 1.0
CLOSE_AUX_POSITION_LONG_ID_V01 = 1
CLOSE_AUX_POSITION_SHORT_ID_V01 = 2
CLOSE_AUX_CLASS_WEIGHT_MODES_V01 = {"none", "balanced"}


@dataclass(frozen=True)
class CloseAuxiliaryLossBatch:
    loss: tf.Tensor
    valid_mask: tf.Tensor
    logits: tf.Tensor | None
    probabilities: tf.Tensor | None
    qc: dict[str, Any]


def extract_q_values_from_model_output_v01(output: Any) -> Any:
    """Return the Q-value tensor from either single-output or aux-output models."""

    if isinstance(output, dict):
        if "q_values" in output:
            return output["q_values"]
        if "q_values_output" in output:
            return output["q_values_output"]
        raise ValueError("model output dict does not contain q_values")
    if isinstance(output, (list, tuple)):
        if not output:
            raise ValueError("model output sequence is empty")
        return output[0]
    return output


def extract_close_aux_logit_from_model_output_v01(output: Any) -> Any | None:
    """Return the close auxiliary logit tensor when present."""

    if isinstance(output, dict):
        return output.get("close_aux_logit")
    if isinstance(output, (list, tuple)):
        if len(output) < 2:
            return None
        return output[1]
    return None


def compute_close_auxiliary_loss_v01(
    *,
    model_output: Any,
    batch: Any,
    class_weight_mode: str = "none",
) -> CloseAuxiliaryLossBatch:
    """Compute masked BCE close/continue auxiliary loss for Long/Short rows."""

    mode = _validate_class_weight_mode(class_weight_mode)
    logits_raw = extract_close_aux_logit_from_model_output_v01(model_output)
    if logits_raw is None:
        raise ValueError("close auxiliary output is missing")
    logits = tf.reshape(tf.convert_to_tensor(logits_raw, dtype=tf.float32), [-1])

    labels_attr = getattr(batch, "close_aux_labels", None)
    valid_attr = getattr(batch, "close_aux_label_valid_masks", None)
    if labels_attr is None or valid_attr is None:
        labels = tf.zeros_like(logits, dtype=tf.float32)
        valid_mask = tf.zeros_like(logits, dtype=tf.bool)
    else:
        labels = tf.reshape(tf.convert_to_tensor(labels_attr, dtype=tf.float32), [-1])
        valid_mask = tf.reshape(tf.convert_to_tensor(valid_attr, dtype=tf.bool), [-1])

    _require_same_1d_shape(logits, labels, "close_aux_labels")
    _require_same_1d_shape(logits, valid_mask, "close_aux_label_valid_masks")

    position_ids = getattr(batch, "position_before_ids", None)
    if position_ids is None:
        position_mask = tf.ones_like(valid_mask, dtype=tf.bool)
    else:
        position_values = tf.reshape(
            tf.convert_to_tensor(position_ids, dtype=tf.int32),
            [-1],
        )
        _require_same_1d_shape(logits, position_values, "position_before_ids")
        position_mask = tf.logical_or(
            tf.equal(position_values, CLOSE_AUX_POSITION_LONG_ID_V01),
            tf.equal(position_values, CLOSE_AUX_POSITION_SHORT_ID_V01),
        )

    label_domain_mask = tf.logical_or(tf.equal(labels, 0.0), tf.equal(labels, 1.0))
    valid_mask = tf.logical_and(
        valid_mask,
        tf.logical_and(
            position_mask,
            tf.logical_and(
                tf.math.is_finite(labels),
                tf.logical_and(tf.math.is_finite(logits), label_domain_mask),
            ),
        ),
    )

    per_sample = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.where(valid_mask, labels, tf.zeros_like(labels)),
        logits=tf.where(valid_mask, logits, tf.zeros_like(logits)),
    )
    weights = _class_weights(labels=labels, valid_mask=valid_mask, mode=mode)
    safe_per_sample = tf.where(valid_mask, per_sample * weights, tf.zeros_like(per_sample))
    denominator = tf.reduce_sum(tf.where(valid_mask, weights, tf.zeros_like(weights)))
    loss = tf.math.divide_no_nan(tf.reduce_sum(safe_per_sample), denominator)
    probabilities = tf.sigmoid(logits)

    valid_count = _count_true(valid_mask)
    positive_mask = tf.logical_and(valid_mask, labels >= 0.5)
    negative_mask = tf.logical_and(valid_mask, labels < 0.5)
    predicted_positive = tf.logical_and(valid_mask, probabilities >= 0.5)
    true_positive = tf.logical_and(predicted_positive, labels >= 0.5)
    false_positive = tf.logical_and(predicted_positive, labels < 0.5)
    false_negative = tf.logical_and(
        tf.logical_and(valid_mask, probabilities < 0.5),
        labels >= 0.5,
    )
    precision = _divide_counts(_count_true(true_positive), _count_true(true_positive) + _count_true(false_positive))
    recall = _divide_counts(_count_true(true_positive), _count_true(true_positive) + _count_true(false_negative))

    qc = {
        "close_aux_enabled": True,
        "close_aux_class_weight_mode": mode,
        "close_aux_loss": _float_tensor(loss),
        "close_aux_valid_count": valid_count,
        "close_aux_positive_count": _count_true(positive_mask),
        "close_aux_negative_count": _count_true(negative_mask),
        "close_aux_masked_count": int(tf.size(valid_mask).numpy()) - valid_count,
        "close_aux_label_missing_count": int(tf.size(valid_mask).numpy()) - _count_true(tf.reshape(tf.convert_to_tensor(valid_attr, dtype=tf.bool), [-1])) if valid_attr is not None else int(tf.size(valid_mask).numpy()),
        "close_aux_precision": precision,
        "close_aux_recall": recall,
        "close_aux_auc": None,
        "close_aux_pred_mean": _finite_stat(probabilities, valid_mask, "mean"),
        "close_aux_pred_p50": _finite_percentile(probabilities, valid_mask, 50.0),
        "close_aux_pred_p90": _finite_percentile(probabilities, valid_mask, 90.0),
        "close_aux_loss_nan_count": _nan_count(loss),
        "close_aux_loss_inf_count": _inf_count(loss),
    }
    return CloseAuxiliaryLossBatch(
        loss=loss,
        valid_mask=valid_mask,
        logits=logits,
        probabilities=probabilities,
        qc=qc,
    )


def zero_close_auxiliary_loss_v01(batch: Any | None = None) -> CloseAuxiliaryLossBatch:
    """Return a zero aux-loss record for disabled or all-masked batches."""

    batch_size = 0
    if batch is not None and getattr(batch, "actions", None) is not None:
        batch_size = int(tf.size(tf.reshape(tf.convert_to_tensor(batch.actions), [-1])).numpy())
    loss = tf.constant(0.0, dtype=tf.float32)
    valid_mask = tf.zeros((batch_size,), dtype=tf.bool)
    qc = {
        "close_aux_enabled": False,
        "close_aux_loss": 0.0,
        "close_aux_valid_count": 0,
        "close_aux_positive_count": 0,
        "close_aux_negative_count": 0,
        "close_aux_masked_count": int(batch_size),
        "close_aux_label_missing_count": int(batch_size),
        "close_aux_precision": None,
        "close_aux_recall": None,
        "close_aux_auc": None,
        "close_aux_pred_mean": None,
        "close_aux_pred_p50": None,
        "close_aux_pred_p90": None,
        "close_aux_loss_nan_count": 0,
        "close_aux_loss_inf_count": 0,
    }
    return CloseAuxiliaryLossBatch(
        loss=loss,
        valid_mask=valid_mask,
        logits=None,
        probabilities=None,
        qc=qc,
    )


def _validate_class_weight_mode(value: str) -> str:
    mode = str(value)
    if mode not in CLOSE_AUX_CLASS_WEIGHT_MODES_V01:
        raise ValueError(
            "class_weight_mode must be one of "
            f"{sorted(CLOSE_AUX_CLASS_WEIGHT_MODES_V01)}, got {value!r}"
        )
    return mode


def _class_weights(*, labels: tf.Tensor, valid_mask: tf.Tensor, mode: str) -> tf.Tensor:
    if mode == "none":
        return tf.ones_like(labels, dtype=tf.float32)
    pos_count = tf.reduce_sum(tf.cast(tf.logical_and(valid_mask, labels >= 0.5), tf.float32))
    neg_count = tf.reduce_sum(tf.cast(tf.logical_and(valid_mask, labels < 0.5), tf.float32))
    pos_weight = tf.math.divide_no_nan(neg_count, pos_count)
    return tf.where(labels >= 0.5, tf.maximum(pos_weight, 1.0), tf.ones_like(labels))


def _require_same_1d_shape(reference: tf.Tensor, value: tf.Tensor, name: str) -> None:
    if reference.shape.rank != 1 or value.shape.rank != 1:
        raise ValueError(f"{name} must be rank 1 after flattening")
    reference_size = int(tf.size(reference).numpy())
    value_size = int(tf.size(value).numpy())
    if value_size != reference_size:
        raise ValueError(f"{name} size must be {reference_size}, got {value_size}")


def _count_true(mask: tf.Tensor) -> int:
    return int(tf.math.count_nonzero(tf.convert_to_tensor(mask, dtype=tf.bool)).numpy())


def _divide_counts(numerator: int, denominator: int) -> float | None:
    if int(denominator) <= 0:
        return None
    return float(numerator) / float(denominator)


def _float_tensor(value: tf.Tensor) -> float:
    return float(tf.convert_to_tensor(value, dtype=tf.float32).numpy())


def _nan_count(value: tf.Tensor) -> int:
    return int(tf.math.count_nonzero(tf.math.is_nan(tf.cast(value, tf.float32))).numpy())


def _inf_count(value: tf.Tensor) -> int:
    return int(tf.math.count_nonzero(tf.math.is_inf(tf.cast(value, tf.float32))).numpy())


def _finite_stat(values: tf.Tensor, valid_mask: tf.Tensor, stat: str) -> float | None:
    selected = tf.boolean_mask(tf.cast(values, tf.float32), valid_mask)
    finite = tf.boolean_mask(selected, tf.math.is_finite(selected))
    if int(tf.size(finite).numpy()) == 0:
        return None
    if stat == "mean":
        return float(tf.reduce_mean(finite).numpy())
    raise ValueError(f"unknown stat: {stat!r}")


def _finite_percentile(values: tf.Tensor, valid_mask: tf.Tensor, percentile: float) -> float | None:
    selected = tf.boolean_mask(tf.cast(values, tf.float32), valid_mask)
    finite = tf.boolean_mask(selected, tf.math.is_finite(selected))
    count = int(tf.size(finite).numpy())
    if count == 0:
        return None
    ordered = sorted(float(value) for value in finite.numpy().tolist())
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(percentile) / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
