"""Minimal Keras Q-network v0.1."""

from __future__ import annotations

import tensorflow as tf


def build_q_network_v01(
    input_dim: int = 114,
    hidden_dim: int = 128,
    num_actions: int = 3,
    name: str = "q_network_v01",
):
    """Build a small MLP Q-network with linear action-value output."""

    input_dim = _validate_positive_int(input_dim, "input_dim")
    hidden_dim = _validate_positive_int(hidden_dim, "hidden_dim")
    num_actions = _validate_positive_int(num_actions, "num_actions")

    inputs = tf.keras.Input(shape=(input_dim,), dtype=tf.float32, name="q_input")
    x = tf.keras.layers.Dense(hidden_dim, activation="relu", name="dense_1")(inputs)
    x = tf.keras.layers.Dense(hidden_dim, activation="relu", name="dense_2")(x)
    outputs = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        dtype=tf.float32,
        name="q_values",
    )(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def build_q_network_with_close_aux_v01(
    input_dim: int = 114,
    hidden_dim: int = 128,
    num_actions: int = 3,
    name: str = "q_network_close_aux_v01",
):
    """Build a 3-action Q-network with a masked Close/Continue aux head."""

    input_dim = _validate_positive_int(input_dim, "input_dim")
    hidden_dim = _validate_positive_int(hidden_dim, "hidden_dim")
    num_actions = _validate_positive_int(num_actions, "num_actions")

    inputs = tf.keras.Input(shape=(input_dim,), dtype=tf.float32, name="q_input")
    x = tf.keras.layers.Dense(hidden_dim, activation="relu", name="dense_1")(inputs)
    x = tf.keras.layers.Dense(hidden_dim, activation="relu", name="dense_2")(x)
    q_values = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        dtype=tf.float32,
        name="q_values",
    )(x)
    close_aux_logit = tf.keras.layers.Dense(
        1,
        activation=None,
        dtype=tf.float32,
        name="close_aux_logit",
    )(x)
    return tf.keras.Model(
        inputs=inputs,
        outputs={"q_values": q_values, "close_aux_logit": close_aux_logit},
        name=name,
    )


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
