"""Pre-optimizer finite validation for formal DQN training updates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import tensorflow as tf


FINITE_GATE_REASON_CODES_V01 = {
    "ok": 0,
    "nonfinite_online_q": 1,
    "nonfinite_target_q": 2,
    "nonfinite_bellman_target": 3,
    "nonfinite_td_error": 4,
    "nonfinite_loss": 5,
    "empty_gradient_set": 6,
    "missing_gradient": 7,
    "nonfinite_gradient": 8,
}
FINITE_GATE_REASON_BY_CODE_V01 = {
    code: reason for reason, code in FINITE_GATE_REASON_CODES_V01.items()
}


@dataclass(frozen=True)
class PreOptimizerFiniteGateResultV01:
    passed: bool
    reason: str
    online_q_nonfinite_count: int
    selected_q_nonfinite_count: int
    target_q_nonfinite_count: int
    bellman_target_nonfinite_count: int
    td_error_nonfinite_count: int
    loss_nonfinite_count: int
    missing_gradient_count: int
    nonfinite_gradient_count: int
    gradient_count: int
    trainable_variable_count: int
    gradient_norm: float | None
    loss: float | None
    affected_variable_names: tuple[str, ...]

    def diagnostics(self) -> dict[str, Any]:
        empty_gradient_set_count = int(
            self.reason == "empty_gradient_set"
        )
        return {
            "finite_gate_passed": bool(self.passed),
            "fatal_reason": None if self.passed else self.reason,
            "online_q_nonfinite_count": int(self.online_q_nonfinite_count),
            "selected_q_nonfinite_count": int(self.selected_q_nonfinite_count),
            "target_q_nonfinite_count": int(self.target_q_nonfinite_count),
            "bellman_target_nonfinite_count": int(
                self.bellman_target_nonfinite_count
            ),
            "td_error_nonfinite_count": int(self.td_error_nonfinite_count),
            "loss_nonfinite_count": int(self.loss_nonfinite_count),
            "missing_gradient_count": int(self.missing_gradient_count),
            "nonfinite_gradient_count": int(self.nonfinite_gradient_count),
            "empty_gradient_set_count": empty_gradient_set_count,
            "gradient_count": int(self.gradient_count),
            "trainable_variable_count": int(self.trainable_variable_count),
            "gradient_norm": self.gradient_norm,
            "loss": self.loss,
            "affected_variable_names": list(self.affected_variable_names),
            "nonfinite_tensor_count": int(
                self.online_q_nonfinite_count
                + self.selected_q_nonfinite_count
                + self.target_q_nonfinite_count
                + self.bellman_target_nonfinite_count
                + self.td_error_nonfinite_count
                + self.loss_nonfinite_count
                + self.nonfinite_gradient_count
            ),
            "optimizer_apply_attempted": False,
            "optimizer_apply_succeeded": False,
            "weights_mutated": False,
            "optimizer_state_mutated": False,
            "target_network_updated": False,
        }


@dataclass(frozen=True)
class FatalTrainResultV01:
    """Runtime-neutral fatal result for a rejected train attempt."""

    fatal: bool
    success: bool
    reason: str
    execution_mode: str
    runtime_mode: str
    train_attempt_index: int
    last_successful_train_step: int
    replay_size: int
    committed_transition_count: int
    pending_train_credit: int
    optimizer_apply_attempted: bool
    weights_mutated: bool
    affected_variable_names: tuple[str, ...]
    nonfinite_tensor_count: int
    last_error: str
    finite_gate_diagnostics: dict[str, Any]

    def diagnostics(self) -> dict[str, Any]:
        return {
            "fatal": bool(self.fatal),
            "success": bool(self.success),
            "reason": self.reason,
            "execution_mode": self.execution_mode,
            "runtime_mode": self.runtime_mode,
            "train_attempt_index": int(self.train_attempt_index),
            "last_successful_train_step": int(
                self.last_successful_train_step
            ),
            "replay_size": int(self.replay_size),
            "committed_transition_count": int(
                self.committed_transition_count
            ),
            "pending_train_credit": int(self.pending_train_credit),
            "optimizer_apply_attempted": bool(
                self.optimizer_apply_attempted
            ),
            "weights_mutated": bool(self.weights_mutated),
            "affected_variable_names": list(
                self.affected_variable_names
            ),
            "nonfinite_tensor_count": int(self.nonfinite_tensor_count),
            "last_error": self.last_error,
            "finite_gate": dict(self.finite_gate_diagnostics),
        }


class TrainFiniteGateViolationV01(RuntimeError):
    """Structured fatal result raised before optimizer mutation."""

    def __init__(self, result: PreOptimizerFiniteGateResultV01):
        if result.passed:
            raise ValueError("TrainFiniteGateViolationV01 requires a failed result")
        self.result = result
        self.reason = result.reason
        self.diagnostics = result.diagnostics()
        super().__init__(f"pre-optimizer finite gate failed: {result.reason}")


def enforce_pre_optimizer_finite_gate_v01(
    *,
    online_q_values: Any,
    selected_action_q_values: Any,
    target_q_values: Any,
    bellman_targets: Any,
    td_errors: Any,
    loss: Any,
    gradients: Sequence[Any | None],
    trainable_variables: Sequence[Any],
) -> PreOptimizerFiniteGateResultV01:
    """Validate a formal train attempt and raise before any mutation."""

    result = validate_pre_optimizer_finite_gate_v01(
        online_q_values=online_q_values,
        selected_action_q_values=selected_action_q_values,
        target_q_values=target_q_values,
        bellman_targets=bellman_targets,
        td_errors=td_errors,
        loss=loss,
        gradients=gradients,
        trainable_variables=trainable_variables,
    )
    if not result.passed:
        raise TrainFiniteGateViolationV01(result)
    return result


def build_fatal_train_result_v01(
    violation: TrainFiniteGateViolationV01,
    *,
    execution_mode: str,
    runtime_mode: str,
    train_attempt_index: int,
    last_successful_train_step: int,
    replay_size: int,
    committed_transition_count: int,
    pending_train_credit: int,
) -> FatalTrainResultV01:
    """Attach compact runtime context to a common finite-gate failure."""

    diagnostics = dict(violation.diagnostics)
    return FatalTrainResultV01(
        fatal=True,
        success=False,
        reason=violation.reason,
        execution_mode=str(execution_mode),
        runtime_mode=str(runtime_mode),
        train_attempt_index=int(train_attempt_index),
        last_successful_train_step=int(last_successful_train_step),
        replay_size=int(replay_size),
        committed_transition_count=int(committed_transition_count),
        pending_train_credit=int(pending_train_credit),
        optimizer_apply_attempted=False,
        weights_mutated=False,
        affected_variable_names=tuple(
            str(value)
            for value in diagnostics.get("affected_variable_names", ())
        ),
        nonfinite_tensor_count=int(
            diagnostics.get("nonfinite_tensor_count", 0)
        ),
        last_error=(
            f"fatal_train_failure:{violation.reason}"
        ),
        finite_gate_diagnostics=diagnostics,
    )


def fatal_train_metrics_v01(
    result: FatalTrainResultV01 | None,
    *,
    last_successful_train_step: int,
    pending_train_credit: int,
) -> dict[str, Any]:
    """Return the common compact metrics surface for every runtime."""

    gate = (
        dict(result.finite_gate_diagnostics)
        if result is not None
        else {}
    )
    return {
        "nonfinite_online_q_count": int(
            gate.get("online_q_nonfinite_count", 0)
        )
        + int(gate.get("selected_q_nonfinite_count", 0)),
        "nonfinite_target_q_count": int(
            gate.get("target_q_nonfinite_count", 0)
        ),
        "nonfinite_bellman_target_count": int(
            gate.get("bellman_target_nonfinite_count", 0)
        ),
        "nonfinite_td_error_count": int(
            gate.get("td_error_nonfinite_count", 0)
        ),
        "nonfinite_loss_count": int(
            gate.get("loss_nonfinite_count", 0)
        ),
        "missing_gradient_count": int(
            gate.get("missing_gradient_count", 0)
        ),
        "nonfinite_gradient_count": int(
            gate.get("nonfinite_gradient_count", 0)
        ),
        "empty_gradient_set_count": int(
            gate.get("empty_gradient_set_count", 0)
        ),
        "fatal_train_failure_count": int(result is not None),
        "optimizer_apply_skipped_count": int(result is not None),
        "last_fatal_train_reason": (
            result.reason if result is not None else None
        ),
        "last_successful_train_step": int(last_successful_train_step),
        "pending_train_credit": int(pending_train_credit),
        "shutdown_reason": (
            result.last_error if result is not None else "completed"
        ),
        "success": result is None,
        "last_error": (
            result.last_error if result is not None else None
        ),
    }


def validate_pre_optimizer_finite_gate_v01(
    *,
    online_q_values: Any,
    selected_action_q_values: Any,
    target_q_values: Any,
    bellman_targets: Any,
    td_errors: Any,
    loss: Any,
    gradients: Sequence[Any | None],
    trainable_variables: Sequence[Any],
) -> PreOptimizerFiniteGateResultV01:
    """Validate all train intermediates and gradient correspondence."""

    variables = list(trainable_variables)
    gradient_values = list(gradients)
    online_count = _nonfinite_count(online_q_values)
    selected_count = _nonfinite_count(selected_action_q_values)
    target_count = _nonfinite_count(target_q_values)
    bellman_count = _nonfinite_count(bellman_targets)
    td_count = _nonfinite_count(td_errors)
    loss_count = _nonfinite_count(loss)

    missing_indices = [
        index
        for index in range(len(variables))
        if index >= len(gradient_values) or gradient_values[index] is None
    ]
    extra_gradient_count = max(0, len(gradient_values) - len(variables))
    non_none = [
        gradient
        for gradient in gradient_values[: len(variables)]
        if gradient is not None
    ]
    gradient_nonfinite_counts = [
        _nonfinite_count(gradient) for gradient in non_none
    ]
    nonfinite_gradient_count = sum(gradient_nonfinite_counts)
    affected_names: list[str] = [
        _variable_name(variables[index], index) for index in missing_indices
    ]
    affected_names.extend(
        f"unexpected_gradient_{index}"
        for index in range(
            len(variables),
            len(variables) + extra_gradient_count,
        )
    )
    non_none_index = 0
    for index, gradient in enumerate(gradient_values[: len(variables)]):
        if gradient is None:
            continue
        if gradient_nonfinite_counts[non_none_index] > 0:
            affected_names.append(_variable_name(variables[index], index))
        non_none_index += 1

    gradient_norm = None
    if non_none:
        gradient_norm = float(tf.linalg.global_norm(non_none).numpy())
        if not math.isfinite(gradient_norm) and nonfinite_gradient_count == 0:
            nonfinite_gradient_count = 1
            affected_names.append("global_gradient_norm")

    reason = finite_gate_reason_v01(
        online_q_nonfinite_count=online_count + selected_count,
        target_q_nonfinite_count=target_count,
        bellman_target_nonfinite_count=bellman_count,
        td_error_nonfinite_count=td_count,
        loss_nonfinite_count=loss_count,
        gradient_count=len(non_none),
        trainable_variable_count=len(variables),
        missing_gradient_count=len(missing_indices) + extra_gradient_count,
        nonfinite_gradient_count=nonfinite_gradient_count,
    )
    loss_value = _finite_scalar_or_none(loss)
    return PreOptimizerFiniteGateResultV01(
        passed=reason == "ok",
        reason=reason,
        online_q_nonfinite_count=online_count,
        selected_q_nonfinite_count=selected_count,
        target_q_nonfinite_count=target_count,
        bellman_target_nonfinite_count=bellman_count,
        td_error_nonfinite_count=td_count,
        loss_nonfinite_count=loss_count,
        missing_gradient_count=len(missing_indices) + extra_gradient_count,
        nonfinite_gradient_count=nonfinite_gradient_count,
        gradient_count=len(non_none),
        trainable_variable_count=len(variables),
        gradient_norm=gradient_norm,
        loss=loss_value,
        affected_variable_names=tuple(dict.fromkeys(affected_names)),
    )


def finite_gate_reason_v01(
    *,
    online_q_nonfinite_count: int,
    target_q_nonfinite_count: int,
    bellman_target_nonfinite_count: int,
    td_error_nonfinite_count: int,
    loss_nonfinite_count: int,
    gradient_count: int,
    trainable_variable_count: int,
    missing_gradient_count: int,
    nonfinite_gradient_count: int,
) -> str:
    if int(online_q_nonfinite_count) > 0:
        return "nonfinite_online_q"
    if int(target_q_nonfinite_count) > 0:
        return "nonfinite_target_q"
    if int(bellman_target_nonfinite_count) > 0:
        return "nonfinite_bellman_target"
    if int(td_error_nonfinite_count) > 0:
        return "nonfinite_td_error"
    if int(loss_nonfinite_count) > 0:
        return "nonfinite_loss"
    if int(trainable_variable_count) <= 0:
        return "empty_gradient_set"
    if int(missing_gradient_count) > 0:
        return "missing_gradient"
    if int(gradient_count) <= 0:
        return "empty_gradient_set"
    if int(nonfinite_gradient_count) > 0:
        return "nonfinite_gradient"
    return "ok"


def finite_gate_reason_from_code_v01(code: int) -> str:
    return FINITE_GATE_REASON_BY_CODE_V01.get(int(code), "invalid_finite_gate_reason")


def _nonfinite_count(value: Any) -> int:
    tensor = tf.cast(tf.convert_to_tensor(value), tf.float32)
    return int(
        tf.math.count_nonzero(tf.logical_not(tf.math.is_finite(tensor))).numpy()
    )


def _variable_name(variable: Any, index: int) -> str:
    name = getattr(variable, "name", None)
    return str(name) if name else f"trainable_variable_{index}"


def _finite_scalar_or_none(value: Any) -> float | None:
    tensor = tf.cast(tf.convert_to_tensor(value), tf.float32)
    if int(tf.size(tensor).numpy()) != 1:
        return None
    normalized = float(tf.reshape(tensor, []).numpy())
    return normalized if math.isfinite(normalized) else None
