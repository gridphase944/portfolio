"""I12 public source subset. Historical semantic bodies retained; private orchestration excluded."""


from __future__ import annotations

from dataclasses import dataclass

from collections import Counter

import hashlib

import json

import math

from pathlib import Path

import time

from typing import Any, Mapping, Sequence

import numpy as np

import pandas as pd

from features.state_vector_v01 import STATE_CANDIDATE_LAG1_AGENT_TRANSITION_CONTEXT_V01, STATE_CANDIDATE_LAG2_AGENT_TRANSITION_CONTEXT_V01, STATE_CANDIDATE_LAG3_AGENT_TRANSITION_CONTEXT_V01, get_state_columns_v01

from trading_env.env_v01 import TradingEnv

from trading_env.execution_latency_v01 import sidecar_env_kwargs_v01

from trading_env.execution_closingrisk_v02 import EXECUTION_CONTRACT_ID as CLOSINGRISK_EXECUTION_CONTRACT_ID, policy_effective_action_mask_v02, risk_management_action_v02

from training.dqn_loss_helper_v01 import compute_per_sample_dqn_loss_v01, gather_q_by_action_v01

from training.ddqn_backup_v01 import compute_next_backup_comparison_v01, mechanism_telemetry_row_v01, validate_backup_algorithm_v01

from training.dqn_target_helper_v01 import masked_reduce_max_next_q_v01

from training.epsilon_scheduler_v01 import compute_epsilon_v01

from training.replay_buffer_v01 import encode_action_v01

from training.seed_utils_v01 import set_global_seed_v01

from training.rl_representation_observability_v01 import E0ObservabilityRecorderV01

from training.rl_representation_integration_v01 import RepresentationReadinessResultV01, build_q_input_batch_v01, build_q_input_v01, build_semantic_state_v01, build_variant_q_network_v01, contract_for_variant_v01

from training.rl_representation_replay_v01 import ExcludedStepValidityV01, RepresentationEpisodeAdmissionV01, RepresentationReplayBufferV01

from training.rl_representation_rollout_v01 import E0EpisodeAccountingV01, RepresentationActionDecisionV01, exclude_policy_transition_for_next_readiness_v01, select_representation_action_v01

from training.state_normalizer_v01 import TRANSFORM_CLIP, TRANSFORM_LOG1P, fit_state_normalizer_v01

from training.target_network_utils_v01 import hard_update_target_network_v01

from training.train_finite_gate_v01 import enforce_pre_optimizer_finite_gate_v01

APPROVED_DQN_GAMMAS = (0.99, 0.9962, 1.0)

def file_sha256_v01(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _validate_approved_gamma_v01(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("DQN gamma must be a numeric scalar")
    gamma = float(value)
    if not math.isfinite(gamma) or gamma not in APPROVED_DQN_GAMMAS:
        raise ValueError(
            f"DQN gamma must be one of {list(APPROVED_DQN_GAMMAS)}"
        )
    return gamma

@dataclass
class _EpisodeArtifactV01:
    episode_df: pd.DataFrame
    ready: np.ndarray
    embeddings: np.ndarray
    readiness_reasons: tuple[tuple[str, ...], ...]
    identity: Mapping[str, Any]

class _FastStateNormalizerV01:
    def __init__(self, state_columns: Sequence[str] | None = None):
        columns = list(state_columns or get_state_columns_v01("current_state"))
        state_dim = len(columns)
        self.fitted = fit_state_normalizer_v01(
            np.zeros((1, state_dim), dtype=np.float64),
            columns,
            np.ones((1, state_dim), dtype=bool),
        )
        self.clip_indices = np.asarray(
            [
                index
                for index, spec in enumerate(self.fitted.feature_specs)
                if spec.transform_type == TRANSFORM_CLIP
            ],
            dtype=np.int64,
        )
        self.log_indices = np.asarray(
            [
                index
                for index, spec in enumerate(self.fitted.feature_specs)
                if spec.transform_type == TRANSFORM_LOG1P
            ],
            dtype=np.int64,
        )

    def transform(self, state: np.ndarray, mask: np.ndarray) -> np.ndarray:
        values = np.asarray(state, dtype=np.float64).copy()
        valid = np.asarray(mask, dtype=bool) & np.isfinite(values)
        values[~valid] = np.nan
        indices = self.clip_indices
        selected = indices[valid[indices]]
        values[selected] = np.clip(values[selected], -0.2, 0.2)
        indices = self.log_indices
        selected = indices[valid[indices]]
        values[selected] = np.log1p(np.maximum(values[selected], 0.0))
        return values.astype(np.float32, copy=False)

    def transform_batch(self, states: np.ndarray, masks: np.ndarray) -> np.ndarray:
        values = np.asarray(states, dtype=np.float64).copy()
        valid = np.asarray(masks, dtype=bool) & np.isfinite(values)
        values[~valid] = np.nan
        if len(self.clip_indices):
            block = values[:, self.clip_indices]
            block_valid = valid[:, self.clip_indices]
            values[:, self.clip_indices] = np.where(
                block_valid, np.clip(block, -0.2, 0.2), block
            )
        if len(self.log_indices):
            block = values[:, self.log_indices]
            block_valid = valid[:, self.log_indices]
            values[:, self.log_indices] = np.where(
                block_valid, np.log1p(np.maximum(block, 0.0)), block
            )
        return values.astype(np.float32, copy=False)

def _state_candidate_for_variant_v01(variant: str) -> str:
    return {
        "LAG1": STATE_CANDIDATE_LAG1_AGENT_TRANSITION_CONTEXT_V01,
        "LAG2": STATE_CANDIDATE_LAG2_AGENT_TRANSITION_CONTEXT_V01,
        "LAG3": STATE_CANDIDATE_LAG3_AGENT_TRANSITION_CONTEXT_V01,
    }.get(str(variant), "current_state")

def _one_step_bellman_targets_v01(
    *, rewards: Any, dones: Any, max_next_q: Any, gamma: float
) -> Any:
    import tensorflow as tf

    actual_gamma = _validate_approved_gamma_v01(gamma)
    reward_values = tf.convert_to_tensor(rewards, dtype=tf.float32)
    done_values = tf.convert_to_tensor(dones, dtype=tf.bool)
    next_values = tf.convert_to_tensor(max_next_q, dtype=tf.float32)
    safe_next_values = tf.where(
        done_values, tf.zeros_like(next_values), next_values
    )
    return reward_values + tf.constant(actual_gamma, tf.float32) * tf.cast(
        tf.logical_not(done_values), tf.float32
    ) * safe_next_values

class _E0LearnerV01:
    def __init__(
        self,
        *,
        variant: str,
        seed: int,
        gamma: float,
        recorder: E0ObservabilityRecorderV01,
        backup_algorithm: str = "dqn",
        mechanism_telemetry_enabled: bool = False,
        target_sync_interval: int = 200,
    ):
        import tensorflow as tf

        tf.keras.backend.clear_session()
        set_global_seed_v01(seed)
        self.variant = variant
        self.contract = contract_for_variant_v01(variant)
        self.seed = int(seed)
        self.gamma = _validate_approved_gamma_v01(gamma)
        self.backup_algorithm = validate_backup_algorithm_v01(
            backup_algorithm
        )
        self.mechanism_telemetry_enabled = bool(
            mechanism_telemetry_enabled
        )
        self.target_sync_interval = int(target_sync_interval)
        self.target_action_mask_source = "env"
        _require(
            self.target_sync_interval == 200,
            "E0 target sync interval must remain 200",
        )
        recorder_gamma = _validate_approved_gamma_v01(
            recorder.run_identity.get("gamma")
        )
        _require(
            recorder_gamma == self.gamma,
            "learner/telemetry gamma mismatch",
        )
        recorder_backup = validate_backup_algorithm_v01(
            recorder.run_identity.get("backup_algorithm", "dqn")
        )
        _require(
            recorder_backup == self.backup_algorithm,
            "learner/telemetry backup algorithm mismatch",
        )
        self.policy_rng = np.random.default_rng(seed)
        self.replay = RepresentationReplayBufferV01(
            capacity=50000, variant=variant, seed=seed
        )
        self.admission = RepresentationEpisodeAdmissionV01(self.replay)
        self.q_network = build_variant_q_network_v01(variant)
        self.target_network = build_variant_q_network_v01(variant)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        hard_update_target_network_v01(self.q_network, self.target_network)
        self.initial_online_weight_identity_sha256 = _network_weight_hash_v01(
            self.q_network
        )
        self.initial_target_weight_identity_sha256 = _network_weight_hash_v01(
            self.target_network
        )
        _require(
            self.initial_online_weight_identity_sha256
            == self.initial_target_weight_identity_sha256,
            "initial online/target hard sync mismatch",
        )
        normalizer_columns = get_state_columns_v01(
            _state_candidate_for_variant_v01(self.variant)
        )
        self.normalizer = _FastStateNormalizerV01(normalizer_columns)
        self.recorder = recorder
        self.global_env_step = 0
        self.optimizer_updates = 0
        self.target_update_count = 1
        self.last_target_sync_update = 0
        self.train_credit_consumed = 0
        self.first_policy_epsilon: float | None = None
        self.greedy_since_update = 0
        self.random_since_update = 0
        self.action_position_since_update: Counter[str] = Counter()
        self._numpy_weights: tuple[np.ndarray, ...] = ()
        self.mechanism_telemetry_rows: list[dict[str, Any]] = []
        self.actual_device = "CPU"
        self.refresh_numpy_weights()

    def refresh_numpy_weights(self) -> None:
        weights = tuple(np.asarray(x, dtype=np.float32) for x in self.q_network.get_weights())
        _require(len(weights) == 6, "Q network weight layout mismatch")
        self._numpy_weights = weights

    def q_values(self, q_input: np.ndarray) -> np.ndarray:
        w1, b1, w2, b2, w3, b3 = self._numpy_weights
        x = np.maximum(np.asarray(q_input, dtype=np.float32) @ w1 + b1, 0.0)
        x = np.maximum(x @ w2 + b2, 0.0)
        q = x @ w3 + b3
        _require(q.shape == (3,) and np.isfinite(q).all(), "online Q finite/shape gate failed")
        return q

    def select_action(
        self,
        *,
        q_input: np.ndarray,
        action_mask: np.ndarray,
        position_before: str,
    ) -> tuple[str, np.ndarray, float, str]:
        epsilon = compute_epsilon_v01(
            global_env_step=self.global_env_step,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay_steps=20000,
        )
        if self.first_policy_epsilon is None:
            self.first_policy_epsilon = float(epsilon)
        valid_ids = np.flatnonzero(action_mask)
        _require(len(valid_ids) > 0, "policy action mask all false")
        q = self.q_values(q_input)
        if float(self.policy_rng.random()) < epsilon:
            action_id = int(self.policy_rng.choice(valid_ids))
            source = "random_valid"
            self.random_since_update += 1
        else:
            masked = np.where(action_mask, q, -np.inf)
            action_id = int(np.argmax(masked))
            source = "greedy"
            self.greedy_since_update += 1
        action = ("Hold", "Buy", "Sell")[action_id]
        self.action_position_since_update[f"{action}|{position_before}"] += 1
        return action, q, float(epsilon), source

    @property
    def pending_train_credit(self) -> int:
        earned = self.admission.stats.replay_committed_count // 4
        return max(0, earned - self.train_credit_consumed)

    def train_one(self, *, epsilon: float) -> dict[str, Any]:
        import tensorflow as tf

        _require(len(self.replay) >= 1024, "Replay train-start gate failed")
        _require(self.pending_train_credit > 0, "train credit unavailable")
        batch = self.replay.sample(128)
        normalized = _normalized_current_batch_from_semantic_v01(
            self.variant,
            batch.states,
            batch.state_valid_masks,
            self.normalizer,
        )
        normalized_next = _normalized_current_batch_from_semantic_v01(
            self.variant,
            batch.next_states,
            batch.next_state_valid_masks,
            self.normalizer,
        )
        current_inputs = build_q_input_batch_v01(
            self.variant,
            semantic_states=batch.states,
            valid_state_masks=batch.state_valid_masks,
            normalized_current_states=normalized,
        )
        next_inputs = build_q_input_batch_v01(
            self.variant,
            semantic_states=batch.next_states,
            valid_state_masks=batch.next_state_valid_masks,
            normalized_current_states=normalized_next,
        )
        actions = tf.convert_to_tensor(batch.actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(batch.rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(batch.dones, dtype=tf.bool)
        if self.target_action_mask_source == "policy_effective":
            selected_next_masks = batch.next_policy_effective_action_masks
        elif self.target_action_mask_source == "env":
            selected_next_masks = batch.next_env_action_masks
        else:
            raise RuntimeError(
                f"unsupported target action mask source: {self.target_action_mask_source}"
            )
        next_masks = tf.convert_to_tensor(selected_next_masks, dtype=tf.bool)
        _require(
            not np.any(
                (~batch.dones) & (~np.any(selected_next_masks, axis=1))
            ),
            "nonterminal next action mask all false",
        )
        before_hash = _network_weight_hash_v01(self.q_network)
        pre_update_index = self.optimizer_updates
        last_target_sync_update = self.last_target_sync_update
        comparison = None
        online_next_q_values = None
        with tf.GradientTape() as tape:
            q_values = tf.convert_to_tensor(
                self.q_network(current_inputs, training=True), dtype=tf.float32
            )
            chosen_q = gather_q_by_action_v01(q_values, actions)
            target_q_values = tf.convert_to_tensor(
                self.target_network(next_inputs, training=False), dtype=tf.float32
            )
            target_q_values = tf.stop_gradient(target_q_values)
            if (
                self.backup_algorithm == "double_dqn"
                or self.mechanism_telemetry_enabled
            ):
                online_next_q_values = tf.stop_gradient(
                    tf.convert_to_tensor(
                        self.q_network(next_inputs, training=False),
                        dtype=tf.float32,
                    )
                )
                comparison = compute_next_backup_comparison_v01(
                    online_next_q_values=online_next_q_values,
                    target_next_q_values=target_q_values,
                    dones=dones,
                    next_action_masks=next_masks,
                    backup_algorithm=self.backup_algorithm,
                )
                max_next = comparison.selected_bootstrap
            else:
                max_next = masked_reduce_max_next_q_v01(
                    target_q_values, next_masks
                )
            bellman = _one_step_bellman_targets_v01(
                rewards=rewards,
                dones=dones,
                max_next_q=max_next,
                gamma=self.gamma,
            )
            td_error = bellman - chosen_q
            per_sample = compute_per_sample_dqn_loss_v01(td_error, loss_type="huber")
            loss = tf.reduce_mean(per_sample)
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        before_norm = float(tf.linalg.global_norm([x for x in gradients if x is not None]).numpy())
        non_none = [x for x in gradients if x is not None]
        clipped_values, _ = tf.clip_by_global_norm(non_none, 10.0)
        iterator = iter(clipped_values)
        clipped = [None if value is None else next(iterator) for value in gradients]
        after_norm = float(tf.linalg.global_norm([x for x in clipped if x is not None]).numpy())
        self.actual_device = str(getattr(q_values, "device", "CPU") or "CPU")
        enforce_pre_optimizer_finite_gate_v01(
            online_q_values=q_values,
            selected_action_q_values=chosen_q,
            target_q_values=target_q_values,
            bellman_targets=bellman,
            td_errors=td_error,
            loss=loss,
            gradients=clipped,
            trainable_variables=self.q_network.trainable_variables,
        )
        pairs = [
            (gradient, variable)
            for gradient, variable in zip(
                clipped, self.q_network.trainable_variables, strict=True
            )
            if gradient is not None
        ]
        self.optimizer.apply_gradients(pairs)
        _require(
            _network_weight_hash_v01(self.q_network) != before_hash,
            "optimizer valid step did not mutate Q network",
        )
        self.train_credit_consumed += 1
        self.optimizer_updates += 1
        scheduled_target = False
        if self.optimizer_updates % self.target_sync_interval == 0:
            hard_update_target_network_v01(self.q_network, self.target_network)
            self.target_update_count += 1
            self.last_target_sync_update = self.optimizer_updates
            scheduled_target = True
        if self.mechanism_telemetry_enabled:
            _require(
                comparison is not None and online_next_q_values is not None,
                "mechanism telemetry comparison missing",
            )
            self.mechanism_telemetry_rows.append(
                dict(
                    mechanism_telemetry_row_v01(
                        comparison=comparison,
                        online_next_q_values=np.asarray(
                            online_next_q_values.numpy()
                        ),
                        target_next_q_values=np.asarray(
                            target_q_values.numpy()
                        ),
                        dones=batch.dones,
                        next_action_masks=selected_next_masks,
                        backup_algorithm=self.backup_algorithm,
                        gamma=self.gamma,
                        pre_update_index=pre_update_index,
                        last_target_sync_update=last_target_sync_update,
                        target_sync_interval=self.target_sync_interval,
                        train_loss=float(loss.numpy()),
                        abs_td_values=np.abs(td_error.numpy()),
                        gradient_global_norm_before_clip=before_norm,
                        gradient_global_norm_after_clip=after_norm,
                        learning_rate=0.001,
                    )
                )
            )
        self.refresh_numpy_weights()
        self.recorder.record_learning_update(
            optimizer_update_index=self.optimizer_updates,
            q_values=np.asarray(q_values.numpy()),
            chosen_action_ids=batch.actions,
            valid_action_masks=batch.action_masks,
            td_errors=np.asarray(td_error.numpy()),
            bellman_targets=np.asarray(bellman.numpy()),
            loss=float(loss.numpy()),
            epsilon=float(epsilon),
            gradient_global_norm_before_clip=before_norm,
            gradient_global_norm_after_clip=after_norm,
            learning_rate=0.001,
            replay_size=len(self.replay),
            greedy_decision_count=self.greedy_since_update,
            random_valid_decision_count=self.random_since_update,
            fallback_source_counts={},
            action_by_position_before_counts=self.action_position_since_update,
        )
        self.greedy_since_update = 0
        self.random_since_update = 0
        self.action_position_since_update.clear()
        return {
            "update": self.optimizer_updates,
            "gamma": self.gamma,
            "backup_algorithm": self.backup_algorithm,
            "loss": float(loss.numpy()),
            "bellman_target_mean": float(
                np.asarray(bellman.numpy(), dtype=np.float64).mean()
            ),
            "td_error_abs_mean": float(np.abs(td_error.numpy()).mean()),
            "gradient_norm_before": before_norm,
            "gradient_norm_after": after_norm,
            "target_updated": scheduled_target,
            "pre_update_index": pre_update_index,
            "sync_age": pre_update_index - last_target_sync_update,
        }

def _network_weight_hash_v01(network: Any) -> str:
    digest = hashlib.sha256()
    for value in network.get_weights():
        digest.update(np.asarray(value).tobytes(order="C"))
    return digest.hexdigest()

def _normalized_current_batch_from_semantic_v01(
    variant: str,
    semantic: np.ndarray,
    masks: np.ndarray,
    normalizer: _FastStateNormalizerV01,
) -> np.ndarray:
    count = len(semantic)
    if variant in {"B0", "LAG1", "LAG2", "LAG3"}:
        current = np.asarray(semantic, dtype=np.float32)
        current_masks = np.asarray(masks, dtype=bool)
    elif variant == "A1":
        current = np.concatenate([semantic[:, :50], semantic[:, -9:]], axis=1)
        current_masks = np.asarray(masks, dtype=bool)
    else:
        current = np.zeros((count, 59), dtype=np.float32)
        current[:, 50:] = semantic[:, -9:]
        current_masks = np.zeros((count, 59), dtype=bool)
        current_masks[:, 50:] = masks
    normalized = normalizer.transform_batch(current, current_masks)
    return normalized[:, 50:] if variant == "R1" else normalized

def _state_mask_from_env_reset_v01(env: TradingEnv, state: np.ndarray) -> np.ndarray:
    result = getattr(env, "_last_state_result", None)
    qc = getattr(result, "qc", None)
    if not isinstance(qc, Mapping) or "valid_state_mask" not in qc:
        raise RuntimeError("Env reset state mask contract unavailable")
    mask = np.asarray(qc["valid_state_mask"], dtype=bool)
    _require(
        mask.shape == np.asarray(state).shape,
        "Env reset state mask shape mismatch",
    )
    return mask

def _action_mask_q_order_v01(env: TradingEnv) -> np.ndarray:
    values = env.get_action_mask()
    result = np.asarray(
        [values["Hold"], values["Buy"], values["Sell"]], dtype=bool
    )
    _require(result.any(), "Env action mask all false")
    return result

def _runtime_action_masks_v01(
    env: TradingEnv, row: Mapping[str, Any]
) -> dict[str, Any]:
    """Return separate Env and outer policy-effective masks.

    Existing/default routes resolve both fields to the unchanged Env mask.
    Only the explicit closing-risk v02 contract activates the outer gate.
    """

    env_mask = _action_mask_q_order_v01(env)
    if getattr(env, "execution_contract_id", None) != CLOSINGRISK_EXECUTION_CONTRACT_ID:
        return {
            "env_action_mask": env_mask,
            "policy_effective_action_mask": env_mask.copy(),
            "entry_cutoff_active": False,
            "entry_cutoff_gate_id": None,
            "policy_effective_mask_reason": None,
            "blocked_action_slot_count": 0,
            "env_valid_action_slot_count": int(env_mask.sum()),
            "is_forced_liquidation_event": False,
        }
    return policy_effective_action_mask_v02(
        env_action_mask=env_mask,
        position=env.position,
        decision_timestamp=row["grid_timestamp"],
        is_forced_liquidation_event=bool(row.get("is_forced_exit", False)),
    )

def _is_closingrisk_forced_event_v01(
    env: TradingEnv, row: Mapping[str, Any]
) -> bool:
    return bool(
        getattr(env, "execution_contract_id", None)
        == CLOSINGRISK_EXECUTION_CONTRACT_ID
        and row.get("is_forced_exit", False)
    )

def _readiness_for_row_v01(
    artifact: _EpisodeArtifactV01, index: int, symbol: str, date: str
) -> RepresentationReadinessResultV01:
    row = artifact.episode_df.iloc[index]
    return RepresentationReadinessResultV01(
        ready=bool(artifact.ready[index]),
        timestamp=pd.Timestamp(row["grid_timestamp"]).isoformat(),
        symbol=symbol,
        trading_date=date,
        session=str(row["session"]),
        reasons=artifact.readiness_reasons[index],
    )

def _semantic_state_and_valid_mask_v01(
    learner: _E0LearnerV01,
    *,
    current_state: np.ndarray,
    current_mask: np.ndarray,
    embedding: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    variant = learner.variant
    semantic = build_semantic_state_v01(
        variant,
        current_state=np.asarray(current_state),
        board_embedding=embedding[:32] if variant in {"R1", "A1"} else None,
        trade_embedding=embedding[32:64] if variant in {"R1", "A1"} else None,
        market_embedding=embedding[64:96] if variant in {"R1", "A1"} else None,
    )
    valid_mask = (
        np.asarray(current_mask[50:], dtype=bool)
        if variant == "R1"
        else np.asarray(current_mask, dtype=bool)
    )
    return semantic, valid_mask

def _semantic_and_q_input_v01(
    learner: _E0LearnerV01,
    *,
    current_state: np.ndarray,
    current_mask: np.ndarray,
    embedding: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    variant = learner.variant
    semantic, valid_mask = _semantic_state_and_valid_mask_v01(
        learner,
        current_state=current_state,
        current_mask=current_mask,
        embedding=embedding,
    )
    normalized_current = learner.normalizer.transform(current_state, current_mask)
    normalized = (
        normalized_current[50:] if variant == "R1" else normalized_current
    )
    q_input = build_q_input_v01(
        variant,
        semantic_state=semantic,
        valid_state_mask=valid_mask,
        normalized_current_state=normalized,
    )
    return semantic, valid_mask, q_input

def _validate_agent_history_state_v01(
    *,
    state: np.ndarray,
    state_mask: np.ndarray,
    history_context: Mapping[str, Any],
    history_depth: int,
    expected_available_groups: int,
) -> list[bool]:
    """Fail closed on per-group availability and one-hot semantics."""

    if history_depth not in {1, 2, 3}:
        raise ValueError("history_depth must be 1, 2, or 3")
    values = np.asarray(state[-6 * history_depth:], dtype=np.float64).reshape(
        history_depth, 6
    )
    masks = np.asarray(state_mask[-6 * history_depth:], dtype=bool).reshape(
        history_depth, 6
    )
    groups = list(history_context.get("groups", ()))
    _require(len(groups) == history_depth, "Agent History group count mismatch")
    availability: list[bool] = []
    position_codes = {
        "Flat": (1.0, 0.0, 0.0),
        "Long": (0.0, 1.0, 0.0),
        "Short": (0.0, 0.0, 1.0),
    }
    action_codes = {
        "Hold": (1.0, 0.0, 0.0),
        "Buy": (0.0, 1.0, 0.0),
        "Sell": (0.0, 0.0, 1.0),
    }
    for lag, (group_values, group_masks, group) in enumerate(
        zip(values, masks, groups, strict=True), start=1
    ):
        available = bool(group.get("available", False))
        availability.append(available)
        _require(
            available == (lag <= expected_available_groups),
            f"Lag-{lag} staged availability mismatch",
        )
        if not available:
            _require(
                (~group_masks).all() and np.isnan(group_values).all(),
                f"unavailable Lag-{lag} must be NaN/all-false",
            )
            continue
        position = group.get("position_before")
        action = group.get("effective_action")
        expected_position = position_codes.get(position)
        expected_action = action_codes.get(action)
        _require(
            group_masks.all()
            and np.isfinite(group_values).all()
            and expected_position is not None
            and expected_action is not None
            and np.array_equal(
                group_values, np.asarray(expected_position + expected_action)
            ),
            f"Lag-{lag} available history semantic/mask invariant failed",
        )
    return availability

def _run_training_episode_v01(
    *,
    learner: _E0LearnerV01,
    artifact: _EpisodeArtifactV01,
    date: str,
    symbol: str,
    run_phase: str,
) -> dict[str, Any]:
    episode_id = f"{date}_{symbol}"
    env = TradingEnv(
        artifact.episode_df,
        step_seconds=5,
        entry_reentry_cooldown_steps=0,
        entry_reentry_cooldown_apply_to="off",
        entry_reentry_cooldown_state_enabled=False,
        entry_reentry_cooldown_session_boundary="continue",
        state_candidate_name=_state_candidate_for_variant_v01(learner.variant),
        **sidecar_env_kwargs_v01(artifact.episode_df),
    )
    state = np.asarray(env.reset(), dtype=np.float32)
    state_mask = _state_mask_from_env_reset_v01(env, state)
    accounting = E0EpisodeAccountingV01(
        episode_id=episode_id, symbol=symbol, trading_date=date
    )
    exclusions: Counter[str] = Counter()
    next_reason_counts: Counter[str] = Counter()
    q_sample_count = 0
    q_sum = np.zeros(3, dtype=np.float64)
    history_depth = {"LAG1": 1, "LAG2": 2, "LAG3": 3}.get(
        learner.variant, 0
    )
    history_available_counts: Counter[int] = Counter()
    history_unavailable_counts: Counter[int] = Counter()
    history_first_unavailable: Counter[str] = Counter()
    previous_session: str | None = None
    session_decision_index = 0
    done = False
    step_index = 0
    while not done:
        row = artifact.episode_df.iloc[step_index]
        session = str(row["session"])
        session_first = previous_session is None or session != previous_session
        session_decision_index = 0 if session_first else session_decision_index + 1
        if history_depth:
            availability = _validate_agent_history_state_v01(
                state=state,
                state_mask=state_mask,
                history_context=env.get_agent_transition_history_context(),
                history_depth=history_depth,
                expected_available_groups=min(
                    session_decision_index, history_depth
                ),
            )
            for lag, available in enumerate(availability, start=1):
                if available:
                    history_available_counts[lag] += 1
                else:
                    history_unavailable_counts[lag] += 1
            if session_first:
                _require(not any(availability), "session first history must be unavailable")
                history_first_unavailable[session] += 1
        readiness = _readiness_for_row_v01(artifact, step_index, symbol, date)
        position_before = env.position
        mask_contract = _runtime_action_masks_v01(env, row)
        env_action_mask = np.asarray(
            mask_contract["env_action_mask"], dtype=bool
        )
        action_mask = np.asarray(
            mask_contract["policy_effective_action_mask"], dtype=bool
        )
        risk_event = _is_closingrisk_forced_event_v01(env, row)
        selected: dict[str, Any] = {}

        def policy_selector() -> str:
            semantic, valid_mask, q_input = _semantic_and_q_input_v01(
                learner,
                current_state=state,
                current_mask=state_mask,
                embedding=artifact.embeddings[step_index],
            )
            action, q_values, epsilon, source = learner.select_action(
                q_input=q_input,
                action_mask=action_mask,
                position_before=position_before,
            )
            selected.update(
                semantic=semantic,
                valid_mask=valid_mask,
                action=action,
                q_values=q_values,
                epsilon=epsilon,
                source=source,
            )
            return action

        if risk_event:
            semantic, valid_mask = _semantic_state_and_valid_mask_v01(
                learner,
                current_state=state,
                current_mask=state_mask,
                embedding=artifact.embeddings[step_index],
            )
            selected.update(
                semantic=semantic,
                valid_mask=valid_mask,
                action=risk_management_action_v02(position_before),
                epsilon=None,
                source="risk_management_forced_liquidation",
            )
            decision = RepresentationActionDecisionV01(
                variant=learner.variant,
                action=str(selected["action"]),
                source="risk_management_forced_liquidation",
                policy_eligible=False,
                readiness_schema_id=readiness.schema_id,
                readiness_reasons=(),
            )
        else:
            decision = select_representation_action_v01(
                learner.variant, readiness, policy_selector
            )
        runtime_timing = getattr(learner, "runtime_timing_v01", None)
        env_step_started = (
            time.perf_counter() if isinstance(runtime_timing, dict) else None
        )
        next_state_raw, reward, done, info = env.step(decision.action)
        info.update(
            {
                "entry_cutoff_active": bool(
                    mask_contract["entry_cutoff_active"]
                ),
                "entry_cutoff_gate_id": mask_contract[
                    "entry_cutoff_gate_id"
                ],
                "policy_effective_mask_reason": mask_contract[
                    "policy_effective_mask_reason"
                ],
                "entry_cutoff_blocked_action_slot_count": int(
                    mask_contract["blocked_action_slot_count"]
                ),
                "env_valid_action_slot_count": int(
                    mask_contract["env_valid_action_slot_count"]
                ),
                "env_action_mask_q_order": env_action_mask.tolist(),
                "policy_effective_action_mask_q_order": action_mask.tolist(),
                "target_action_mask_source": getattr(
                    learner, "target_action_mask_source", "env"
                ),
                "agent_decision_generated": not risk_event,
                "risk_management_transition": risk_event,
            }
        )
        if env_step_started is not None:
            runtime_timing["environment_step_seconds"] = (
                runtime_timing.get("environment_step_seconds", 0.0)
                + time.perf_counter()
                - env_step_started
            )
            runtime_timing["environment_step_count"] = int(
                runtime_timing.get("environment_step_count", 0)
            ) + 1
        reward_valid = bool(info.get("reward_valid", False))
        normalized_reward = (
            float(reward)
            if reward_valid and reward is not None and math.isfinite(float(reward))
            else None
        )
        raw_reward = info.get("reward_raw_yen") if reward_valid else None
        accounting.observe_step(
            decision=decision,
            timestamp=pd.Timestamp(row["grid_timestamp"]).isoformat(),
            session=str(row["session"]),
            position_before=str(info["position_before"]),
            position_after=str(info["position_after"]),
            normalized_reward=normalized_reward,
            raw_yen_reward=raw_reward,
            info=info,
        )
        if decision.policy_eligible or risk_event:
            if decision.policy_eligible:
                q_sample_count += 1
                q_sum += np.asarray(selected["q_values"], dtype=np.float64)
            if done:
                next_semantic = np.zeros(
                    learner.contract.state_dim, dtype=np.float32
                )
                next_valid_mask = np.zeros(
                    learner.contract.valid_mask_dim, dtype=bool
                )
                next_action_mask = np.zeros(3, dtype=bool)
                next_env_action_mask = np.zeros(3, dtype=bool)
                excluded = False
                next_readiness = None
            else:
                next_readiness = _readiness_for_row_v01(
                    artifact, step_index + 1, symbol, date
                )
                excluded = exclude_policy_transition_for_next_readiness_v01(
                    current_readiness=readiness,
                    done=False,
                    next_readiness=next_readiness,
                )
            event_reward = (
                float(reward)
                if reward is not None and math.isfinite(float(reward))
                else 0.0
            )
            if excluded:
                exclusion = learner.admission.observe_next_representation_unready_nonterminal(
                    ExcludedStepValidityV01(
                        episode_id=episode_id,
                        step_index=step_index,
                        reward=event_reward,
                        reward_valid=reward_valid,
                        training_valid=bool(info.get("training_valid", True)),
                        review_required=bool(info.get("review_required", False)),
                        done=False,
                        valuation_failed=bool(info.get("valuation_failed", False)),
                        forced_exit_failed=bool(info.get("forced_exit_failed", False)),
                        failure_reason=info.get("failure_reason"),
                    ),
                    next_readiness_reasons=next_readiness.reasons,
                )
                exclusions[exclusion["exclusion_reason"]] += 1
                next_reason_counts.update(next_readiness.reasons)
            else:
                if not done:
                    assert next_state_raw is not None and next_readiness is not None
                    _require(next_readiness.ready, "admitted nonterminal next representation unready")
                    next_mask59 = np.asarray(info["valid_state_mask"], dtype=bool)
                    next_semantic, next_valid_mask, _ = _semantic_and_q_input_v01(
                        learner,
                        current_state=np.asarray(next_state_raw, dtype=np.float32),
                        current_mask=next_mask59,
                        embedding=artifact.embeddings[step_index + 1],
                    )
                    next_mask_contract = _runtime_action_masks_v01(
                        env, artifact.episode_df.iloc[step_index + 1]
                    )
                    next_env_action_mask = np.asarray(
                        next_mask_contract["env_action_mask"], dtype=bool
                    )
                    next_action_mask = np.asarray(
                        next_mask_contract["policy_effective_action_mask"],
                        dtype=bool,
                    )
                transition = _build_representation_transition_v01(
                    learner=learner,
                    episode_id=episode_id,
                    step_index=step_index,
                    semantic=np.asarray(selected["semantic"]),
                    valid_mask=np.asarray(selected["valid_mask"], dtype=bool),
                    action=decision.action,
                    reward=event_reward,
                    next_semantic=next_semantic,
                    next_valid_mask=next_valid_mask,
                    done=bool(done),
                    action_mask=action_mask,
                    next_action_mask=next_action_mask,
                    env_action_mask=env_action_mask,
                    next_env_action_mask=next_env_action_mask,
                    metadata={
                        "date": date,
                        "symbol": symbol,
                        "policy_source": selected["source"],
                        "epsilon": selected["epsilon"],
                        "agent_decision": not risk_event,
                        "risk_management_transition": risk_event,
                        "entry_cutoff_active": bool(
                            mask_contract["entry_cutoff_active"]
                        ),
                        "policy_effective_mask_reason": mask_contract[
                            "policy_effective_mask_reason"
                        ],
                        "target_action_mask_source": getattr(
                            learner, "target_action_mask_source", "env"
                        ),
                    },
                )
                learner.admission.stage_policy_transition(
                    transition,
                    reward_valid=reward_valid,
                    training_valid=bool(info.get("training_valid", True)),
                    review_required=bool(info.get("review_required", False)),
                    valuation_failed=bool(info.get("valuation_failed", False)),
                    forced_exit_failed=bool(info.get("forced_exit_failed", False)),
                    failure_reason=info.get("failure_reason"),
                    policy_eligible=not risk_event,
                )
        else:
            learner.admission.observe_warmup_step(
                ExcludedStepValidityV01(
                    episode_id=episode_id,
                    step_index=step_index,
                    reward=(
                        float(reward)
                        if reward is not None and math.isfinite(float(reward))
                        else 0.0
                    ),
                    reward_valid=reward_valid,
                    training_valid=bool(info.get("training_valid", True)),
                    review_required=bool(info.get("review_required", False)),
                    done=bool(done),
                    valuation_failed=bool(info.get("valuation_failed", False)),
                    forced_exit_failed=bool(info.get("forced_exit_failed", False)),
                    failure_reason=info.get("failure_reason"),
                )
            )
        learner.global_env_step += 1
        step_index += 1
        previous_session = session
        if not done:
            if next_state_raw is None:
                raise RuntimeError("nonterminal Env next state missing")
            state = np.asarray(next_state_raw, dtype=np.float32)
            state_mask = np.asarray(info["valid_state_mask"], dtype=bool)
    summary = accounting.finalize()
    summary.update(
        {
            "run_phase": run_phase,
            "checkpoint_id": "training",
            "next_representation_exclusion_counts": dict(exclusions),
            "next_representation_unready_reason_counts": dict(
                next_reason_counts
            ),
            "next_representation_exclusion_rate": {
                "numerator": int(sum(exclusions.values())),
                "denominator": int(summary["policy_eligible_decision_count"]),
                "aggregation_unit": "policy_eligible_env_step",
                "value": (
                    sum(exclusions.values())
                    / summary["policy_eligible_decision_count"]
                    if summary["policy_eligible_decision_count"]
                    else None
                ),
            },
            "q_by_action_policy_mean": (
                (q_sum / q_sample_count).tolist() if q_sample_count else None
            ),
            "q_policy_sample_count": q_sample_count,
        }
    )
    if history_depth:
        summary["agent_history_diagnostics"] = {
            "state_count": int(
                history_available_counts[1] + history_unavailable_counts[1]
            ),
            "per_lag": {
                f"prev{lag}": {
                    "available_count": int(history_available_counts[lag]),
                    "unavailable_count": int(history_unavailable_counts[lag]),
                }
                for lag in range(1, history_depth + 1)
            },
            "morning_first_unavailable_count": int(
                history_first_unavailable["morning"]
            ),
            "afternoon_first_unavailable_count": int(
                history_first_unavailable["afternoon"]
            ),
            "history_invariant_violation_count": 0,
        }
    learner.recorder.record_episode(summary)
    return summary

def _build_representation_transition_v01(
    *,
    learner: _E0LearnerV01,
    episode_id: str,
    step_index: int,
    semantic: np.ndarray,
    valid_mask: np.ndarray,
    action: str,
    reward: float,
    next_semantic: np.ndarray,
    next_valid_mask: np.ndarray,
    done: bool,
    action_mask: np.ndarray,
    next_action_mask: np.ndarray,
    env_action_mask: np.ndarray | None = None,
    next_env_action_mask: np.ndarray | None = None,
    metadata: Mapping[str, Any],
):
    from training.rl_representation_replay_v01 import (
        RepresentationReplayTransitionV01,
    )

    return RepresentationReplayTransitionV01(
        variant=learner.variant,
        state_schema_id=learner.contract.state_schema_id,
        q_input_schema_id=learner.contract.q_input_schema_id,
        episode_id=episode_id,
        step_index=step_index,
        state=np.asarray(semantic, dtype=np.float32),
        state_valid_mask=np.asarray(valid_mask, dtype=bool),
        action_id=encode_action_v01(action),
        reward=float(reward),
        next_state=np.asarray(next_semantic, dtype=np.float32),
        next_state_valid_mask=np.asarray(next_valid_mask, dtype=bool),
        done=bool(done),
        action_mask=np.asarray(action_mask, dtype=bool),
        next_action_mask=np.asarray(next_action_mask, dtype=bool),
        env_action_mask=(
            None
            if env_action_mask is None
            else np.asarray(env_action_mask, dtype=bool)
        ),
        next_env_action_mask=(
            None
            if next_env_action_mask is None
            else np.asarray(next_env_action_mask, dtype=bool)
        ),
        policy_effective_action_mask=np.asarray(action_mask, dtype=bool),
        next_policy_effective_action_mask=np.asarray(
            next_action_mask, dtype=bool
        ),
        metadata=dict(metadata),
    )

def _save_checkpoint_v01(
    network: Any,
    run_root: Path,
    update: int,
    *,
    gamma: float,
    target_network: Any | None = None,
    backup_algorithm: str = "dqn",
    last_target_sync_update: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    actual_gamma = _validate_approved_gamma_v01(gamma)
    actual_backup_algorithm = validate_backup_algorithm_v01(
        backup_algorithm
    )
    path = Path(run_root) / "checkpoints" / f"update_{update:04d}"
    path.mkdir(parents=True, exist_ok=True)
    weights_path = path / "q_weights.npz"
    np.savez_compressed(
        weights_path,
        **{f"weight_{index}": value for index, value in enumerate(network.get_weights())},
    )
    target_weights_path = None
    if target_network is not None:
        target_weights_path = path / "target_q_weights.npz"
        np.savez_compressed(
            target_weights_path,
            **{
                f"weight_{index}": value
                for index, value in enumerate(target_network.get_weights())
            },
        )
    last_sync = (
        int(last_target_sync_update)
        if last_target_sync_update is not None
        else None
    )
    receipt = {
        "checkpoint_type": "fixed_diagnostic_observation_point",
        "optimizer_update": update,
        "gamma": actual_gamma,
        "backup_algorithm": actual_backup_algorithm,
        "weights_path": str(weights_path),
        "weights_sha256": file_sha256_v01(weights_path),
        "online_weights_path": str(weights_path),
        "online_weight_sha256": file_sha256_v01(weights_path),
        "online_network_identity_sha256": _network_weight_hash_v01(network),
        "target_weights_path": (
            str(target_weights_path) if target_weights_path is not None else None
        ),
        "target_weight_sha256": (
            file_sha256_v01(target_weights_path)
            if target_weights_path is not None
            else None
        ),
        "target_network_identity_sha256": (
            _network_weight_hash_v01(target_network)
            if target_network is not None
            else None
        ),
        "last_target_sync_update": last_sync,
        "sync_age_at_checkpoint": (
            int(update) - last_sync if last_sync is not None else None
        ),
        "best_checkpoint_selection": False,
    }
    if metadata:
        overlap = set(receipt) & set(metadata)
        _require(not overlap, f"checkpoint metadata overlap: {sorted(overlap)}")
        receipt.update(dict(metadata))
    _write_json(
        path / "receipt.json",
        receipt,
    )

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def _write_json(path: Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
