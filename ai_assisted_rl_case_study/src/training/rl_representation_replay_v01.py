"""Strict versioned Replay and episode admission for B0/R1/A1 integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from training.episode_replay_admission_v01 import classify_reward_admission_v01
from training.replay_buffer_v01 import ACTION_COUNT_V01, validate_finite_reward_scalar_v01
from training.rl_representation_integration_v01 import (
    RepresentationVariantV01,
    contract_for_variant_v01,
)


REPLAY_SCHEMA_ID = "rl_representation_replay_v01"
NO_READY_STATUS_ID = "rl_representation_no_policy_eligible_decision_v01"


@dataclass(frozen=True)
class RepresentationReplayTransitionV01:
    variant: str
    state_schema_id: str
    q_input_schema_id: str
    episode_id: str
    step_index: int
    state: np.ndarray
    state_valid_mask: np.ndarray
    action_id: int
    reward: float
    next_state: np.ndarray
    next_state_valid_mask: np.ndarray
    done: bool
    action_mask: np.ndarray
    next_action_mask: np.ndarray
    env_action_mask: np.ndarray | None = None
    next_env_action_mask: np.ndarray | None = None
    policy_effective_action_mask: np.ndarray | None = None
    next_policy_effective_action_mask: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_id: str = REPLAY_SCHEMA_ID


@dataclass(frozen=True)
class RepresentationReplayBatchV01:
    states: np.ndarray
    state_valid_masks: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    next_state_valid_masks: np.ndarray
    dones: np.ndarray
    action_masks: np.ndarray
    next_action_masks: np.ndarray
    env_action_masks: np.ndarray
    next_env_action_masks: np.ndarray
    policy_effective_action_masks: np.ndarray
    next_policy_effective_action_masks: np.ndarray
    indices: np.ndarray


class RepresentationReplayBufferV01:
    """Fixed-size CPU buffer that permits state_dim != valid_mask_dim."""

    def __init__(self, *, capacity: int, variant: RepresentationVariantV01 | str, seed: int = 0):
        if isinstance(capacity, bool) or int(capacity) <= 0:
            raise ValueError("capacity must be a positive integer")
        self.contract = contract_for_variant_v01(variant)
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(int(seed))
        self.states = np.zeros((capacity, self.contract.state_dim), dtype=np.float32)
        self.state_valid_masks = np.zeros(
            (capacity, self.contract.valid_mask_dim), dtype=bool
        )
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, self.contract.state_dim), dtype=np.float32)
        self.next_state_valid_masks = np.zeros(
            (capacity, self.contract.valid_mask_dim), dtype=bool
        )
        self.dones = np.zeros(capacity, dtype=bool)
        self.action_masks = np.zeros((capacity, ACTION_COUNT_V01), dtype=bool)
        self.next_action_masks = np.zeros((capacity, ACTION_COUNT_V01), dtype=bool)
        self.env_action_masks = np.zeros((capacity, ACTION_COUNT_V01), dtype=bool)
        self.next_env_action_masks = np.zeros((capacity, ACTION_COUNT_V01), dtype=bool)
        self.policy_effective_action_masks = np.zeros(
            (capacity, ACTION_COUNT_V01), dtype=bool
        )
        self.next_policy_effective_action_masks = np.zeros(
            (capacity, ACTION_COUNT_V01), dtype=bool
        )
        self.size = 0
        self.write_index = 0
        self.total_added = 0

    def __len__(self) -> int:
        return self.size

    def add(self, transition: RepresentationReplayTransitionV01) -> None:
        values = validate_transition_v01(transition, expected_variant=self.contract.variant)
        index = self.write_index
        self.states[index] = values.state
        self.state_valid_masks[index] = values.state_valid_mask
        self.actions[index] = values.action_id
        self.rewards[index] = validate_finite_reward_scalar_v01(values.reward)
        self.next_states[index] = values.next_state
        self.next_state_valid_masks[index] = values.next_state_valid_mask
        self.dones[index] = values.done
        self.action_masks[index] = values.action_mask
        self.next_action_masks[index] = values.next_action_mask
        self.env_action_masks[index] = _resolved_mask(
            values.env_action_mask, values.action_mask
        )
        self.next_env_action_masks[index] = _resolved_mask(
            values.next_env_action_mask, values.next_action_mask
        )
        self.policy_effective_action_masks[index] = _resolved_mask(
            values.policy_effective_action_mask, values.action_mask
        )
        self.next_policy_effective_action_masks[index] = _resolved_mask(
            values.next_policy_effective_action_mask, values.next_action_mask
        )
        self.write_index = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_added += 1

    def add_many(self, transitions: list[RepresentationReplayTransitionV01]) -> None:
        validated = [
            validate_transition_v01(item, expected_variant=self.contract.variant)
            for item in transitions
        ]
        for item in validated:
            self.add(item)

    def sample(self, batch_size: int) -> RepresentationReplayBatchV01:
        if isinstance(batch_size, bool) or int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if int(batch_size) > self.size:
            raise ValueError("batch_size exceeds Replay size")
        indices = self.rng.choice(self.size, size=int(batch_size), replace=False)
        return RepresentationReplayBatchV01(
            states=self.states[indices].copy(),
            state_valid_masks=self.state_valid_masks[indices].copy(),
            actions=self.actions[indices].copy(),
            rewards=self.rewards[indices].copy(),
            next_states=self.next_states[indices].copy(),
            next_state_valid_masks=self.next_state_valid_masks[indices].copy(),
            dones=self.dones[indices].copy(),
            action_masks=self.action_masks[indices].copy(),
            next_action_masks=self.next_action_masks[indices].copy(),
            env_action_masks=self.env_action_masks[indices].copy(),
            next_env_action_masks=self.next_env_action_masks[indices].copy(),
            policy_effective_action_masks=self.policy_effective_action_masks[
                indices
            ].copy(),
            next_policy_effective_action_masks=(
                self.next_policy_effective_action_masks[indices].copy()
            ),
            indices=indices.astype(np.int64, copy=False),
        )


def terminal_next_v01(
    variant: RepresentationVariantV01 | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    contract = contract_for_variant_v01(variant)
    return (
        np.zeros(contract.state_dim, dtype=np.float32),
        np.zeros(contract.valid_mask_dim, dtype=bool),
        np.zeros(ACTION_COUNT_V01, dtype=bool),
    )


def validate_transition_v01(
    transition: RepresentationReplayTransitionV01,
    *,
    expected_variant: RepresentationVariantV01 | str,
) -> RepresentationReplayTransitionV01:
    if not isinstance(transition, RepresentationReplayTransitionV01):
        raise TypeError("transition must be RepresentationReplayTransitionV01")
    if transition.schema_id != REPLAY_SCHEMA_ID:
        raise ValueError("Replay transition schema mismatch")
    contract = contract_for_variant_v01(expected_variant)
    transition_contract = contract_for_variant_v01(transition.variant)
    if transition_contract.variant is not contract.variant:
        raise ValueError("Replay variant mismatch")
    if transition.state_schema_id != contract.state_schema_id:
        raise ValueError("Replay state schema mismatch")
    if transition.q_input_schema_id != contract.q_input_schema_id:
        raise ValueError("Replay Q-input schema mismatch")
    _array(transition.state, (contract.state_dim,), np.floating, "state")
    _array(transition.next_state, (contract.state_dim,), np.floating, "next_state")
    _bool_array(transition.state_valid_mask, (contract.valid_mask_dim,), "state_valid_mask")
    _bool_array(
        transition.next_state_valid_mask,
        (contract.valid_mask_dim,),
        "next_state_valid_mask",
    )
    _bool_array(transition.action_mask, (ACTION_COUNT_V01,), "action_mask")
    _bool_array(transition.next_action_mask, (ACTION_COUNT_V01,), "next_action_mask")
    for name in (
        "env_action_mask",
        "next_env_action_mask",
        "policy_effective_action_mask",
        "next_policy_effective_action_mask",
    ):
        value = getattr(transition, name)
        if value is not None:
            _bool_array(value, (ACTION_COUNT_V01,), name)
    if isinstance(transition.action_id, bool) or int(transition.action_id) not in range(ACTION_COUNT_V01):
        raise ValueError("Replay action_id is invalid")
    validate_finite_reward_scalar_v01(transition.reward)
    if type(transition.done) is not bool:
        raise ValueError("Replay done must be bool")
    if transition.done:
        if np.any(transition.next_state):
            raise ValueError("terminal next semantic state must be all-zero")
        if np.any(transition.next_state_valid_mask):
            raise ValueError("terminal next validity mask must be all-false")
        if np.any(transition.next_action_mask):
            raise ValueError("terminal next action mask must be all-false")
        for name in (
            "next_env_action_mask",
            "next_policy_effective_action_mask",
        ):
            value = getattr(transition, name)
            if value is not None and np.any(value):
                raise ValueError(f"terminal {name} must be all-false")
    return transition


def _resolved_mask(value: np.ndarray | None, fallback: np.ndarray) -> np.ndarray:
    return np.asarray(fallback if value is None else value, dtype=bool)


@dataclass(frozen=True)
class ExcludedStepValidityV01:
    episode_id: str
    step_index: int
    reward: float
    reward_valid: bool
    training_valid: bool
    review_required: bool
    done: bool
    valuation_failed: bool = False
    forced_exit_failed: bool = False
    failure_reason: str | None = None


@dataclass
class RepresentationAdmissionStatsV01:
    replay_staged_count: int = 0
    replay_committed_count: int = 0
    replay_skipped_count: int = 0
    replay_discarded_count: int = 0
    warmup_replay_excluded_count: int = 0
    training_invalid_episode_count: int = 0
    review_required_step_count: int = 0
    no_ready_episode_count: int = 0
    train_credit_added: int = 0
    policy_eligible_step_count: int = 0
    risk_management_transition_count: int = 0
    next_representation_unready_nonterminal_count: int = 0
    next_representation_unready_reason_counts: dict[str, int] = field(
        default_factory=dict
    )
    failure_reason_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class _EpisodeV01:
    transitions: list[tuple[RepresentationReplayTransitionV01, object]] = field(default_factory=list)
    last_step: int = -1
    training_valid: bool = True
    review_required: bool = False
    policy_eligible_count: int = 0


class RepresentationEpisodeAdmissionV01:
    """Propagate warm-up validity while staging only policy-eligible transitions."""

    def __init__(self, replay_buffer: RepresentationReplayBufferV01):
        self.replay_buffer = replay_buffer
        self.stats = RepresentationAdmissionStatsV01()
        self._episodes: dict[str, _EpisodeV01] = {}
        self._closed: set[str] = set()

    def observe_warmup_step(self, event: ExcludedStepValidityV01) -> dict[str, Any]:
        stage = self._stage(event.episode_id, event.step_index)
        decision = self._apply_validity(stage, event)
        self.stats.warmup_replay_excluded_count += 1
        if event.done:
            return self._finalize(event.episode_id)
        return {"finalized": False, "replay_committed": 0}

    def stage_policy_transition(
        self,
        transition: RepresentationReplayTransitionV01,
        *,
        reward_valid: bool,
        training_valid: bool,
        review_required: bool,
        valuation_failed: bool = False,
        forced_exit_failed: bool = False,
        failure_reason: str | None = None,
        policy_eligible: bool = True,
    ) -> dict[str, Any]:
        validate_transition_v01(
            transition,
            expected_variant=self.replay_buffer.contract.variant,
        )
        stage = self._stage(transition.episode_id, transition.step_index)
        event = ExcludedStepValidityV01(
            episode_id=transition.episode_id,
            step_index=transition.step_index,
            reward=transition.reward,
            reward_valid=reward_valid,
            training_valid=training_valid,
            review_required=review_required,
            done=transition.done,
            valuation_failed=valuation_failed,
            forced_exit_failed=forced_exit_failed,
            failure_reason=failure_reason,
        )
        decision = self._apply_validity(stage, event)
        stage.transitions.append((transition, decision))
        if policy_eligible:
            stage.policy_eligible_count += 1
            self.stats.policy_eligible_step_count += 1
        else:
            self.stats.risk_management_transition_count += 1
        self.stats.replay_staged_count += 1
        if transition.done:
            return self._finalize(transition.episode_id)
        return {"finalized": False, "replay_committed": 0}

    def observe_next_representation_unready_nonterminal(
        self,
        event: ExcludedStepValidityV01,
        *,
        next_readiness_reasons: tuple[str, ...],
    ) -> dict[str, Any]:
        """Propagate validity for one real policy step excluded from Replay.

        This is only the PM-approved current-ready, nonterminal, next-unready
        boundary.  It never constructs a next representation, adds Replay, or
        grants train credit.
        """

        if event.done:
            raise ValueError(
                "terminal policy transition must use normal Replay admission"
            )
        reasons = tuple(str(reason) for reason in next_readiness_reasons)
        if not reasons or any(not reason for reason in reasons):
            raise ValueError("next readiness reasons must be nonempty strings")
        stage = self._stage(event.episode_id, event.step_index)
        decision = self._apply_validity(stage, event)
        stage.policy_eligible_count += 1
        self.stats.policy_eligible_step_count += 1
        self.stats.next_representation_unready_nonterminal_count += 1
        for reason in reasons:
            counts = self.stats.next_representation_unready_reason_counts
            counts[reason] = counts.get(reason, 0) + 1
        return {
            "finalized": False,
            "replay_staged": 0,
            "replay_committed": 0,
            "train_credit_added": 0,
            "exclusion_reason": "next_representation_unready_nonterminal",
            "next_readiness_reasons": list(reasons),
            "training_valid": decision.training_valid,
            "review_required": decision.review_required,
            "exclusion_rate": {
                "numerator": (
                    self.stats.next_representation_unready_nonterminal_count
                ),
                "denominator": self.stats.policy_eligible_step_count,
                "aggregation_unit": "policy_eligible_env_step",
                "value": (
                    self.stats.next_representation_unready_nonterminal_count
                    / self.stats.policy_eligible_step_count
                ),
            },
        }

    def _stage(self, episode_id: str, step_index: int) -> _EpisodeV01:
        episode_id = str(episode_id)
        if not episode_id or episode_id in self._closed:
            raise ValueError("invalid or closed episode_id")
        stage = self._episodes.setdefault(episode_id, _EpisodeV01())
        if isinstance(step_index, bool) or int(step_index) <= stage.last_step:
            raise ValueError("step_index must strictly increase")
        stage.last_step = int(step_index)
        return stage

    def _apply_validity(self, stage: _EpisodeV01, event: ExcludedStepValidityV01):
        decision = classify_reward_admission_v01(
            reward=event.reward,
            reward_valid=event.reward_valid,
            training_valid=event.training_valid,
            review_required=event.review_required,
            valuation_failed=event.valuation_failed,
            forced_exit_failed=event.forced_exit_failed,
            failure_reason=event.failure_reason,
        )
        stage.training_valid = stage.training_valid and decision.training_valid
        stage.review_required = stage.review_required or decision.review_required
        self.stats.review_required_step_count += int(decision.review_required)
        if decision.failure_reason:
            counts = self.stats.failure_reason_counts
            counts[decision.failure_reason] = counts.get(decision.failure_reason, 0) + 1
        return decision

    def _finalize(self, episode_id: str) -> dict[str, Any]:
        stage = self._episodes.pop(episode_id)
        self._closed.add(episode_id)
        if not stage.training_valid:
            discarded = len(stage.transitions)
            self.stats.training_invalid_episode_count += 1
            self.stats.replay_discarded_count += discarded
            return {
                "finalized": True,
                "replay_committed": 0,
                "replay_discarded": discarded,
                "training_valid": False,
                "review_required": stage.review_required,
            }
        accepted = [item for item, decision in stage.transitions if decision.commit_transition]
        self.replay_buffer.add_many(accepted)
        committed = len(accepted)
        skipped = len(stage.transitions) - committed
        self.stats.replay_committed_count += committed
        self.stats.replay_skipped_count += skipped
        self.stats.train_credit_added += committed
        if stage.policy_eligible_count == 0:
            self.stats.no_ready_episode_count += 1
        return {
            "finalized": True,
            "replay_committed": committed,
            "replay_skipped": skipped,
            "training_valid": True,
            "review_required": stage.review_required,
            "policy_status": (
                NO_READY_STATUS_ID if stage.policy_eligible_count == 0 else "policy_eligible"
            ),
            "policy_status_reason": (
                "common_representation_readiness_never_true"
                if stage.policy_eligible_count == 0
                else None
            ),
        }


def _array(value: np.ndarray, shape: tuple[int, ...], kind, name: str) -> None:
    array = np.asarray(value)
    if array.shape != shape or not np.issubdtype(array.dtype, kind):
        raise ValueError(f"{name} must have shape {shape} and floating dtype")


def _bool_array(value: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.bool_:
        raise ValueError(f"{name} must be bool shape {shape}")


__all__ = [
    "ExcludedStepValidityV01",
    "NO_READY_STATUS_ID",
    "REPLAY_SCHEMA_ID",
    "RepresentationEpisodeAdmissionV01",
    "RepresentationReplayBufferV01",
    "RepresentationReplayTransitionV01",
    "terminal_next_v01",
    "validate_transition_v01",
]
