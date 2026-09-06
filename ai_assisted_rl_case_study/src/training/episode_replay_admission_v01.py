"""Common episode-staged Replay admission for supported training modes.

This module owns reward-validity admission, episode-level discard, atomic
episode commit, and compact diagnostics. It does not collect environments,
sample Replay, run Bellman updates, or write artifacts.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from training.replay_buffer_v01 import validate_finite_reward_scalar_v01


@dataclass(frozen=True)
class RewardAdmissionDecisionV01:
    reward_valid: bool
    training_valid: bool
    review_required: bool
    reward_is_finite: bool
    commit_transition: bool
    discard_episode: bool
    valuation_failed: bool
    forced_exit_failed: bool
    failure_reason: str | None


@dataclass
class ReplayAdmissionTransitionV01:
    episode_id: str
    step_index: int
    state: np.ndarray
    action_id: int
    reward: Any
    next_state: np.ndarray | None
    done: bool
    state_valid_mask: np.ndarray
    next_state_valid_mask: np.ndarray | None
    reward_valid: bool
    training_valid: bool
    review_required: bool
    failure_reason: str | None = None
    valuation_failed: bool = False
    forced_exit_failed: bool = False
    action_mask: np.ndarray | None = None
    next_action_mask: np.ndarray | None = None
    policy_effective_action_mask: np.ndarray | None = None
    next_policy_effective_action_mask: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplayAdmissionUpdateV01:
    episode_id: str
    step_index: int
    finalized: bool
    committed_transition_count: int
    skipped_transition_count: int
    discarded_episode: bool
    discarded_episode_transition_count: int


@dataclass
class EpisodeReplayAdmissionStatsV01:
    staged_transition_count: int = 0
    committed_transition_count: int = 0
    skipped_reward_invalid_count: int = 0
    skipped_nonfinite_reward_count: int = 0
    valuation_failed_transition_skip_count: int = 0
    training_invalid_episode_discard_count: int = 0
    discarded_episode_transition_count: int = 0
    committed_episode_count: int = 0
    pending_episode_count: int = 0
    replay_buffer_hard_reject_count: int = 0
    failure_reason_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class _StagedItemV01:
    transition: ReplayAdmissionTransitionV01
    decision: RewardAdmissionDecisionV01


@dataclass
class _EpisodeStageV01:
    episode_id: str
    items: list[_StagedItemV01] = field(default_factory=list)
    last_step_index: int | None = None
    training_valid: bool = True
    review_required: bool = False


def classify_reward_admission_v01(
    *,
    reward,
    reward_valid: bool,
    training_valid: bool,
    review_required: bool,
    valuation_failed: bool = False,
    forced_exit_failed: bool = False,
    failure_reason: str | None = None,
) -> RewardAdmissionDecisionV01:
    """Apply the canonical transition-skip versus episode-discard contract."""

    forced = bool(forced_exit_failed)
    valuation = bool(valuation_failed)
    effective_training_valid = bool(training_valid) and not forced
    effective_review_required = bool(review_required) or forced
    reward_is_finite = _is_finite_replay_reward_v01(reward)
    effective_reward_valid = (
        bool(reward_valid)
        and not forced
        and not valuation
        and reward_is_finite
    )
    commit_transition = (
        effective_training_valid
        and effective_reward_valid
        and reward_is_finite
    )
    return RewardAdmissionDecisionV01(
        reward_valid=effective_reward_valid,
        training_valid=effective_training_valid,
        review_required=effective_review_required,
        reward_is_finite=reward_is_finite,
        commit_transition=commit_transition,
        discard_episode=not effective_training_valid,
        valuation_failed=valuation,
        forced_exit_failed=forced,
        failure_reason=_normalize_failure_reason_v01(failure_reason),
    )


class EpisodeReplayAdmissionV01:
    """Stage transitions per episode and commit accepted episodes atomically."""

    def __init__(self, replay_buffer):
        if replay_buffer is None:
            raise ValueError("replay_buffer must not be None")
        self.replay_buffer = replay_buffer
        self._episodes: dict[str, _EpisodeStageV01] = {}
        self._closed_episode_ids: set[str] = set()
        self._stats = EpisodeReplayAdmissionStatsV01()
        self._failure_reason_counts: Counter[str] = Counter()

    def stage_transition(
        self,
        transition: ReplayAdmissionTransitionV01,
    ) -> ReplayAdmissionUpdateV01:
        if not isinstance(transition, ReplayAdmissionTransitionV01):
            raise TypeError("transition must be ReplayAdmissionTransitionV01")
        episode_id = str(transition.episode_id)
        if not episode_id:
            raise ValueError("episode_id must not be empty")
        if episode_id in self._closed_episode_ids:
            raise ValueError(f"transition after terminal for episode={episode_id}")
        step_index = _nonnegative_int_v01(transition.step_index, "step_index")
        done = _strict_bool_v01(transition.done, "done")

        stage = self._episodes.get(episode_id)
        if stage is None:
            stage = _EpisodeStageV01(episode_id=episode_id)
            self._episodes[episode_id] = stage
        if stage.last_step_index is not None and step_index <= stage.last_step_index:
            raise ValueError(
                "step_index must increase within episode: "
                f"{step_index} <= {stage.last_step_index}"
            )

        decision = classify_reward_admission_v01(
            reward=transition.reward,
            reward_valid=transition.reward_valid,
            training_valid=transition.training_valid,
            review_required=transition.review_required,
            valuation_failed=transition.valuation_failed,
            forced_exit_failed=transition.forced_exit_failed,
            failure_reason=transition.failure_reason,
        )
        stage.items.append(_StagedItemV01(transition=transition, decision=decision))
        stage.last_step_index = step_index
        stage.training_valid = stage.training_valid and decision.training_valid
        stage.review_required = stage.review_required or decision.review_required

        self._stats.staged_transition_count += 1
        if not decision.reward_valid:
            self._stats.skipped_reward_invalid_count += 1
        if not decision.reward_is_finite:
            self._stats.skipped_nonfinite_reward_count += 1
        if decision.valuation_failed and decision.training_valid:
            self._stats.valuation_failed_transition_skip_count += 1
        if decision.failure_reason:
            self._failure_reason_counts[decision.failure_reason] += 1

        if not done:
            self._refresh_stats_v01()
            return ReplayAdmissionUpdateV01(
                episode_id=episode_id,
                step_index=step_index,
                finalized=False,
                committed_transition_count=0,
                skipped_transition_count=int(not decision.commit_transition),
                discarded_episode=False,
                discarded_episode_transition_count=0,
            )
        return self._finalize_episode_v01(stage, terminal_step_index=step_index)

    def discard_pending_episode(
        self,
        episode_id: str,
        *,
        failure_reason: str,
    ) -> int:
        """Discard an incomplete episode after a runtime failure."""

        normalized_id = str(episode_id)
        stage = self._episodes.pop(normalized_id, None)
        if stage is None:
            return 0
        count = len(stage.items)
        self._closed_episode_ids.add(normalized_id)
        self._stats.training_invalid_episode_discard_count += 1
        self._stats.discarded_episode_transition_count += count
        reason = _normalize_failure_reason_v01(failure_reason)
        if reason:
            self._failure_reason_counts[reason] += 1
        self._refresh_stats_v01()
        return count

    def get_stats_snapshot(self) -> EpisodeReplayAdmissionStatsV01:
        self._refresh_stats_v01()
        return EpisodeReplayAdmissionStatsV01(
            staged_transition_count=self._stats.staged_transition_count,
            committed_transition_count=self._stats.committed_transition_count,
            skipped_reward_invalid_count=self._stats.skipped_reward_invalid_count,
            skipped_nonfinite_reward_count=(
                self._stats.skipped_nonfinite_reward_count
            ),
            valuation_failed_transition_skip_count=(
                self._stats.valuation_failed_transition_skip_count
            ),
            training_invalid_episode_discard_count=(
                self._stats.training_invalid_episode_discard_count
            ),
            discarded_episode_transition_count=(
                self._stats.discarded_episode_transition_count
            ),
            committed_episode_count=self._stats.committed_episode_count,
            pending_episode_count=len(self._episodes),
            replay_buffer_hard_reject_count=_replay_hard_reject_count_v01(
                self.replay_buffer
            ),
            failure_reason_counts=dict(sorted(self._failure_reason_counts.items())),
        )

    def _finalize_episode_v01(
        self,
        stage: _EpisodeStageV01,
        *,
        terminal_step_index: int,
    ) -> ReplayAdmissionUpdateV01:
        episode_id = stage.episode_id
        self._closed_episode_ids.add(episode_id)
        if not stage.training_valid:
            discarded_count = len(stage.items)
            self._episodes.pop(episode_id, None)
            self._stats.training_invalid_episode_discard_count += 1
            self._stats.discarded_episode_transition_count += discarded_count
            self._refresh_stats_v01()
            return ReplayAdmissionUpdateV01(
                episode_id=episode_id,
                step_index=terminal_step_index,
                finalized=True,
                committed_transition_count=0,
                skipped_transition_count=sum(
                    int(not item.decision.commit_transition) for item in stage.items
                ),
                discarded_episode=True,
                discarded_episode_transition_count=discarded_count,
            )

        eligible = [
            item.transition
            for item in stage.items
            if item.decision.commit_transition
        ]
        validate_episode_commit_transitions_v01(
            eligible,
            replay_buffer=self.replay_buffer,
        )
        add_episode_transitions_to_replay_v01(
            self.replay_buffer,
            eligible,
        )
        self._episodes.pop(episode_id, None)
        committed_count = len(eligible)
        self._stats.committed_transition_count += committed_count
        self._stats.committed_episode_count += 1
        self._refresh_stats_v01()
        return ReplayAdmissionUpdateV01(
            episode_id=episode_id,
            step_index=terminal_step_index,
            finalized=True,
            committed_transition_count=committed_count,
            skipped_transition_count=len(stage.items) - committed_count,
            discarded_episode=False,
            discarded_episode_transition_count=0,
        )

    def _refresh_stats_v01(self) -> None:
        self._stats.pending_episode_count = len(self._episodes)
        self._stats.replay_buffer_hard_reject_count = (
            _replay_hard_reject_count_v01(self.replay_buffer)
        )
        self._stats.failure_reason_counts = dict(
            sorted(self._failure_reason_counts.items())
        )


def validate_episode_commit_transitions_v01(
    transitions: list[ReplayAdmissionTransitionV01],
    *,
    replay_buffer,
) -> None:
    """Prevalidate every transition before any row of an episode is mutated."""

    if not transitions:
        return
    inferred_state_dim = (
        int(np.asarray(transitions[0].state).shape[0]) if transitions else 0
    )
    inferred_action_count = (
        int(np.asarray(transitions[0].action_mask).shape[0])
        if transitions and transitions[0].action_mask is not None
        else max(
            (int(transition.action_id) for transition in transitions),
            default=-1,
        )
        + 1
    )
    state_dim = int(
        getattr(replay_buffer, "state_dim", inferred_state_dim)
    )
    action_count = int(
        getattr(replay_buffer, "action_count", max(3, inferred_action_count))
    )
    if state_dim <= 0:
        raise ValueError("replay_buffer.state_dim must be positive")
    if action_count <= 0:
        raise ValueError("replay_buffer.action_count must be positive")
    for index, transition in enumerate(transitions):
        _validate_commit_transition_v01(
            transition,
            state_dim=state_dim,
            action_count=action_count,
            name=f"transitions[{index}]",
        )


def add_episode_transitions_to_replay_v01(
    replay_buffer,
    transitions: list[ReplayAdmissionTransitionV01],
) -> None:
    """Commit one prevalidated accepted episode in original step order."""

    if not transitions:
        return
    ordered = sorted(transitions, key=lambda item: int(item.step_index))
    validate_episode_commit_transitions_v01(
        ordered,
        replay_buffer=replay_buffer,
    )
    state_dim = int(
        getattr(
            replay_buffer,
            "state_dim",
            np.asarray(ordered[0].state).shape[0],
        )
    )
    states = np.asarray([item.state for item in ordered], dtype=np.float32)
    action_ids = np.asarray([item.action_id for item in ordered], dtype=np.int32)
    rewards = np.asarray(
        [
            validate_finite_reward_scalar_v01(
                item.reward,
                name=f"reward[{index}]",
            )
            for index, item in enumerate(ordered)
        ],
        dtype=np.float32,
    )
    dones = np.asarray([item.done for item in ordered], dtype=bool)
    state_masks = np.asarray(
        [item.state_valid_mask for item in ordered],
        dtype=bool,
    )
    next_states = np.zeros((len(ordered), state_dim), dtype=np.float32)
    next_masks = np.zeros((len(ordered), state_dim), dtype=bool)
    for index, item in enumerate(ordered):
        if item.next_state is not None:
            next_states[index] = np.asarray(item.next_state, dtype=np.float32)
        if item.next_state_valid_mask is not None:
            next_masks[index] = np.asarray(
                item.next_state_valid_mask,
                dtype=bool,
            )
    kwargs: dict[str, Any] = {
        "states": states,
        "action_ids": action_ids,
        "rewards": rewards,
        "next_states": next_states,
        "dones": dones,
        "state_valid_masks": state_masks,
        "next_state_valid_masks": next_masks,
    }
    _add_optional_mask_batches_v01(kwargs, ordered)
    if hasattr(replay_buffer, "add_many"):
        replay_buffer.add_many(**kwargs)
        return
    for item in ordered:
        replay_buffer.add(
            state=item.state,
            action_id=item.action_id,
            reward=item.reward,
            next_state=item.next_state,
            done=item.done,
            state_valid_mask=item.state_valid_mask,
            next_state_valid_mask=item.next_state_valid_mask,
            action_mask=item.action_mask,
            next_action_mask=item.next_action_mask,
            policy_effective_action_mask=item.policy_effective_action_mask,
            next_policy_effective_action_mask=(
                item.next_policy_effective_action_mask
            ),
        )


def replay_admission_stats_to_dict_v01(
    stats: EpisodeReplayAdmissionStatsV01,
) -> dict[str, Any]:
    return {
        "staged_transition_count": int(stats.staged_transition_count),
        "committed_transition_count": int(stats.committed_transition_count),
        "skipped_reward_invalid_count": int(
            stats.skipped_reward_invalid_count
        ),
        "skipped_nonfinite_reward_count": int(
            stats.skipped_nonfinite_reward_count
        ),
        "valuation_failed_transition_skip_count": int(
            stats.valuation_failed_transition_skip_count
        ),
        "training_invalid_episode_discard_count": int(
            stats.training_invalid_episode_discard_count
        ),
        "discarded_episode_transition_count": int(
            stats.discarded_episode_transition_count
        ),
        "committed_episode_count": int(stats.committed_episode_count),
        "pending_episode_count": int(stats.pending_episode_count),
        "replay_buffer_hard_reject_count": int(
            stats.replay_buffer_hard_reject_count
        ),
        "failure_reason_counts": dict(stats.failure_reason_counts),
    }


def _validate_commit_transition_v01(
    transition: ReplayAdmissionTransitionV01,
    *,
    state_dim: int,
    action_count: int,
    name: str,
) -> None:
    decision = classify_reward_admission_v01(
        reward=transition.reward,
        reward_valid=transition.reward_valid,
        training_valid=transition.training_valid,
        review_required=transition.review_required,
        valuation_failed=transition.valuation_failed,
        forced_exit_failed=transition.forced_exit_failed,
        failure_reason=transition.failure_reason,
    )
    if not decision.commit_transition:
        raise ValueError(f"{name} is not Replay-commit eligible")
    validate_finite_reward_scalar_v01(transition.reward, name=f"{name}.reward")
    _state_vector_v01(transition.state, state_dim, f"{name}.state")
    _mask_vector_v01(
        transition.state_valid_mask,
        state_dim,
        f"{name}.state_valid_mask",
    )
    action_id = _nonnegative_int_v01(transition.action_id, f"{name}.action_id")
    if action_id >= action_count:
        raise ValueError(f"{name}.action_id is out of range")
    done = _strict_bool_v01(transition.done, f"{name}.done")
    if done:
        if (
            transition.next_state is not None
            or transition.next_state_valid_mask is not None
        ):
            raise ValueError(f"{name} terminal next state and mask must be None")
    else:
        _state_vector_v01(
            transition.next_state,
            state_dim,
            f"{name}.next_state",
        )
        _mask_vector_v01(
            transition.next_state_valid_mask,
            state_dim,
            f"{name}.next_state_valid_mask",
        )
    for field_name in (
        "action_mask",
        "policy_effective_action_mask",
    ):
        value = getattr(transition, field_name)
        if value is not None:
            mask = _mask_vector_v01(
                value,
                action_count,
                f"{name}.{field_name}",
            )
            if not np.any(mask):
                raise ValueError(f"{name}.{field_name} must contain a valid action")
    for field_name in (
        "next_action_mask",
        "next_policy_effective_action_mask",
    ):
        value = getattr(transition, field_name)
        if done:
            if value is not None and np.any(
                _mask_vector_v01(value, action_count, f"{name}.{field_name}")
            ):
                raise ValueError(f"{name}.{field_name} must be all false at terminal")
        elif value is not None:
            mask = _mask_vector_v01(
                value,
                action_count,
                f"{name}.{field_name}",
            )
            if not np.any(mask):
                raise ValueError(f"{name}.{field_name} must contain a valid action")


def _add_optional_mask_batches_v01(
    kwargs: dict[str, Any],
    transitions: list[ReplayAdmissionTransitionV01],
) -> None:
    field_pairs = (
        ("action_masks", "action_mask"),
        ("next_action_masks", "next_action_mask"),
        ("policy_effective_action_masks", "policy_effective_action_mask"),
        (
            "next_policy_effective_action_masks",
            "next_policy_effective_action_mask",
        ),
    )
    action_count = int(
        getattr(
            transitions[0],
            "action_mask",
            np.zeros(0, dtype=bool),
        ).shape[0]
    ) if transitions[0].action_mask is not None else None
    for batch_name, field_name in field_pairs:
        values = [getattr(item, field_name) for item in transitions]
        if all(value is None for value in values):
            continue
        if any(value is None for value in values):
            if field_name.startswith("next_"):
                if action_count is None:
                    continue
                values = [
                    np.zeros(action_count, dtype=bool)
                    if value is None
                    else value
                    for value in values
                ]
            else:
                raise ValueError(f"{field_name} must be present for all transitions")
        kwargs[batch_name] = np.asarray(values, dtype=bool)


def _state_vector_v01(value, size: int, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} must not be None")
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    return array


def _mask_vector_v01(value, size: int, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} must not be None")
    array = np.asarray(value, dtype=bool)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    return array


def _is_finite_replay_reward_v01(value) -> bool:
    try:
        validate_finite_reward_scalar_v01(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _strict_bool_v01(value, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be bool")
    return bool(value)


def _nonnegative_int_v01(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if normalized < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return normalized


def _normalize_failure_reason_v01(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _replay_hard_reject_count_v01(replay_buffer) -> int:
    return int(getattr(replay_buffer, "reward_hard_reject_count", 0))
