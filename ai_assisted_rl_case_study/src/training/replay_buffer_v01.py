"""NumPy replay buffer utilities v0.1.

This module keeps transitions in CPU-side NumPy arrays only. It does not import
TensorFlow, build tensors, collect rollouts, or implement prioritized replay.
"""

from __future__ import annotations

from dataclasses import dataclass
import numbers

import numpy as np

from features.state_vector_v01 import STATE_DIM_V01


ACTION_TO_ID_V01 = {
    "Hold": 0,
    "Buy": 1,
    "Sell": 2,
}
ID_TO_ACTION_V01 = {value: key for key, value in ACTION_TO_ID_V01.items()}
VALID_ACTION_IDS_V01 = frozenset(ID_TO_ACTION_V01)
ACTION_COUNT_V01 = len(ACTION_TO_ID_V01)
POSITION_SIDE_FLAT_INDEX_V01 = 50
POSITION_SIDE_LONG_INDEX_V01 = 51
POSITION_SIDE_SHORT_INDEX_V01 = 52

REPLAY_SAMPLING_MODE_UNIFORM = "uniform"
REPLAY_SAMPLING_MODE_TD_PER = "td_per"
REPLAY_SAMPLING_MODE_EVENT_BALANCED = "event_balanced"
REPLAY_SAMPLING_MODE_HYBRID_PER_EVENT = "hybrid_per_event"
REPLAY_SAMPLING_MODES_V01 = frozenset(
    {
        REPLAY_SAMPLING_MODE_UNIFORM,
        REPLAY_SAMPLING_MODE_TD_PER,
        REPLAY_SAMPLING_MODE_EVENT_BALANCED,
        REPLAY_SAMPLING_MODE_HYBRID_PER_EVENT,
    }
)

REPLAY_EVENT_FLAT_HOLD = 1 << 0
REPLAY_EVENT_NEW_LONG_ENTRY = 1 << 1
REPLAY_EVENT_NEW_SHORT_ENTRY = 1 << 2
REPLAY_EVENT_LONG_HOLD = 1 << 3
REPLAY_EVENT_LONG_CLOSE = 1 << 4
REPLAY_EVENT_SHORT_HOLD = 1 << 5
REPLAY_EVENT_SHORT_CLOSE = 1 << 6
REPLAY_EVENT_FORCED_EXIT = 1 << 7
REPLAY_EVENT_CLOSE_LABEL_CLOSE = 1 << 8
REPLAY_EVENT_ADVERSE_HOLD_CLOSE_LABEL = 1 << 9
REPLAY_EVENT_FORCED_OR_LATE_CLOSE_WINDOW = 1 << 10
REPLAY_EVENT_NO_EDGE_ENTRY = 1 << 11
REPLAY_EVENT_HIGH_CONFIDENCE_WRONG_ENTRY = 1 << 12
REPLAY_EVENT_GENERIC_ENTRY = 1 << 13
REPLAY_EVENT_RARE_FALLBACK = 1 << 14

REPLAY_EVENT_FLAG_NAMES_V01 = {
    REPLAY_EVENT_FLAT_HOLD: "FlatHold",
    REPLAY_EVENT_NEW_LONG_ENTRY: "NewLongEntry",
    REPLAY_EVENT_NEW_SHORT_ENTRY: "NewShortEntry",
    REPLAY_EVENT_LONG_HOLD: "LongHold",
    REPLAY_EVENT_LONG_CLOSE: "LongClose",
    REPLAY_EVENT_SHORT_HOLD: "ShortHold",
    REPLAY_EVENT_SHORT_CLOSE: "ShortClose",
    REPLAY_EVENT_FORCED_EXIT: "forced_exit",
    REPLAY_EVENT_CLOSE_LABEL_CLOSE: "close_label_Close",
    REPLAY_EVENT_ADVERSE_HOLD_CLOSE_LABEL: "q_hold_gt_close_and_close_label",
    REPLAY_EVENT_FORCED_OR_LATE_CLOSE_WINDOW: "forced_exit_or_late_close",
    REPLAY_EVENT_NO_EDGE_ENTRY: "no_edge_entry",
    REPLAY_EVENT_HIGH_CONFIDENCE_WRONG_ENTRY: "high_confidence_wrong_entry",
    REPLAY_EVENT_GENERIC_ENTRY: "entry_event",
    REPLAY_EVENT_RARE_FALLBACK: "rare_event",
}

DEFAULT_EVENT_BALANCED_BUCKET_MIX_V01 = {
    "uniform_background": 0.50,
    "close_event": 0.25,
    "adverse_hold": 0.10,
    "forced_exit_or_late_close": 0.05,
    "entry_event": 0.05,
    "rare_event": 0.05,
}


def encode_action_v01(action: str) -> int:
    """Encode a TradingEnv string action into the fixed v0.1 integer id."""

    try:
        return ACTION_TO_ID_V01[action]
    except KeyError as exc:
        raise ValueError(f"invalid action: {action!r}") from exc


def decode_action_v01(action_id: int) -> str:
    """Decode a fixed v0.1 integer action id into a TradingEnv string action."""

    normalized_id = _validate_action_id(action_id)
    return ID_TO_ACTION_V01[normalized_id]


def event_flag_names_from_mask_v01(mask: int) -> list[str]:
    """Return stable event flag names from a replay event bit mask."""

    value = int(mask)
    names = [
        name
        for flag, name in REPLAY_EVENT_FLAG_NAMES_V01.items()
        if bool(value & int(flag))
    ]
    return names or ["background"]


def validate_finite_reward_scalar_v01(
    value,
    *,
    name: str = "reward",
) -> np.float32:
    """Return the finite float32 replay representation of one numeric scalar."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite scalar numeric value")
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name} must be a finite scalar numeric value")
    scalar = array.item()
    if not isinstance(scalar, numbers.Real):
        raise ValueError(f"{name} must be a finite scalar numeric value")
    normalized = float(scalar)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        stored = np.float32(normalized)
    if not np.isfinite(stored):
        raise ValueError(f"{name} is outside finite float32 range")
    return stored


@dataclass
class ReplayBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    state_valid_masks: np.ndarray
    next_state_valid_masks: np.ndarray
    indices: np.ndarray
    action_masks: np.ndarray | None = None
    next_action_masks: np.ndarray | None = None
    policy_effective_action_masks: np.ndarray | None = None
    next_policy_effective_action_masks: np.ndarray | None = None
    position_before_ids: np.ndarray | None = None
    position_after_ids: np.ndarray | None = None
    selected_head_ids: np.ndarray | None = None
    local_action_ids: np.ndarray | None = None
    semantic_action_ids: np.ndarray | None = None
    legacy_action_ids: np.ndarray | None = None
    is_weights: np.ndarray | None = None
    sampling_probabilities: np.ndarray | None = None
    event_flags: np.ndarray | None = None
    priority_values: np.ndarray | None = None
    event_multipliers: np.ndarray | None = None
    symbol_ids: np.ndarray | None = None
    date_ids: np.ndarray | None = None
    symbol_date_ids: np.ndarray | None = None
    close_aux_labels: np.ndarray | None = None
    close_aux_label_valid_masks: np.ndarray | None = None


class ReplayBufferV01:
    """Fixed-size ring replay buffer backed by NumPy arrays."""

    def __init__(
        self,
        capacity: int,
        state_dim: int = STATE_DIM_V01,
        seed: int | None = None,
        action_count: int = ACTION_COUNT_V01,
        replay_sampling_mode: str = REPLAY_SAMPLING_MODE_UNIFORM,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        per_beta_steps: int = 100_000,
        priority_min: float = 1e-6,
        priority_max: float = 1_000_000.0,
        max_event_multiplier: float = 4.0,
        event_bucket_mix: dict[str, float] | None = None,
    ):
        self.capacity = _validate_positive_int(capacity, "capacity")
        self.state_dim = _validate_positive_int(state_dim, "state_dim")
        self.action_count = _validate_action_count(action_count)
        self.rng = np.random.default_rng(seed)
        self.replay_sampling_mode = _validate_sampling_mode(replay_sampling_mode)
        self.per_alpha = _validate_unit_float(per_alpha, "per_alpha", allow_zero=True)
        self.per_beta_start = _validate_unit_float(
            per_beta_start,
            "per_beta_start",
            allow_zero=True,
        )
        self.per_beta_end = _validate_unit_float(
            per_beta_end,
            "per_beta_end",
            allow_zero=True,
        )
        self.per_beta_steps = _validate_positive_int(per_beta_steps, "per_beta_steps")
        self.priority_min = _validate_positive_float(priority_min, "priority_min")
        self.priority_max = _validate_positive_float(priority_max, "priority_max")
        if self.priority_max < self.priority_min:
            raise ValueError("priority_max must be >= priority_min")
        self.max_event_multiplier = _validate_positive_float(
            max_event_multiplier,
            "max_event_multiplier",
        )
        self.event_bucket_mix = _validate_event_bucket_mix(event_bucket_mix)

        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros(
            (self.capacity, self.state_dim), dtype=np.float32
        )
        self.dones = np.zeros(self.capacity, dtype=bool)
        self.state_valid_masks = np.zeros(
            (self.capacity, self.state_dim), dtype=bool
        )
        self.next_state_valid_masks = np.zeros(
            (self.capacity, self.state_dim), dtype=bool
        )
        self.action_masks = np.zeros(
            (self.capacity, self.action_count), dtype=bool
        )
        self.next_action_masks = np.zeros(
            (self.capacity, self.action_count), dtype=bool
        )
        self.policy_effective_action_masks = np.zeros(
            (self.capacity, self.action_count), dtype=bool
        )
        self.next_policy_effective_action_masks = np.zeros(
            (self.capacity, self.action_count), dtype=bool
        )
        self.position_before_ids = np.full(self.capacity, -1, dtype=np.int32)
        self.position_after_ids = np.full(self.capacity, -1, dtype=np.int32)
        self.selected_head_ids = np.full(self.capacity, -1, dtype=np.int32)
        self.local_action_ids = np.full(self.capacity, -1, dtype=np.int32)
        self.semantic_action_ids = np.full(self.capacity, -1, dtype=np.int32)
        self.legacy_action_ids = np.full(self.capacity, -1, dtype=np.int32)
        self.priority_values = np.ones(self.capacity, dtype=np.float32)
        self.event_flags = np.zeros(self.capacity, dtype=np.int32)
        self.event_multipliers = np.ones(self.capacity, dtype=np.float32)
        self.symbol_ids = np.full(self.capacity, -1, dtype=np.int32)
        self.date_ids = np.full(self.capacity, -1, dtype=np.int32)
        self.symbol_date_ids = np.full(self.capacity, -1, dtype=np.int64)
        self.close_aux_labels = np.zeros(self.capacity, dtype=np.float32)
        self.close_aux_label_valid_masks = np.zeros(self.capacity, dtype=bool)

        self.size = 0
        self.write_index = 0
        self.is_full = False
        self.sample_call_count = 0
        self.priority_update_count = 0
        self.priority_nan_count = 0
        self.priority_inf_count = 0
        self.priority_clipping_count = 0
        self.is_weight_nan_count = 0
        self.is_weight_inf_count = 0
        self.event_bucket_empty_count = 0
        self.fallback_to_uniform_count = 0
        self.reward_hard_reject_count = 0
        self.sampled_event_counts_total: dict[str, int] = {}
        self.sampled_position_counts_total: dict[str, int] = {}
        self.sampled_bucket_counts_total: dict[str, int] = {}
        self.sampled_symbol_counts_total: dict[str, int] = {}
        self.sampled_date_counts_total: dict[str, int] = {}
        self.sampled_symbol_date_counts_total: dict[str, int] = {}
        self.sampled_bucket_symbol_counts_total: dict[str, dict[str, int]] = {}
        self.sampled_bucket_date_counts_total: dict[str, dict[str, int]] = {}
        self.sampled_bucket_symbol_date_counts_total: dict[str, dict[str, int]] = {}
        self.sampled_event_category_symbol_counts_total: dict[str, dict[str, int]] = {}
        self.sampled_event_category_date_counts_total: dict[str, dict[str, int]] = {}
        self.sampled_event_category_symbol_date_counts_total: dict[str, dict[str, int]] = {}

    def add(
        self,
        state: np.ndarray,
        action_id: int,
        reward: float,
        next_state: np.ndarray | None,
        done: bool,
        state_valid_mask: np.ndarray | None = None,
        next_state_valid_mask: np.ndarray | None = None,
        action_mask: np.ndarray | None = None,
        next_action_mask: np.ndarray | None = None,
        policy_effective_action_mask: np.ndarray | None = None,
        next_policy_effective_action_mask: np.ndarray | None = None,
        position_before_id: int | None = None,
        position_after_id: int | None = None,
        selected_head_id: int | None = None,
        local_action_id: int | None = None,
        semantic_action_id: int | None = None,
        legacy_action_id: int | None = None,
        event_flag: int | None = None,
        event_multiplier: float | None = None,
        symbol_id: int | None = None,
        date_id: int | None = None,
        symbol_date_id: int | None = None,
        close_aux_label: float | int | None = None,
        close_aux_label_valid_mask: bool | None = None,
    ) -> None:
        """Add one transition to the ring buffer."""

        normalized_reward = self._validate_reward_or_reject(reward, "reward")
        state_array = self._validate_state_vector(state, "state")
        normalized_action_id = _validate_action_id(action_id, action_count=self.action_count)
        normalized_done = bool(done)

        if next_state is None:
            if not normalized_done:
                raise ValueError("next_state must not be None when done is False")
            next_state_array = np.zeros(self.state_dim, dtype=np.float32)
            normalized_next_mask = np.zeros(self.state_dim, dtype=bool)
        else:
            next_state_array = self._validate_state_vector(next_state, "next_state")
            normalized_next_mask = self._validate_mask_or_default(
                next_state_valid_mask,
                next_state_array,
                "next_state_valid_mask",
            )

        normalized_state_mask = self._validate_mask_or_default(
            state_valid_mask,
            state_array,
            "state_valid_mask",
        )
        normalized_action_mask = self._validate_action_mask_or_default(
            action_mask,
            state_array,
            normalized_state_mask,
            "action_mask",
        )
        normalized_policy_effective_action_mask = self._validate_action_mask_or_default(
            policy_effective_action_mask,
            state_array,
            normalized_state_mask,
            "policy_effective_action_mask",
            default_mask=normalized_action_mask,
        )
        if normalized_done:
            normalized_next_action_mask = np.zeros(self.action_count, dtype=bool)
            normalized_next_policy_effective_action_mask = np.zeros(
                self.action_count,
                dtype=bool,
            )
        else:
            normalized_next_action_mask = self._validate_action_mask_or_default(
                next_action_mask,
                next_state_array,
                normalized_next_mask,
                "next_action_mask",
            )
            normalized_next_policy_effective_action_mask = (
                self._validate_action_mask_or_default(
                    next_policy_effective_action_mask,
                    next_state_array,
                    normalized_next_mask,
                    "next_policy_effective_action_mask",
                    default_mask=normalized_next_action_mask,
                )
            )

        index = self.write_index
        self.states[index] = state_array
        self.actions[index] = normalized_action_id
        self.rewards[index] = normalized_reward
        self.next_states[index] = next_state_array
        self.dones[index] = normalized_done
        self.state_valid_masks[index] = normalized_state_mask
        self.next_state_valid_masks[index] = normalized_next_mask
        self.action_masks[index] = normalized_action_mask
        self.next_action_masks[index] = normalized_next_action_mask
        self.policy_effective_action_masks[index] = (
            normalized_policy_effective_action_mask
        )
        self.next_policy_effective_action_masks[index] = (
            normalized_next_policy_effective_action_mask
        )
        self.position_before_ids[index] = _optional_metadata_id(position_before_id)
        self.position_after_ids[index] = _optional_metadata_id(position_after_id)
        self.selected_head_ids[index] = _optional_metadata_id(selected_head_id)
        self.local_action_ids[index] = _optional_metadata_id(local_action_id)
        self.semantic_action_ids[index] = _optional_metadata_id(semantic_action_id)
        self.legacy_action_ids[index] = _optional_metadata_id(legacy_action_id)
        self.priority_values[index] = self._initial_priority_value()
        self.event_flags[index] = _optional_event_flag(event_flag)
        self.event_multipliers[index] = _normalize_event_multiplier(
            event_multiplier,
            max_event_multiplier=self.max_event_multiplier,
        )
        normalized_symbol_id = _optional_metadata_id(symbol_id)
        normalized_date_id = _optional_metadata_id(date_id)
        self.symbol_ids[index] = normalized_symbol_id
        self.date_ids[index] = normalized_date_id
        self.symbol_date_ids[index] = _optional_symbol_date_id(
            symbol_date_id,
            symbol_id=normalized_symbol_id,
            date_id=normalized_date_id,
        )
        normalized_close_aux_label = _optional_close_aux_label(close_aux_label)
        self.close_aux_labels[index] = normalized_close_aux_label
        self.close_aux_label_valid_masks[index] = _optional_close_aux_valid_mask(
            close_aux_label_valid_mask,
            label=normalized_close_aux_label,
        )

        self.write_index = (self.write_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.is_full = self.size == self.capacity

    def add_many(
        self,
        *,
        states: np.ndarray,
        action_ids: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        state_valid_masks: np.ndarray | None = None,
        next_state_valid_masks: np.ndarray | None = None,
        action_masks: np.ndarray | None = None,
        next_action_masks: np.ndarray | None = None,
        policy_effective_action_masks: np.ndarray | None = None,
        next_policy_effective_action_masks: np.ndarray | None = None,
        position_before_ids: np.ndarray | None = None,
        position_after_ids: np.ndarray | None = None,
        selected_head_ids: np.ndarray | None = None,
        local_action_ids: np.ndarray | None = None,
        semantic_action_ids: np.ndarray | None = None,
        legacy_action_ids: np.ndarray | None = None,
        event_flags: np.ndarray | None = None,
        event_multipliers: np.ndarray | None = None,
        symbol_ids: np.ndarray | None = None,
        date_ids: np.ndarray | None = None,
        symbol_date_ids: np.ndarray | None = None,
        close_aux_labels: np.ndarray | None = None,
        close_aux_label_valid_masks: np.ndarray | None = None,
    ) -> None:
        """Add a batch of transitions to the ring buffer."""

        states_array = self._validate_state_matrix(states, "states")
        count = int(states_array.shape[0])
        if count <= 0:
            return
        action_array = np.asarray(action_ids, dtype=np.int32)
        reward_array = self._validate_reward_array_or_reject(rewards, count)
        next_states_array = self._validate_state_matrix(next_states, "next_states")
        dones_array = np.asarray(dones, dtype=bool)
        if action_array.shape != (count,):
            raise ValueError(f"action_ids must have shape ({count},)")
        if reward_array.shape != (count,):
            raise ValueError(f"rewards must have shape ({count},)")
        if dones_array.shape != (count,):
            raise ValueError(f"dones must have shape ({count},)")
        for action_id in action_array:
            _validate_action_id(int(action_id), action_count=self.action_count)
        state_masks = self._validate_mask_matrix_or_default(
            state_valid_masks,
            states_array,
            "state_valid_masks",
        )
        next_state_masks = self._validate_mask_matrix_or_default(
            next_state_valid_masks,
            next_states_array,
            "next_state_valid_masks",
        )
        normalized_action_masks = self._validate_action_mask_matrix_or_default(
            action_masks,
            states_array,
            state_masks,
            "action_masks",
        )
        normalized_next_action_masks = self._validate_action_mask_matrix_or_default(
            next_action_masks,
            next_states_array,
            next_state_masks,
            "next_action_masks",
        )
        normalized_policy_effective_action_masks = (
            self._validate_action_mask_matrix_or_default(
                policy_effective_action_masks,
                states_array,
                state_masks,
                "policy_effective_action_masks",
                default_masks=normalized_action_masks,
            )
        )
        normalized_next_policy_effective_action_masks = (
            self._validate_action_mask_matrix_or_default(
                next_policy_effective_action_masks,
                next_states_array,
                next_state_masks,
                "next_policy_effective_action_masks",
                default_masks=normalized_next_action_masks,
            )
        )
        normalized_position_before_ids = _metadata_id_array(position_before_ids, count)
        normalized_position_after_ids = _metadata_id_array(position_after_ids, count)
        normalized_selected_head_ids = _metadata_id_array(selected_head_ids, count)
        normalized_local_action_ids = _metadata_id_array(local_action_ids, count)
        normalized_semantic_action_ids = _metadata_id_array(semantic_action_ids, count)
        normalized_legacy_action_ids = _metadata_id_array(legacy_action_ids, count)
        normalized_event_flags = _event_flag_array(event_flags, count)
        normalized_event_multipliers = _event_multiplier_array(
            event_multipliers,
            count,
            max_event_multiplier=self.max_event_multiplier,
        )
        normalized_symbol_ids = _metadata_id_array(symbol_ids, count)
        normalized_date_ids = _metadata_id_array(date_ids, count)
        normalized_symbol_date_ids = _symbol_date_id_array(
            symbol_date_ids,
            count,
            symbol_ids=normalized_symbol_ids,
            date_ids=normalized_date_ids,
        )
        normalized_close_aux_labels = _close_aux_label_array(close_aux_labels, count)
        normalized_close_aux_label_valid_masks = _close_aux_valid_mask_array(
            close_aux_label_valid_masks,
            count,
            labels=normalized_close_aux_labels,
        )
        normalized_next_action_masks[dones_array] = False
        normalized_next_policy_effective_action_masks[dones_array] = False
        if count > self.capacity:
            states_array = states_array[-self.capacity :]
            action_array = action_array[-self.capacity :]
            reward_array = reward_array[-self.capacity :]
            next_states_array = next_states_array[-self.capacity :]
            dones_array = dones_array[-self.capacity :]
            state_masks = state_masks[-self.capacity :]
            next_state_masks = next_state_masks[-self.capacity :]
            normalized_action_masks = normalized_action_masks[-self.capacity :]
            normalized_next_action_masks = normalized_next_action_masks[-self.capacity :]
            normalized_policy_effective_action_masks = (
                normalized_policy_effective_action_masks[-self.capacity :]
            )
            normalized_next_policy_effective_action_masks = (
                normalized_next_policy_effective_action_masks[-self.capacity :]
            )
            normalized_position_before_ids = normalized_position_before_ids[-self.capacity :]
            normalized_position_after_ids = normalized_position_after_ids[-self.capacity :]
            normalized_selected_head_ids = normalized_selected_head_ids[-self.capacity :]
            normalized_local_action_ids = normalized_local_action_ids[-self.capacity :]
            normalized_semantic_action_ids = normalized_semantic_action_ids[-self.capacity :]
            normalized_legacy_action_ids = normalized_legacy_action_ids[-self.capacity :]
            normalized_event_flags = normalized_event_flags[-self.capacity :]
            normalized_event_multipliers = normalized_event_multipliers[-self.capacity :]
            normalized_symbol_ids = normalized_symbol_ids[-self.capacity :]
            normalized_date_ids = normalized_date_ids[-self.capacity :]
            normalized_symbol_date_ids = normalized_symbol_date_ids[-self.capacity :]
            normalized_close_aux_labels = normalized_close_aux_labels[-self.capacity :]
            normalized_close_aux_label_valid_masks = (
                normalized_close_aux_label_valid_masks[-self.capacity :]
            )
            count = self.capacity

        indices = (np.arange(count, dtype=np.int64) + int(self.write_index)) % int(
            self.capacity
        )
        initial_priority = self._initial_priority_value()
        self.states[indices] = states_array
        self.actions[indices] = action_array
        self.rewards[indices] = reward_array
        self.next_states[indices] = next_states_array
        self.dones[indices] = dones_array
        self.state_valid_masks[indices] = state_masks
        self.next_state_valid_masks[indices] = next_state_masks
        self.action_masks[indices] = normalized_action_masks
        self.next_action_masks[indices] = normalized_next_action_masks
        self.policy_effective_action_masks[indices] = (
            normalized_policy_effective_action_masks
        )
        self.next_policy_effective_action_masks[indices] = (
            normalized_next_policy_effective_action_masks
        )
        self.position_before_ids[indices] = normalized_position_before_ids
        self.position_after_ids[indices] = normalized_position_after_ids
        self.selected_head_ids[indices] = normalized_selected_head_ids
        self.local_action_ids[indices] = normalized_local_action_ids
        self.semantic_action_ids[indices] = normalized_semantic_action_ids
        self.legacy_action_ids[indices] = normalized_legacy_action_ids
        self.priority_values[indices] = initial_priority
        self.event_flags[indices] = normalized_event_flags
        self.event_multipliers[indices] = normalized_event_multipliers
        self.symbol_ids[indices] = normalized_symbol_ids
        self.date_ids[indices] = normalized_date_ids
        self.symbol_date_ids[indices] = normalized_symbol_date_ids
        self.close_aux_labels[indices] = normalized_close_aux_labels
        self.close_aux_label_valid_masks[indices] = (
            normalized_close_aux_label_valid_masks
        )

        self.write_index = (self.write_index + count) % self.capacity
        self.size = min(self.size + count, self.capacity)
        self.is_full = self.size == self.capacity

    def sample_batch(
        self,
        batch_size: int,
        replace: bool = False,
    ) -> ReplayBatch:
        """Sample currently stored transitions using the configured replay mode."""

        normalized_batch_size = _validate_positive_int(batch_size, "batch_size")
        if self.size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        if not replace and normalized_batch_size > self.size:
            raise ValueError("batch_size must be <= size when replace is False")

        indices, probabilities, bucket_names = self._sample_indices(
            normalized_batch_size,
            replace=replace,
        )
        self.sample_call_count += 1
        is_weights = self._importance_weights(indices, probabilities)
        self._record_sample_diagnostics(indices, bucket_names)

        return ReplayBatch(
            states=self.states[indices].copy(),
            actions=self.actions[indices].copy(),
            rewards=self.rewards[indices].copy(),
            next_states=self.next_states[indices].copy(),
            dones=self.dones[indices].copy(),
            state_valid_masks=self.state_valid_masks[indices].copy(),
            next_state_valid_masks=self.next_state_valid_masks[indices].copy(),
            indices=indices,
            action_masks=self.action_masks[indices].copy(),
            next_action_masks=self.next_action_masks[indices].copy(),
            policy_effective_action_masks=(
                self.policy_effective_action_masks[indices].copy()
            ),
            next_policy_effective_action_masks=(
                self.next_policy_effective_action_masks[indices].copy()
            ),
            position_before_ids=self.position_before_ids[indices].copy(),
            position_after_ids=self.position_after_ids[indices].copy(),
            selected_head_ids=self.selected_head_ids[indices].copy(),
            local_action_ids=self.local_action_ids[indices].copy(),
            semantic_action_ids=self.semantic_action_ids[indices].copy(),
            legacy_action_ids=self.legacy_action_ids[indices].copy(),
            is_weights=is_weights,
            sampling_probabilities=probabilities.astype(np.float32).copy(),
            event_flags=self.event_flags[indices].copy(),
            priority_values=self.priority_values[indices].copy(),
            event_multipliers=self.event_multipliers[indices].copy(),
            symbol_ids=self.symbol_ids[indices].copy(),
            date_ids=self.date_ids[indices].copy(),
            symbol_date_ids=self.symbol_date_ids[indices].copy(),
            close_aux_labels=self.close_aux_labels[indices].copy(),
            close_aux_label_valid_masks=(
                self.close_aux_label_valid_masks[indices].copy()
            ),
        )

    def update_priorities(self, indices, td_errors) -> dict[str, int | float | None]:
        """Update PER base priorities from per-row TD errors."""

        if indices is None or td_errors is None:
            return {
                "updated_count": 0,
                "priority_nan_count": int(self.priority_nan_count),
                "priority_inf_count": int(self.priority_inf_count),
                "priority_clipping_count": int(self.priority_clipping_count),
            }
        index_array = np.asarray(indices, dtype=np.int64).reshape(-1)
        td_array = np.asarray(td_errors, dtype=np.float64).reshape(-1)
        if index_array.shape != td_array.shape:
            raise ValueError("indices and td_errors must have the same shape")
        if index_array.size == 0:
            return {
                "updated_count": 0,
                "priority_nan_count": int(self.priority_nan_count),
                "priority_inf_count": int(self.priority_inf_count),
                "priority_clipping_count": int(self.priority_clipping_count),
            }
        invalid_index = [
            int(index)
            for index in index_array.tolist()
            if int(index) < 0 or int(index) >= int(self.size)
        ]
        if invalid_index:
            raise ValueError(f"priority update index out of active range: {invalid_index[:5]}")

        finite_mask = np.isfinite(td_array)
        self.priority_nan_count += int(np.isnan(td_array).sum())
        self.priority_inf_count += int(np.isinf(td_array).sum())
        safe_td = np.where(finite_mask, np.abs(td_array), float(self.priority_max))
        raw_priority = safe_td + float(self.priority_min)
        clipped = np.clip(raw_priority, float(self.priority_min), float(self.priority_max))
        self.priority_clipping_count += int(np.count_nonzero(raw_priority != clipped))
        self.priority_values[index_array] = clipped.astype(np.float32)
        self.priority_update_count += int(index_array.size)
        return {
            "updated_count": int(index_array.size),
            "priority_nan_count": int(self.priority_nan_count),
            "priority_inf_count": int(self.priority_inf_count),
            "priority_clipping_count": int(self.priority_clipping_count),
            "priority_min_active": _finite_stat_np(self.priority_values[: self.size], "min"),
            "priority_max_active": _finite_stat_np(self.priority_values[: self.size], "max"),
        }

    def _sample_indices(
        self,
        batch_size: int,
        *,
        replace: bool,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        mode = str(self.replay_sampling_mode)
        if mode == REPLAY_SAMPLING_MODE_UNIFORM:
            return self._sample_uniform(batch_size, replace=replace, bucket="uniform")
        if mode == REPLAY_SAMPLING_MODE_TD_PER:
            return self._sample_probability_weighted(
                batch_size,
                replace=replace,
                weights=self._td_per_sampling_weights(),
                bucket="td_per",
            )
        if mode == REPLAY_SAMPLING_MODE_HYBRID_PER_EVENT:
            weights = self._td_per_sampling_weights() * self.event_multipliers[: self.size]
            return self._sample_probability_weighted(
                batch_size,
                replace=replace,
                weights=weights,
                bucket="hybrid_per_event",
            )
        if mode == REPLAY_SAMPLING_MODE_EVENT_BALANCED:
            return self._sample_event_balanced(batch_size, replace=replace)
        raise ValueError(f"invalid replay_sampling_mode: {mode!r}")

    def _sample_uniform(
        self,
        batch_size: int,
        *,
        replace: bool,
        bucket: str,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        indices = self.rng.choice(self.size, size=batch_size, replace=replace).astype(np.int64)
        probabilities = np.full(batch_size, 1.0 / float(self.size), dtype=np.float32)
        return indices, probabilities, [bucket] * int(batch_size)

    def _sample_probability_weighted(
        self,
        batch_size: int,
        *,
        replace: bool,
        weights: np.ndarray,
        bucket: str,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        probability_by_index = _normalize_sampling_weights(weights)
        if probability_by_index is None:
            self.fallback_to_uniform_count += 1
            return self._sample_uniform(batch_size, replace=replace, bucket=f"{bucket}_fallback_uniform")
        indices = self.rng.choice(
            self.size,
            size=batch_size,
            replace=replace,
            p=probability_by_index,
        ).astype(np.int64)
        probabilities = probability_by_index[indices].astype(np.float32)
        return indices, probabilities, [bucket] * int(batch_size)

    def _sample_event_balanced(
        self,
        batch_size: int,
        *,
        replace: bool,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        chunks: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        bucket_names: list[str] = []
        remaining = int(batch_size)
        items = list(self.event_bucket_mix.items())
        for offset, (bucket_name, ratio) in enumerate(items):
            if remaining <= 0:
                break
            if offset == len(items) - 1:
                bucket_size = remaining
            else:
                bucket_size = int(round(float(batch_size) * float(ratio)))
                bucket_size = max(0, min(bucket_size, remaining))
            if bucket_size <= 0:
                continue
            candidates = self._event_bucket_indices(bucket_name)
            if candidates.size == 0:
                self.event_bucket_empty_count += 1
                self.fallback_to_uniform_count += 1
                sampled = self.rng.choice(self.size, size=bucket_size, replace=True).astype(np.int64)
                prob = np.full(bucket_size, 1.0 / float(self.size), dtype=np.float32)
                name = f"{bucket_name}_fallback_uniform"
            else:
                effective_replace = bool(replace or bucket_size > candidates.size)
                sampled = self.rng.choice(candidates, size=bucket_size, replace=effective_replace).astype(np.int64)
                prob = np.full(bucket_size, 1.0 / float(candidates.size), dtype=np.float32)
                name = bucket_name
            chunks.append(sampled)
            probabilities.append(prob)
            bucket_names.extend([name] * int(bucket_size))
            remaining -= int(bucket_size)
        if remaining > 0:
            sampled, prob, names = self._sample_uniform(
                remaining,
                replace=True,
                bucket="event_mix_remainder_uniform",
            )
            chunks.append(sampled)
            probabilities.append(prob)
            bucket_names.extend(names)
        if not chunks:
            return self._sample_uniform(batch_size, replace=replace, bucket="event_mix_empty_uniform")
        indices = np.concatenate(chunks).astype(np.int64)
        probs = np.concatenate(probabilities).astype(np.float32)
        order = self.rng.permutation(indices.shape[0])
        return indices[order], probs[order], [bucket_names[int(i)] for i in order.tolist()]

    def _td_per_sampling_weights(self) -> np.ndarray:
        active_priority = np.asarray(self.priority_values[: self.size], dtype=np.float64)
        safe_priority = np.clip(
            np.where(np.isfinite(active_priority), active_priority, float(self.priority_max)),
            float(self.priority_min),
            float(self.priority_max),
        )
        return np.power(safe_priority, float(self.per_alpha))

    def _event_bucket_indices(self, bucket_name: str) -> np.ndarray:
        flags = np.asarray(self.event_flags[: self.size], dtype=np.int32)
        if bucket_name == "uniform_background":
            return np.arange(self.size, dtype=np.int64)
        if bucket_name == "close_event":
            mask = (
                _has_flag(flags, REPLAY_EVENT_CLOSE_LABEL_CLOSE)
                | _has_flag(flags, REPLAY_EVENT_LONG_CLOSE)
                | _has_flag(flags, REPLAY_EVENT_SHORT_CLOSE)
            )
        elif bucket_name == "adverse_hold":
            mask = _has_flag(flags, REPLAY_EVENT_ADVERSE_HOLD_CLOSE_LABEL)
        elif bucket_name == "forced_exit_or_late_close":
            mask = (
                _has_flag(flags, REPLAY_EVENT_FORCED_OR_LATE_CLOSE_WINDOW)
                | _has_flag(flags, REPLAY_EVENT_FORCED_EXIT)
            )
        elif bucket_name == "entry_event":
            mask = (
                _has_flag(flags, REPLAY_EVENT_GENERIC_ENTRY)
                | _has_flag(flags, REPLAY_EVENT_NO_EDGE_ENTRY)
                | _has_flag(flags, REPLAY_EVENT_HIGH_CONFIDENCE_WRONG_ENTRY)
            )
        elif bucket_name == "rare_event":
            mask = flags != 0
        else:
            mask = np.zeros(self.size, dtype=bool)
        return np.nonzero(mask)[0].astype(np.int64)

    def _importance_weights(
        self,
        indices: np.ndarray,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        if str(self.replay_sampling_mode) == REPLAY_SAMPLING_MODE_UNIFORM:
            return np.ones(indices.shape[0], dtype=np.float32)
        beta = self._current_beta()
        probs = np.asarray(probabilities, dtype=np.float64)
        weights = np.power(float(self.size) * probs, -float(beta))
        finite_mask = np.isfinite(weights)
        self.is_weight_nan_count += int(np.isnan(weights).sum())
        self.is_weight_inf_count += int(np.isinf(weights).sum())
        if not np.all(finite_mask):
            weights = np.where(finite_mask, weights, 1.0)
        max_weight = float(np.max(weights)) if weights.size else 1.0
        if max_weight > 0.0 and np.isfinite(max_weight):
            weights = weights / max_weight
        return weights.astype(np.float32)

    def _current_beta(self) -> float:
        progress = min(1.0, float(self.sample_call_count) / float(self.per_beta_steps))
        return float(self.per_beta_start) + (
            float(self.per_beta_end) - float(self.per_beta_start)
        ) * progress

    def _initial_priority_value(self) -> np.float32:
        active = self.priority_values[: self.size]
        finite = active[np.isfinite(active)] if active.size else np.asarray([], dtype=np.float32)
        if finite.size:
            value = float(np.max(finite))
        else:
            value = 1.0
        value = float(np.clip(value, float(self.priority_min), float(self.priority_max)))
        return np.float32(value)

    def _record_sample_diagnostics(
        self,
        indices: np.ndarray,
        bucket_names: list[str],
    ) -> None:
        for flag in np.asarray(self.event_flags[indices], dtype=np.int32).tolist():
            for name in event_flag_names_from_mask_v01(int(flag)):
                self.sampled_event_counts_total[name] = (
                    int(self.sampled_event_counts_total.get(name, 0)) + 1
                )
        for position_id in np.asarray(self.position_before_ids[indices], dtype=np.int32).tolist():
            key = str(int(position_id))
            self.sampled_position_counts_total[key] = (
                int(self.sampled_position_counts_total.get(key, 0)) + 1
            )
        for bucket in bucket_names:
            self.sampled_bucket_counts_total[str(bucket)] = (
                int(self.sampled_bucket_counts_total.get(str(bucket), 0)) + 1
            )
        sampled_symbols = np.asarray(self.symbol_ids[indices], dtype=np.int32)
        sampled_dates = np.asarray(self.date_ids[indices], dtype=np.int32)
        sampled_symbol_dates = np.asarray(self.symbol_date_ids[indices], dtype=np.int64)
        _record_group_counts(sampled_symbols, self.sampled_symbol_counts_total)
        _record_group_counts(sampled_dates, self.sampled_date_counts_total)
        _record_group_counts(sampled_symbol_dates, self.sampled_symbol_date_counts_total)
        for offset, bucket in enumerate(bucket_names):
            _record_nested_group_count(
                self.sampled_bucket_symbol_counts_total,
                str(bucket),
                int(sampled_symbols[offset]),
            )
            _record_nested_group_count(
                self.sampled_bucket_date_counts_total,
                str(bucket),
                int(sampled_dates[offset]),
            )
            _record_nested_group_count(
                self.sampled_bucket_symbol_date_counts_total,
                str(bucket),
                int(sampled_symbol_dates[offset]),
            )
        for flag, symbol_id, date_id, symbol_date_id in zip(
            np.asarray(self.event_flags[indices], dtype=np.int32).tolist(),
            sampled_symbols.tolist(),
            sampled_dates.tolist(),
            sampled_symbol_dates.tolist(),
        ):
            for category in _event_categories_from_mask(int(flag)):
                _record_nested_group_count(
                    self.sampled_event_category_symbol_counts_total,
                    category,
                    int(symbol_id),
                )
                _record_nested_group_count(
                    self.sampled_event_category_date_counts_total,
                    category,
                    int(date_id),
                )
                _record_nested_group_count(
                    self.sampled_event_category_symbol_date_counts_total,
                    category,
                    int(symbol_date_id),
                )

    def __len__(self) -> int:
        return self.size

    def clear(self) -> None:
        """Clear logical contents while retaining allocated arrays."""

        self.size = 0
        self.write_index = 0
        self.is_full = False
        self.priority_values.fill(1.0)
        self.event_flags.fill(0)
        self.event_multipliers.fill(1.0)
        self.symbol_ids.fill(-1)
        self.date_ids.fill(-1)
        self.symbol_date_ids.fill(-1)
        self.sample_call_count = 0
        self.priority_update_count = 0
        self.priority_nan_count = 0
        self.priority_inf_count = 0
        self.priority_clipping_count = 0
        self.is_weight_nan_count = 0
        self.is_weight_inf_count = 0
        self.event_bucket_empty_count = 0
        self.fallback_to_uniform_count = 0
        self.reward_hard_reject_count = 0
        self.sampled_event_counts_total.clear()
        self.sampled_position_counts_total.clear()
        self.sampled_bucket_counts_total.clear()
        self.sampled_symbol_counts_total.clear()
        self.sampled_date_counts_total.clear()
        self.sampled_symbol_date_counts_total.clear()
        self.sampled_bucket_symbol_counts_total.clear()
        self.sampled_bucket_date_counts_total.clear()
        self.sampled_bucket_symbol_date_counts_total.clear()
        self.sampled_event_category_symbol_counts_total.clear()
        self.sampled_event_category_date_counts_total.clear()
        self.sampled_event_category_symbol_date_counts_total.clear()

    def stats(self) -> dict:
        """Return lightweight diagnostics for currently stored transitions."""

        active = slice(0, self.size)
        return {
            "capacity": self.capacity,
            "size": self.size,
            "write_index": self.write_index,
            "is_full": self.is_full,
            "state_dim": self.state_dim,
            "action_count": self.action_count,
            "replay_sampling_mode": str(self.replay_sampling_mode),
            "per_alpha": float(self.per_alpha),
            "per_beta_start": float(self.per_beta_start),
            "per_beta_end": float(self.per_beta_end),
            "per_beta_current": float(self._current_beta()),
            "per_beta_steps": int(self.per_beta_steps),
            "priority_min": float(self.priority_min),
            "priority_max": float(self.priority_max),
            "priority_update_count": int(self.priority_update_count),
            "priority_nan_count": int(self.priority_nan_count),
            "priority_inf_count": int(self.priority_inf_count),
            "priority_clipping_count": int(self.priority_clipping_count),
            "is_weight_nan_count": int(self.is_weight_nan_count),
            "is_weight_inf_count": int(self.is_weight_inf_count),
            "event_bucket_empty_count": int(self.event_bucket_empty_count),
            "fallback_to_uniform_count": int(self.fallback_to_uniform_count),
            "reward_hard_reject_count": int(self.reward_hard_reject_count),
            "sample_call_count": int(self.sample_call_count),
            "nan_count_states": int(np.isnan(self.states[active]).sum()),
            "nan_count_next_states": int(np.isnan(self.next_states[active]).sum()),
            "done_count": int(self.dones[active].sum()),
            "nonfinite_reward_count": int(
                np.count_nonzero(~np.isfinite(self.rewards[active]))
            ),
            "action_mask_false_count": int((~self.action_masks[active]).sum()),
            "next_action_mask_false_count": int(
                (~self.next_action_masks[active]).sum()
            ),
            "policy_effective_action_mask_false_count": int(
                (~self.policy_effective_action_masks[active]).sum()
            ),
            "next_policy_effective_action_mask_false_count": int(
                (~self.next_policy_effective_action_masks[active]).sum()
            ),
            "metadata_position_before_present_count": int(
                (self.position_before_ids[active] >= 0).sum()
            ),
            "metadata_symbol_present_count": int((self.symbol_ids[active] >= 0).sum()),
            "metadata_date_present_count": int((self.date_ids[active] >= 0).sum()),
            "metadata_symbol_date_present_count": int(
                (self.symbol_date_ids[active] >= 0).sum()
            ),
            "metadata_local_action_present_count": int(
                (self.local_action_ids[active] >= 0).sum()
            ),
            "action_id_counts": _metadata_counts(self.actions[active]),
            "position_before_id_counts": _metadata_counts(
                self.position_before_ids[active]
            ),
            "position_after_id_counts": _metadata_counts(
                self.position_after_ids[active]
            ),
            "selected_head_id_counts": _metadata_counts(self.selected_head_ids[active]),
            "local_action_id_counts": _metadata_counts(self.local_action_ids[active]),
            "semantic_action_id_counts": _metadata_counts(
                self.semantic_action_ids[active]
            ),
            "legacy_action_id_counts": _metadata_counts(self.legacy_action_ids[active]),
            "event_flag_counts": _event_flag_counts(self.event_flags[active]),
            "sampled_event_counts_total": dict(self.sampled_event_counts_total),
            "sampled_position_counts_total": dict(self.sampled_position_counts_total),
            "sampled_bucket_counts_total": dict(self.sampled_bucket_counts_total),
            "symbol_id_counts": _metadata_counts(self.symbol_ids[active]),
            "date_id_counts": _metadata_counts(self.date_ids[active]),
            "symbol_date_id_counts": _metadata_counts(self.symbol_date_ids[active]),
            "sampled_symbol_counts_total": dict(self.sampled_symbol_counts_total),
            "sampled_date_counts_total": dict(self.sampled_date_counts_total),
            "sampled_symbol_date_counts_total": dict(
                self.sampled_symbol_date_counts_total
            ),
            "sampled_bucket_symbol_counts_total": _nested_dict_copy(
                self.sampled_bucket_symbol_counts_total
            ),
            "sampled_bucket_date_counts_total": _nested_dict_copy(
                self.sampled_bucket_date_counts_total
            ),
            "sampled_bucket_symbol_date_counts_total": _nested_dict_copy(
                self.sampled_bucket_symbol_date_counts_total
            ),
            "sampled_event_category_symbol_counts_total": _nested_dict_copy(
                self.sampled_event_category_symbol_counts_total
            ),
            "sampled_event_category_date_counts_total": _nested_dict_copy(
                self.sampled_event_category_date_counts_total
            ),
            "sampled_event_category_symbol_date_counts_total": _nested_dict_copy(
                self.sampled_event_category_symbol_date_counts_total
            ),
            "priority_stats_by_event": _priority_stats_by_event(
                self.priority_values[active],
                self.event_flags[active],
            ),
            "priority_stats_by_symbol": _priority_stats_by_group(
                self.priority_values[active],
                self.symbol_ids[active],
            ),
            "priority_stats_by_date": _priority_stats_by_group(
                self.priority_values[active],
                self.date_ids[active],
            ),
            "priority_stats_by_symbol_date": _priority_stats_by_group(
                self.priority_values[active],
                self.symbol_date_ids[active],
            ),
        }

    def _validate_state_vector(self, value, name: str) -> np.ndarray:
        if value is None:
            raise ValueError(f"{name} must not be None")
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (self.state_dim,):
            raise ValueError(
                f"{name} must have shape ({self.state_dim},), got {array.shape}"
            )
        return array

    def _validate_reward_or_reject(self, value, name: str) -> np.float32:
        try:
            return validate_finite_reward_scalar_v01(value, name=name)
        except (TypeError, ValueError, OverflowError):
            self.reward_hard_reject_count += 1
            raise

    def _validate_reward_array_or_reject(
        self,
        values,
        count: int,
    ) -> np.ndarray:
        try:
            raw = np.asarray(values)
            if raw.shape != (count,):
                raise ValueError(f"rewards must have shape ({count},)")
            if raw.dtype.kind not in {"i", "u", "f"}:
                raise ValueError("rewards must contain scalar numeric values")
            normalized = np.empty(count, dtype=np.float32)
            for index, value in enumerate(raw):
                normalized[index] = validate_finite_reward_scalar_v01(
                    value,
                    name=f"rewards[{index}]",
                )
            return normalized
        except (TypeError, ValueError, OverflowError):
            self.reward_hard_reject_count += 1
            raise

    def _validate_state_matrix(self, value, name: str) -> np.ndarray:
        if value is None:
            raise ValueError(f"{name} must not be None")
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.state_dim:
            raise ValueError(
                f"{name} must have shape (batch, {self.state_dim}), got {array.shape}"
            )
        return array

    def _validate_mask_or_default(
        self,
        mask,
        state_array: np.ndarray,
        name: str,
    ) -> np.ndarray:
        if mask is None:
            return ~np.isnan(state_array)
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != (self.state_dim,):
            raise ValueError(
                f"{name} must have shape ({self.state_dim},), got {mask_array.shape}"
            )
        return mask_array

    def _validate_mask_matrix_or_default(
        self,
        mask,
        state_matrix: np.ndarray,
        name: str,
    ) -> np.ndarray:
        if mask is None:
            return ~np.isnan(state_matrix)
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != state_matrix.shape:
            raise ValueError(
                f"{name} must have shape {state_matrix.shape}, got {mask_array.shape}"
            )
        return mask_array

    def _validate_action_mask_or_default(
        self,
        mask,
        state_array: np.ndarray,
        state_mask: np.ndarray,
        name: str,
        default_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        if mask is None:
            if default_mask is not None:
                return np.asarray(default_mask, dtype=bool).copy()
            return _infer_action_mask_from_state_v01(
                state_array,
                state_mask,
                action_count=self.action_count,
            )
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != (self.action_count,):
            raise ValueError(
                f"{name} must have shape ({self.action_count},), got {mask_array.shape}"
            )
        return mask_array

    def _validate_action_mask_matrix_or_default(
        self,
        masks,
        state_matrix: np.ndarray,
        state_mask_matrix: np.ndarray,
        name: str,
        default_masks: np.ndarray | None = None,
    ) -> np.ndarray:
        if masks is not None:
            mask_array = np.asarray(masks, dtype=bool)
            expected = (state_matrix.shape[0], self.action_count)
            if mask_array.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {mask_array.shape}")
            return mask_array
        if default_masks is not None:
            mask_array = np.asarray(default_masks, dtype=bool)
            expected = (state_matrix.shape[0], self.action_count)
            if mask_array.shape != expected:
                raise ValueError(
                    f"default {name} must have shape {expected}, got {mask_array.shape}"
                )
            return mask_array.copy()
        return np.asarray(
            [
                _infer_action_mask_from_state_v01(
                    state,
                    state_mask,
                    action_count=self.action_count,
                )
                for state, state_mask in zip(state_matrix, state_mask_matrix)
            ],
            dtype=bool,
        )


def _validate_action_id(action_id: int, *, action_count: int = ACTION_COUNT_V01) -> int:
    if isinstance(action_id, (bool, np.bool_)):
        raise ValueError(f"invalid action_id: {action_id!r}")
    try:
        normalized_id = int(action_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid action_id: {action_id!r}") from exc
    if normalized_id < 0 or normalized_id >= int(action_count):
        raise ValueError(f"invalid action_id: {action_id!r}")
    return normalized_id


def _validate_sampling_mode(value: str) -> str:
    mode = str(value)
    if mode not in REPLAY_SAMPLING_MODES_V01:
        raise ValueError(
            "replay_sampling_mode must be one of "
            + ", ".join(sorted(REPLAY_SAMPLING_MODES_V01))
        )
    return mode


def _validate_unit_float(value: float, name: str, *, allow_zero: bool) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float") from exc
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    lower_ok = normalized >= 0.0 if allow_zero else normalized > 0.0
    if not lower_ok or normalized > 1.0:
        raise ValueError(f"{name} must be in {'[0, 1]' if allow_zero else '(0, 1]'}")
    return normalized


def _validate_positive_float(value: float, name: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive finite") from exc
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be positive finite")
    return normalized


def _validate_event_bucket_mix(value: dict[str, float] | None) -> dict[str, float]:
    source = DEFAULT_EVENT_BALANCED_BUCKET_MIX_V01 if value is None else dict(value)
    normalized: dict[str, float] = {}
    for key, raw in source.items():
        bucket = str(key)
        try:
            ratio = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"event bucket ratio must be numeric: {bucket}") from exc
        if not np.isfinite(ratio) or ratio < 0.0:
            raise ValueError(f"event bucket ratio must be nonnegative finite: {bucket}")
        normalized[bucket] = ratio
    total = float(sum(normalized.values()))
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("event_bucket_mix must have positive finite total weight")
    return {key: float(value) / total for key, value in normalized.items()}


def _optional_event_flag(value: int | None) -> int:
    if value is None:
        return 0
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"invalid event flag: {value!r}")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid event flag: {value!r}") from exc
    if normalized < 0:
        raise ValueError("event flag must be nonnegative")
    return normalized


def _event_flag_array(values, count: int) -> np.ndarray:
    if values is None:
        return np.zeros(int(count), dtype=np.int32)
    array = np.asarray(values, dtype=np.int32)
    if array.shape != (int(count),):
        raise ValueError(f"event_flags must have shape ({int(count)},)")
    if np.any(array < 0):
        raise ValueError("event_flags must be nonnegative")
    return array


def _normalize_event_multiplier(
    value: float | None,
    *,
    max_event_multiplier: float,
) -> np.float32:
    if value is None:
        return np.float32(1.0)
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid event multiplier: {value!r}") from exc
    if not np.isfinite(normalized):
        raise ValueError("event multiplier must be finite")
    normalized = float(np.clip(normalized, 1.0, float(max_event_multiplier)))
    return np.float32(normalized)


def _event_multiplier_array(
    values,
    count: int,
    *,
    max_event_multiplier: float,
) -> np.ndarray:
    if values is None:
        return np.ones(int(count), dtype=np.float32)
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (int(count),):
        raise ValueError(f"event_multipliers must have shape ({int(count)},)")
    if not np.isfinite(array).all():
        raise ValueError("event_multipliers must be finite")
    return np.clip(array, 1.0, float(max_event_multiplier)).astype(np.float32)


def _normalize_sampling_weights(weights: np.ndarray) -> np.ndarray | None:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("sampling weights must be one-dimensional")
    values = np.where(np.isfinite(values), values, 0.0)
    values = np.maximum(values, 0.0)
    total = float(values.sum())
    if total <= 0.0 or not np.isfinite(total):
        return None
    return values / total


def _has_flag(flags: np.ndarray, flag: int) -> np.ndarray:
    return (np.asarray(flags, dtype=np.int32) & int(flag)) != 0


def _finite_stat_np(values: np.ndarray, stat: str) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None
    if stat == "min":
        return float(np.min(finite))
    if stat == "max":
        return float(np.max(finite))
    if stat == "mean":
        return float(np.mean(finite))
    raise ValueError(f"unknown stat: {stat!r}")


def _finite_percentile_np(values: np.ndarray, percentile: float) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None
    return float(np.percentile(finite, float(percentile)))


def _event_flag_counts(values: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for flag in np.asarray(values, dtype=np.int32).tolist():
        for name in event_flag_names_from_mask_v01(int(flag)):
            counts[name] = int(counts.get(name, 0)) + 1
    return counts


def _priority_stats_by_event(
    priorities: np.ndarray,
    flags: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    priority_array = np.asarray(priorities, dtype=np.float64)
    flag_array = np.asarray(flags, dtype=np.int32)
    for flag, name in REPLAY_EVENT_FLAG_NAMES_V01.items():
        mask = _has_flag(flag_array, int(flag))
        values = priority_array[mask]
        result[name] = {
            "count": int(values.size),
            "avg": _finite_stat_np(values, "mean"),
            "p50": _finite_percentile_np(values, 50.0),
            "p90": _finite_percentile_np(values, 90.0),
            "p99": _finite_percentile_np(values, 99.0),
        }
    return result


def _optional_metadata_id(value: int | None) -> int:
    if value is None:
        return -1
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"invalid metadata id: {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid metadata id: {value!r}") from exc


def _metadata_id_array(values, count: int) -> np.ndarray:
    if values is None:
        return np.full(int(count), -1, dtype=np.int32)
    array = np.asarray(values, dtype=np.int32)
    if array.shape != (int(count),):
        raise ValueError(f"metadata id array must have shape ({int(count)},)")
    return array


def _optional_close_aux_label(value: float | int | None) -> np.float32:
    if value is None:
        return np.float32(0.0)
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid close_aux_label: {value!r}") from exc
    if not np.isfinite(normalized) or normalized not in {0.0, 1.0}:
        raise ValueError("close_aux_label must be finite 0.0 or 1.0")
    return np.float32(normalized)


def _optional_close_aux_valid_mask(value: bool | None, *, label: float) -> bool:
    if value is None:
        return False
    return bool(value) and bool(np.isfinite(float(label)))


def _close_aux_label_array(values, count: int) -> np.ndarray:
    if values is None:
        return np.zeros(int(count), dtype=np.float32)
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (int(count),):
        raise ValueError(f"close_aux_labels must have shape ({int(count)},)")
    finite = np.isfinite(array)
    valid_domain = (array == 0.0) | (array == 1.0)
    if not bool(np.all(finite & valid_domain)):
        raise ValueError("close_aux_labels must contain only finite 0.0/1.0")
    return array


def _close_aux_valid_mask_array(
    values,
    count: int,
    *,
    labels: np.ndarray,
) -> np.ndarray:
    if values is None:
        return np.zeros(int(count), dtype=bool)
    array = np.asarray(values, dtype=bool)
    if array.shape != (int(count),):
        raise ValueError(
            f"close_aux_label_valid_masks must have shape ({int(count)},)"
        )
    label_array = np.asarray(labels, dtype=np.float32)
    return array & np.isfinite(label_array)


def _metadata_counts(values: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in np.asarray(values, dtype=np.int64).tolist():
        normalized = int(value)
        if normalized < 0:
            continue
        key = str(normalized)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _optional_symbol_date_id(
    value: int | None,
    *,
    symbol_id: int,
    date_id: int,
) -> int:
    if value is not None:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"invalid symbol_date id: {value!r}")
        try:
            normalized = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid symbol_date id: {value!r}") from exc
        return normalized
    if int(symbol_id) < 0 or int(date_id) < 0:
        return -1
    return int(date_id) * 1_000_000 + int(symbol_id)


def _symbol_date_id_array(
    values,
    count: int,
    *,
    symbol_ids: np.ndarray,
    date_ids: np.ndarray,
) -> np.ndarray:
    if values is not None:
        array = np.asarray(values, dtype=np.int64)
        if array.shape != (int(count),):
            raise ValueError(f"symbol_date id array must have shape ({int(count)},)")
        return array
    symbols = np.asarray(symbol_ids, dtype=np.int64)
    dates = np.asarray(date_ids, dtype=np.int64)
    if symbols.shape != (int(count),) or dates.shape != (int(count),):
        raise ValueError("symbol/date id arrays must match count")
    result = np.full(int(count), -1, dtype=np.int64)
    mask = (symbols >= 0) & (dates >= 0)
    result[mask] = dates[mask] * 1_000_000 + symbols[mask]
    return result


def _record_group_counts(values: np.ndarray, target: dict[str, int]) -> None:
    for value in np.asarray(values, dtype=np.int64).tolist():
        normalized = int(value)
        if normalized < 0:
            continue
        key = str(normalized)
        target[key] = int(target.get(key, 0)) + 1


def _record_nested_group_count(
    target: dict[str, dict[str, int]],
    bucket: str,
    value: int,
) -> None:
    normalized = int(value)
    if normalized < 0:
        return
    nested = target.setdefault(str(bucket), {})
    key = str(normalized)
    nested[key] = int(nested.get(key, 0)) + 1


def _nested_dict_copy(source: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {
        str(bucket): {str(key): int(value) for key, value in counts.items()}
        for bucket, counts in source.items()
    }


def _event_categories_from_mask(mask: int) -> list[str]:
    value = int(mask)
    categories: list[str] = []
    if value == 0:
        return ["background"]
    if value & (
        REPLAY_EVENT_CLOSE_LABEL_CLOSE
        | REPLAY_EVENT_LONG_CLOSE
        | REPLAY_EVENT_SHORT_CLOSE
    ):
        categories.append("close_event")
    if value & REPLAY_EVENT_ADVERSE_HOLD_CLOSE_LABEL:
        categories.append("adverse_hold")
    if value & (REPLAY_EVENT_FORCED_OR_LATE_CLOSE_WINDOW | REPLAY_EVENT_FORCED_EXIT):
        categories.append("forced_exit_or_late_close")
    if value & (
        REPLAY_EVENT_GENERIC_ENTRY
        | REPLAY_EVENT_NO_EDGE_ENTRY
        | REPLAY_EVENT_HIGH_CONFIDENCE_WRONG_ENTRY
    ):
        categories.append("entry_event")
    if not categories:
        categories.append("rare_event")
    elif value & REPLAY_EVENT_RARE_FALLBACK:
        categories.append("rare_event")
    return categories


def _priority_stats_by_group(
    priorities: np.ndarray,
    groups: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    priority_array = np.asarray(priorities, dtype=np.float64)
    group_array = np.asarray(groups, dtype=np.int64)
    for group in sorted(int(value) for value in np.unique(group_array) if int(value) >= 0):
        values = priority_array[group_array == int(group)]
        result[str(group)] = {
            "count": int(values.size),
            "avg": _finite_stat_np(values, "mean"),
            "p50": _finite_percentile_np(values, 50.0),
            "p90": _finite_percentile_np(values, 90.0),
            "p99": _finite_percentile_np(values, 99.0),
            "max": _finite_stat_np(values, "max"),
        }
    return result


def _validate_action_count(value: int) -> int:
    normalized = _validate_positive_int(value, "action_count")
    if normalized not in {ACTION_COUNT_V01, 5, 7}:
        raise ValueError("action_count must be 3, 5, or 7")
    return normalized


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        normalized_value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if normalized_value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return normalized_value


def _infer_action_mask_from_state_v01(
    state: np.ndarray,
    valid_state_mask: np.ndarray | None = None,
    *,
    action_count: int = ACTION_COUNT_V01,
) -> np.ndarray:
    """Infer train/replay order [Hold, Buy, Sell] mask from position one-hot state."""

    state_array = np.asarray(state, dtype=np.float32)
    mask_array = (
        np.ones_like(state_array, dtype=bool)
        if valid_state_mask is None
        else np.asarray(valid_state_mask, dtype=bool)
    )
    if (
        state_array.shape[0] <= POSITION_SIDE_SHORT_INDEX_V01
        or mask_array.shape != state_array.shape
        or not bool(mask_array[POSITION_SIDE_FLAT_INDEX_V01])
        or not bool(mask_array[POSITION_SIDE_LONG_INDEX_V01])
        or not bool(mask_array[POSITION_SIDE_SHORT_INDEX_V01])
    ):
        return np.ones(action_count, dtype=bool)

    flat = float(state_array[POSITION_SIDE_FLAT_INDEX_V01])
    long = float(state_array[POSITION_SIDE_LONG_INDEX_V01])
    short = float(state_array[POSITION_SIDE_SHORT_INDEX_V01])
    if not np.isfinite([flat, long, short]).all():
        return np.ones(action_count, dtype=bool)
    is_flat = flat > 0.5 and long <= 0.5 and short <= 0.5
    is_long = long > 0.5 and flat <= 0.5 and short <= 0.5
    is_short = short > 0.5 and flat <= 0.5 and long <= 0.5
    if int(action_count) == 5:
        if is_flat:
            return np.asarray([True, True, False, True, False], dtype=bool)
        if is_long:
            return np.asarray([True, False, True, False, False], dtype=bool)
        if is_short:
            return np.asarray([True, False, False, False, True], dtype=bool)
        return np.ones(action_count, dtype=bool)
    if int(action_count) == 7:
        if is_flat:
            return np.asarray([True, True, True, False, False, False, False], dtype=bool)
        if is_long:
            return np.asarray([False, False, False, True, True, False, False], dtype=bool)
        if is_short:
            return np.asarray([False, False, False, False, False, True, True], dtype=bool)
        return np.ones(action_count, dtype=bool)
    if is_flat:
        return np.asarray([True, True, True], dtype=bool)
    if is_long:
        return np.asarray([True, False, True], dtype=bool)
    if is_short:
        return np.asarray([True, True, False], dtype=bool)
    return np.ones(action_count, dtype=bool)
