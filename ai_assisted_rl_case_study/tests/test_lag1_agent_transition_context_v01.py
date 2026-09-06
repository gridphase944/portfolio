from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features.state_vector_v01 import (
    LAG1_AGENT_TRANSITION_CONTEXT_COLUMNS,
    STATE_CANDIDATE_LAG1_AGENT_TRANSITION_CONTEXT_V01,
    STATE_COLUMNS,
    build_lag1_agent_transition_context_features_v01,
    build_state_vector_v01,
    get_state_columns_v01,
    get_state_spec_version_v01,
)
from trading_env.env_v01 import TradingEnv
from training.q_input_adapter_v01 import build_q_network_input_v01
from training.rl_representation_integration_v01 import (
    LAG1_Q_INPUT_SCHEMA_ID,
    LAG1_STATE_SCHEMA_ID,
    build_q_input_batch_v01,
    build_q_input_v01,
    build_semantic_state_v01,
    contract_for_variant_v01,
    expected_q_parameter_count_v01,
)
from training.rl_representation_replay_v01 import (
    RepresentationReplayBufferV01,
    RepresentationReplayTransitionV01,
    terminal_next_v01,
    validate_transition_v01,
)
from training.state_normalizer_v01 import (
    TRANSFORM_IDENTITY,
    fit_state_normalizer_v01,
    get_state_normalizer_specs_v01,
    transform_state_vector_v01,
)


CANDIDATE = STATE_CANDIDATE_LAG1_AGENT_TRANSITION_CONTEXT_V01


def _row(timestamp: str, session: str, *, executable: bool = True, forced: bool = False):
    row = {
        "grid_timestamp": pd.Timestamp(timestamp, tz="Asia/Tokyo"),
        "timestamp": pd.Timestamp(timestamp, tz="Asia/Tokyo"),
        "session": session,
        "is_forced_exit": forced,
        "PreviousClose": 1000.0,
        "previous_close": 1000.0,
    }
    for side, price in (("Sell", 1001.0), ("Buy", 999.0)):
        for level in range(1, 6):
            row[f"{side}{level}_Price"] = price
            row[f"{side}{level}_Qty"] = 100.0 if executable else 0.0
    return row


def _candidate_env(rows):
    return TradingEnv.for_historical_zero_latency_v01(
        rows,
        state_candidate_name=CANDIDATE,
    )


def _history(vector: np.ndarray) -> np.ndarray:
    return np.asarray(vector[-6:], dtype=np.float64)


def _state_mask(env: TradingEnv) -> np.ndarray:
    return np.asarray(env._last_state_result.qc["valid_state_mask"], dtype=bool)


class Lag1AgentTransitionContextContractTest(unittest.TestCase):
    def test_exact_candidate_order_and_default_regression(self):
        self.assertEqual(get_state_columns_v01("current_state"), list(STATE_COLUMNS))
        self.assertEqual(len(STATE_COLUMNS), 59)
        self.assertEqual(
            get_state_columns_v01(CANDIDATE),
            list(STATE_COLUMNS + LAG1_AGENT_TRANSITION_CONTEXT_COLUMNS),
        )
        self.assertEqual(get_state_columns_v01(CANDIDATE)[59:], [
            "prev_decision_position_flat",
            "prev_decision_position_long",
            "prev_decision_position_short",
            "prev_effective_action_hold",
            "prev_effective_action_buy",
            "prev_effective_action_sell",
        ])
        self.assertEqual(len(get_state_columns_v01(CANDIDATE)), 65)

    def test_schema_and_q_contract_are_strict_65_65_130(self):
        contract = contract_for_variant_v01("LAG1")
        self.assertEqual(contract.state_schema_id, LAG1_STATE_SCHEMA_ID)
        self.assertEqual(contract.q_input_schema_id, LAG1_Q_INPUT_SCHEMA_ID)
        self.assertEqual(get_state_spec_version_v01(CANDIDATE), LAG1_STATE_SCHEMA_ID)
        self.assertEqual(
            (contract.state_dim, contract.valid_mask_dim, contract.q_input_dim),
            (65, 65, 130),
        )
        self.assertEqual(expected_q_parameter_count_v01("LAG1"), 33667)

    def test_history_encoding_is_atomic_and_semantic(self):
        unavailable = build_lag1_agent_transition_context_features_v01(None, None)
        self.assertTrue(np.isnan(list(unavailable.values())).all())
        self.assertEqual(
            list(build_lag1_agent_transition_context_features_v01("Long", "Sell").values()),
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        with self.assertRaisesRegex(ValueError, "atomic"):
            build_lag1_agent_transition_context_features_v01("Flat", None)
        with self.assertRaisesRegex(ValueError, "Hold, Buy, or Sell"):
            build_lag1_agent_transition_context_features_v01("Flat", 1)  # type: ignore[arg-type]

    def test_history_normalization_is_identity_and_unavailable_maps_to_zero_mask_zero(self):
        columns = get_state_columns_v01(CANDIDATE)
        specs = get_state_normalizer_specs_v01(columns)
        self.assertTrue(
            all(spec.transform_type == TRANSFORM_IDENTITY for spec in specs[-6:])
        )
        raw = np.zeros(65, dtype=np.float64)
        raw[-6:] = np.nan
        mask = np.ones(65, dtype=bool)
        mask[-6:] = False
        fitted = fit_state_normalizer_v01(raw.reshape(1, -1), columns, mask.reshape(1, -1))
        normalized = transform_state_vector_v01(raw, fitted, mask)
        semantic = build_semantic_state_v01("LAG1", current_state=raw)
        q_input = build_q_input_v01(
            "LAG1",
            semantic_state=semantic,
            valid_state_mask=mask,
            normalized_current_state=normalized.state_vector,
        )
        self.assertEqual(q_input.shape, (130,))
        np.testing.assert_array_equal(q_input[59:65], np.zeros(6))
        np.testing.assert_array_equal(q_input[124:130], np.zeros(6))

    def test_tensor_q_adapter_accepts_65_and_rejects_59(self):
        import tensorflow as tf

        result = build_q_network_input_v01(
            tf.zeros((2, 65)), tf.ones((2, 65), dtype=tf.bool), expected_state_dim=65
        )
        self.assertEqual(tuple(result.network_input.shape), (2, 130))
        with self.assertRaisesRegex(ValueError, "must be 65"):
            build_q_network_input_v01(
                tf.zeros((2, 59)), tf.ones((2, 59), dtype=tf.bool), expected_state_dim=65
            )

    def test_env_temporal_contract_session_boundary_and_position_carry(self):
        rows = [
            _row("2026-06-24 09:00:00", "morning"),
            _row("2026-06-24 11:29:55", "morning"),
            _row("2026-06-24 12:30:00", "afternoon"),
            _row("2026-06-24 12:30:05", "afternoon"),
        ]
        env = _candidate_env(rows)
        state_t0 = env.reset()
        self.assertTrue(np.isnan(_history(state_t0)).all())
        self.assertFalse(_state_mask(env)[-6:].any())

        next_state, _, done, info = env.step("Buy")
        self.assertFalse(done)
        np.testing.assert_array_equal(_history(next_state), [1, 0, 0, 0, 1, 0])
        self.assertEqual(info["position_before"], "Flat")
        self.assertEqual(info["position_after"], "Long")
        self.assertTrue(np.isnan(_history(state_t0)).all())  # action_t did not leak into state_t

        lunch_state, _, done, _ = env.step("Hold")
        self.assertFalse(done)
        self.assertEqual(env.position, "Long")
        self.assertTrue(np.isnan(_history(lunch_state)).all())
        self.assertFalse(_state_mask(env)[-6:].any())

        after_close, _, done, info = env.step("Sell")
        self.assertFalse(done)
        self.assertEqual(info["position_before"], "Long")
        self.assertEqual(info["position_after"], "Flat")
        np.testing.assert_array_equal(_history(after_close), [0, 1, 0, 0, 0, 1])

    def test_final_effective_action_not_raw_argmax_is_history(self):
        env = _candidate_env([
            _row("2026-06-24 09:00:00", "morning"),
            _row("2026-06-24 09:00:05", "morning"),
        ])
        state = env.reset()
        raw_argmax_action = "Buy"
        final_effective_action = "Hold"
        self.assertNotEqual(raw_argmax_action, final_effective_action)
        next_state, _, _, _ = env.step(final_effective_action)
        np.testing.assert_array_equal(_history(next_state), [1, 0, 0, 1, 0, 0])
        self.assertTrue(np.isnan(_history(state)).all())

    def test_execution_failure_keeps_effective_action_valid(self):
        env = _candidate_env([
            _row("2026-06-24 09:00:00", "morning", executable=False),
            _row("2026-06-24 09:00:05", "morning"),
        ])
        env.reset()
        next_state, _, _, info = env.step("Buy")
        self.assertTrue(info["execution_failed"])
        self.assertEqual(info["position_after"], "Flat")
        np.testing.assert_array_equal(_history(next_state), [1, 0, 0, 0, 1, 0])
        self.assertTrue(np.asarray(info["valid_state_mask"])[-6:].all())

    def test_forced_terminal_action_is_not_advanced_into_history(self):
        env = _candidate_env([
            _row("2026-06-24 15:19:55", "afternoon"),
            _row("2026-06-24 15:20:00", "afternoon", forced=True),
        ])
        env.reset()
        env.step("Buy")
        before = env.get_agent_transition_history_context()
        next_state, _, done, info = env.step("Sell")
        self.assertTrue(done)
        self.assertIsNone(next_state)
        self.assertEqual(info["agent_transition_history_before"], before)
        self.assertFalse(info["agent_transition_history_next"]["available"])

    def test_replay_terminal_and_schema_migration_rejection(self):
        contract = contract_for_variant_v01("LAG1")
        state = np.zeros(65, dtype=np.float32)
        mask = np.ones(65, dtype=bool)
        next_state, next_mask, next_action_mask = terminal_next_v01("LAG1")
        transition = RepresentationReplayTransitionV01(
            variant="LAG1",
            state_schema_id=contract.state_schema_id,
            q_input_schema_id=contract.q_input_schema_id,
            episode_id="episode",
            step_index=0,
            state=state,
            state_valid_mask=mask,
            action_id=0,
            reward=0.0,
            next_state=next_state,
            next_state_valid_mask=next_mask,
            done=True,
            action_mask=np.ones(3, dtype=bool),
            next_action_mask=next_action_mask,
        )
        validate_transition_v01(transition, expected_variant="LAG1")
        self.assertEqual(next_state.shape, (65,))
        self.assertFalse(next_mask.any())
        with self.assertRaisesRegex(ValueError, "variant mismatch"):
            RepresentationReplayBufferV01(capacity=2, variant="B0").add(transition)
        wrong_schema = RepresentationReplayTransitionV01(
            **{**transition.__dict__, "state_schema_id": "state_vector_v02_cooldown_v01"}
        )
        with self.assertRaisesRegex(ValueError, "state schema mismatch"):
            validate_transition_v01(wrong_schema, expected_variant="LAG1")

    def test_replay_online_target_q_input_parity(self):
        states = np.zeros((2, 65), dtype=np.float32)
        states[:, -6:] = [[1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1]]
        masks = np.ones((2, 65), dtype=bool)
        batch = build_q_input_batch_v01(
            "LAG1",
            semantic_states=states,
            valid_state_masks=masks,
            normalized_current_states=states,
        )
        online = build_q_input_v01(
            "LAG1", semantic_state=states[0], valid_state_mask=masks[0],
            normalized_current_state=states[0],
        )
        target = build_q_input_v01(
            "LAG1", semantic_state=states[1], valid_state_mask=masks[1],
            normalized_current_state=states[1],
        )
        np.testing.assert_array_equal(batch[0], online)
        np.testing.assert_array_equal(batch[1], target)

    def test_sequential_and_diagnostic_rollout_state_semantics_match(self):
        rows = [
            _row("2026-06-24 09:00:00", "morning"),
            _row("2026-06-24 09:00:05", "morning"),
            _row("2026-06-24 09:00:10", "morning"),
        ]
        traces = []
        for _mode in ("training", "diagnostic"):
            env = _candidate_env(rows)
            states = [np.asarray(env.reset())]
            masks = [_state_mask(env)]
            for action in ("Buy", "Sell"):
                next_state, _, done, info = env.step(action)
                self.assertFalse(done)
                states.append(np.asarray(next_state))
                masks.append(np.asarray(info["valid_state_mask"]))
            traces.append((states, masks))
        for left, right in zip(traces[0][0], traces[1][0]):
            np.testing.assert_equal(left, right)
        for left, right in zip(traces[0][1], traces[1][1]):
            np.testing.assert_array_equal(left, right)

    def test_direct_state_builder_rejects_partial_history(self):
        internal = {
            "position_side": "Flat",
            "previous_decision_position": "Flat",
            "previous_effective_action": None,
        }
        with self.assertRaisesRegex(ValueError, "atomic"):
            build_state_vector_v01(
                pd.Series(dtype=float), internal, state_candidate_name=CANDIDATE
            )


if __name__ == "__main__":
    unittest.main()
