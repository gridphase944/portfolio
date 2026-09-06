from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features.state_vector_v01 import (
    LAG1_AGENT_TRANSITION_CONTEXT_COLUMNS,
    LAG2_AGENT_TRANSITION_CONTEXT_COLUMNS,
    LAG3_AGENT_TRANSITION_CONTEXT_COLUMNS,
    STATE_CANDIDATE_LAG1_AGENT_TRANSITION_CONTEXT_V01,
    STATE_CANDIDATE_LAG2_AGENT_TRANSITION_CONTEXT_V01,
    STATE_CANDIDATE_LAG3_AGENT_TRANSITION_CONTEXT_V01,
    STATE_COLUMNS,
    build_agent_transition_history_features_v01,
    get_state_columns_v01,
)
from trading_env.env_v01 import TradingEnv
from training.agent_history_observability_v01 import (
    reconstruct_close_reentry_diagnostics_v01,
)
from training.q_input_adapter_v01 import build_q_network_input_v01
from training.rl_representation_integration_v01 import (
    build_q_input_batch_v01,
    build_q_input_v01,
    build_semantic_state_v01,
    build_variant_q_network_v01,
    contract_for_variant_v01,
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


CANDIDATES = {
    "LAG2": STATE_CANDIDATE_LAG2_AGENT_TRANSITION_CONTEXT_V01,
    "LAG3": STATE_CANDIDATE_LAG3_AGENT_TRANSITION_CONTEXT_V01,
}


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


def _env(variant: str, rows):
    return TradingEnv.for_historical_zero_latency_v01(
        rows,
        state_candidate_name=CANDIDATES[variant],
    )


def _mask(env: TradingEnv) -> np.ndarray:
    return np.asarray(env._last_state_result.qc["valid_state_mask"], dtype=bool)


def _groups(state: np.ndarray, depth: int) -> np.ndarray:
    return np.asarray(state[-6 * depth:], dtype=np.float64).reshape(depth, 6)


class Lag2Lag3SchemaContractTest(unittest.TestCase):
    def test_exact_columns_and_semantic_prefixes(self):
        self.assertEqual(len(STATE_COLUMNS), 59)
        self.assertEqual(get_state_columns_v01("current_state"), list(STATE_COLUMNS))
        self.assertEqual(
            get_state_columns_v01(STATE_CANDIDATE_LAG1_AGENT_TRANSITION_CONTEXT_V01),
            list(STATE_COLUMNS + LAG1_AGENT_TRANSITION_CONTEXT_COLUMNS),
        )
        lag2 = get_state_columns_v01(CANDIDATES["LAG2"])
        lag3 = get_state_columns_v01(CANDIDATES["LAG3"])
        self.assertEqual(lag2, list(STATE_COLUMNS + LAG2_AGENT_TRANSITION_CONTEXT_COLUMNS))
        self.assertEqual(lag3, list(STATE_COLUMNS + LAG3_AGENT_TRANSITION_CONTEXT_COLUMNS))
        self.assertEqual(lag2[:59], list(STATE_COLUMNS))
        self.assertEqual(lag3[:71], lag2)
        self.assertEqual(
            [name.replace("prev1_", "prev_") for name in lag2[59:65]],
            list(LAG1_AGENT_TRANSITION_CONTEXT_COLUMNS),
        )
        self.assertEqual((len(lag2), len(lag3)), (71, 77))

    def test_contract_mask_and_q_dimensions(self):
        expected = {"B0": (59, 59, 118), "LAG1": (65, 65, 130),
                    "LAG2": (71, 71, 142), "LAG3": (77, 77, 154)}
        for variant, dimensions in expected.items():
            contract = contract_for_variant_v01(variant)
            self.assertEqual(
                (contract.state_dim, contract.valid_mask_dim, contract.q_input_dim),
                dimensions,
            )

    def test_all_candidate_history_normalizers_are_identity(self):
        for variant, depth in (("LAG2", 2), ("LAG3", 3)):
            columns = get_state_columns_v01(CANDIDATES[variant])
            specs = get_state_normalizer_specs_v01(columns)
            self.assertTrue(
                all(spec.transform_type == TRANSFORM_IDENTITY for spec in specs[-6 * depth:])
            )
            raw = np.zeros(len(columns), dtype=np.float64)
            raw[-6 * depth:] = np.nan
            mask = np.ones(len(columns), dtype=bool)
            mask[-6 * depth:] = False
            fitted = fit_state_normalizer_v01(raw[None, :], columns, mask[None, :])
            normalized = transform_state_vector_v01(raw, fitted, mask)
            semantic = build_semantic_state_v01(variant, current_state=raw)
            q_input = build_q_input_v01(
                variant,
                semantic_state=semantic,
                valid_state_mask=mask,
                normalized_current_state=normalized.state_vector,
            )
            self.assertEqual(q_input.shape, (2 * len(columns),))
            self.assertTrue(np.isfinite(q_input).all())
            np.testing.assert_array_equal(q_input[-6 * depth:], 0.0)

    def test_generic_encoder_rejects_partial_or_gapped_availability(self):
        with self.assertRaisesRegex(ValueError, "atomic"):
            build_agent_transition_history_features_v01(
                [("Flat", None)], state_candidate_name=CANDIDATES["LAG2"]
            )
        with self.assertRaisesRegex(ValueError, "newest-first prefix"):
            build_agent_transition_history_features_v01(
                [None, ("Flat", "Hold")],
                state_candidate_name=CANDIDATES["LAG2"],
            )

    def test_tensor_adapter_and_replay_q_batch_shapes(self):
        import tensorflow as tf

        for variant, dim in (("LAG2", 71), ("LAG3", 77)):
            tensor = build_q_network_input_v01(
                tf.zeros((2, dim)),
                tf.ones((2, dim), dtype=tf.bool),
                expected_state_dim=dim,
            )
            self.assertEqual(tuple(tensor.network_input.shape), (2, dim * 2))
            states = np.zeros((2, dim), dtype=np.float32)
            masks = np.ones((2, dim), dtype=bool)
            q_batch = build_q_input_batch_v01(
                variant,
                semantic_states=states,
                valid_state_masks=masks,
                normalized_current_states=states,
            )
            self.assertEqual(q_batch.shape, (2, dim * 2))

    def test_network_architecture_and_cross_schema_weight_rejection(self):
        networks = {variant: build_variant_q_network_v01(variant) for variant in CANDIDATES}
        self.assertEqual(networks["LAG2"].input_shape[-1], 142)
        self.assertEqual(networks["LAG3"].input_shape[-1], 154)
        self.assertEqual(networks["LAG2"].output_shape[-1], 3)
        self.assertEqual(networks["LAG3"].output_shape[-1], 3)
        self.assertNotEqual(networks["LAG2"].count_params(), networks["LAG3"].count_params())
        with self.assertRaises(ValueError):
            networks["LAG3"].set_weights(networks["LAG2"].get_weights())


class Lag2Lag3TemporalEnvTest(unittest.TestCase):
    def test_lag3_shift_warmup_session_reset_and_position_continuity(self):
        rows = [
            _row("2026-06-24 09:00:00", "morning"),
            _row("2026-06-24 09:00:05", "morning"),
            _row("2026-06-24 09:00:10", "morning"),
            _row("2026-06-24 11:29:55", "morning"),
            _row("2026-06-24 12:30:00", "afternoon"),
            _row("2026-06-24 12:30:05", "afternoon"),
        ]
        env = _env("LAG3", rows)
        first = env.reset()
        self.assertTrue(np.isnan(_groups(first, 3)).all())
        self.assertFalse(_mask(env)[-18:].any())

        second, _, _, info = env.step("Buy")
        np.testing.assert_array_equal(_groups(second, 3)[0], [1, 0, 0, 0, 1, 0])
        self.assertTrue(np.isnan(_groups(second, 3)[1:]).all())
        self.assertEqual(info["position_before"], "Flat")
        self.assertEqual(info["position_after"], "Long")
        self.assertTrue(np.isnan(_groups(first, 3)).all())

        third, _, _, _ = env.step("Hold")
        np.testing.assert_array_equal(_groups(third, 3)[0], [0, 1, 0, 1, 0, 0])
        np.testing.assert_array_equal(_groups(third, 3)[1], [1, 0, 0, 0, 1, 0])
        self.assertTrue(np.isnan(_groups(third, 3)[2]).all())

        fourth, _, _, _ = env.step("Hold")
        self.assertTrue(np.isfinite(_groups(fourth, 3)).all())
        np.testing.assert_array_equal(_groups(fourth, 3)[2], [1, 0, 0, 0, 1, 0])

        afternoon_first, _, _, _ = env.step("Hold")
        self.assertEqual(env.position, "Long")
        self.assertTrue(np.isnan(_groups(afternoon_first, 3)).all())
        self.assertFalse(_mask(env)[-18:].any())

        after_close, _, done, info = env.step("Sell")
        self.assertFalse(done)
        self.assertEqual((info["position_before"], info["position_after"]), ("Long", "Flat"))
        np.testing.assert_array_equal(_groups(after_close, 3)[0], [0, 1, 0, 0, 0, 1])
        self.assertTrue(np.isnan(_groups(after_close, 3)[1:]).all())

    def test_lag2_shift_and_hold_buy_sell_effective_semantics(self):
        env = _env("LAG2", [
            _row("2026-06-24 09:00:00", "morning"),
            _row("2026-06-24 09:00:05", "morning"),
            _row("2026-06-24 09:00:10", "morning"),
            _row("2026-06-24 09:00:15", "morning"),
        ])
        state_t = env.reset()
        raw_argmax_action = "Buy"
        final_effective_action = "Hold"
        next_state, _, _, _ = env.step(final_effective_action)
        self.assertNotEqual(raw_argmax_action, final_effective_action)
        np.testing.assert_array_equal(_groups(next_state, 2)[0], [1, 0, 0, 1, 0, 0])
        self.assertTrue(np.isnan(_groups(state_t, 2)).all())
        next_state, _, _, _ = env.step("Sell")
        np.testing.assert_array_equal(_groups(next_state, 2)[0], [1, 0, 0, 0, 0, 1])
        np.testing.assert_array_equal(_groups(next_state, 2)[1], [1, 0, 0, 1, 0, 0])
        next_state, _, _, _ = env.step("Buy")
        np.testing.assert_array_equal(_groups(next_state, 2)[0], [0, 0, 1, 0, 1, 0])

    def test_execution_failure_keeps_final_effective_action_valid(self):
        for variant, depth in (("LAG2", 2), ("LAG3", 3)):
            env = _env(variant, [
                _row("2026-06-24 09:00:00", "morning", executable=False),
                _row("2026-06-24 09:00:05", "morning"),
            ])
            env.reset()
            next_state, _, _, info = env.step("Sell")
            self.assertTrue(info["execution_failed"])
            self.assertEqual(info["position_after"], "Flat")
            np.testing.assert_array_equal(_groups(next_state, depth)[0], [1, 0, 0, 0, 0, 1])
            self.assertTrue(np.isnan(_groups(next_state, depth)[1:]).all())

    def test_forced_terminal_action_is_not_shifted(self):
        for variant in CANDIDATES:
            env = _env(variant, [
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


class Lag2Lag3ReplayAndObservabilityTest(unittest.TestCase):
    def test_terminal_replay_and_strict_schema_rejection(self):
        for variant, dim in (("LAG2", 71), ("LAG3", 77)):
            contract = contract_for_variant_v01(variant)
            next_state, next_mask, next_action_mask = terminal_next_v01(variant)
            transition = RepresentationReplayTransitionV01(
                variant=variant,
                state_schema_id=contract.state_schema_id,
                q_input_schema_id=contract.q_input_schema_id,
                episode_id="episode",
                step_index=0,
                state=np.zeros(dim, dtype=np.float32),
                state_valid_mask=np.ones(dim, dtype=bool),
                action_id=0,
                reward=0.0,
                next_state=next_state,
                next_state_valid_mask=next_mask,
                done=True,
                action_mask=np.ones(3, dtype=bool),
                next_action_mask=next_action_mask,
            )
            validate_transition_v01(transition, expected_variant=variant)
            self.assertEqual(next_state.shape, (dim,))
            self.assertFalse(next_mask.any())
            wrong = "LAG3" if variant == "LAG2" else "LAG2"
            with self.assertRaisesRegex(ValueError, "variant mismatch"):
                RepresentationReplayBufferV01(capacity=2, variant=wrong).add(transition)
            migrated = RepresentationReplayTransitionV01(
                **{**transition.__dict__, "state_schema_id": "state_vector_v02_cooldown_v01"}
            )
            with self.assertRaisesRegex(ValueError, "state schema mismatch"):
                validate_transition_v01(migrated, expected_variant=variant)

    def test_close_reentry_window_drop_direction_and_q_reconstruction(self):
        ts = lambda seconds: pd.Timestamp("2026-06-24 09:00:00", tz="Asia/Tokyo") + pd.Timedelta(seconds=seconds)
        close_group = {"available": True, "position_before": "Long", "effective_action": "Sell"}
        flat_hold = {"available": True, "position_before": "Flat", "effective_action": "Hold"}
        unavailable = {"available": False, "position_before": None, "effective_action": None}
        decisions = [
            {"timestamp": ts(0), "session": "morning", "position_before": "Long", "position_after": "Flat", "selected_action": "Sell", "execution_failed": False, "forced_exit": False},
            {"timestamp": ts(5), "session": "morning", "position_before": "Flat", "position_after": "Flat", "selected_action": "Hold", "history_groups": [close_group, unavailable], "q_hold": 1.0, "q_buy": 0.5, "q_sell": 0.0},
            {"timestamp": ts(10), "session": "morning", "position_before": "Flat", "position_after": "Flat", "selected_action": "Hold", "history_groups": [flat_hold, close_group], "q_hold": 0.9, "q_buy": 0.8, "q_sell": 0.1},
            {"timestamp": ts(15), "session": "morning", "position_before": "Flat", "position_after": "Long", "selected_action": "Buy", "history_groups": [flat_hold, flat_hold], "q_hold": 0.2, "q_buy": 1.2, "q_sell": -0.1},
        ]
        result = reconstruct_close_reentry_diagnostics_v01(decisions, history_depth=2)
        self.assertEqual(result["expected_first_decision_without_close_context_seconds"], 15)
        self.assertEqual([row["elapsed_bucket"] for row in result["rows"]], ["<=5s", "<=10s", "<=15s"])
        self.assertEqual([row["original_close_context_present"] for row in result["rows"]], [True, True, False])
        self.assertTrue(result["rows"][-1]["first_decision_without_original_close_context"])
        self.assertEqual(result["rows"][-1]["reentry_direction"], "same_side_reentry")
        self.assertAlmostEqual(result["rows"][-1]["best_q"], 1.2)
        self.assertAlmostEqual(result["rows"][-1]["q_margin"], 1.0)

        reversal = reconstruct_close_reentry_diagnostics_v01([
            {"timestamp": ts(0), "session": "morning", "position_before": "Short", "position_after": "Flat", "selected_action": "Buy"},
            {"timestamp": ts(5), "session": "morning", "position_before": "Flat", "position_after": "Long", "selected_action": "Buy", "history_groups": [
                {"available": True, "position_before": "Short", "effective_action": "Buy"}, unavailable, unavailable
            ]},
        ], history_depth=3)
        self.assertEqual(reversal["rows"][0]["reentry_direction"], "reversal")
        self.assertEqual(reversal["expected_first_decision_without_close_context_seconds"], 20)


if __name__ == "__main__":
    unittest.main()
