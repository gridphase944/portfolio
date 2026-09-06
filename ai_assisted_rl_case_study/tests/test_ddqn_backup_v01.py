from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.ddqn_backup_v01 import (
    compute_next_backup_comparison_v01,
    validate_backup_algorithm_v01,
)
from training.replay_buffer_v01 import encode_action_v01
from training.rl_representation_e0_runner_v01 import (
    _E0LearnerV01,
    _network_weight_hash_v01,
    _save_checkpoint_v01,
)
from training.rl_representation_observability_v01 import (
    E0ObservabilityRecorderV01,
)
from training.rl_representation_replay_v01 import (
    RepresentationReplayTransitionV01,
)


class DoubleDQNBackupContractTest(unittest.TestCase):
    def test_versioned_identity_rejects_silent_fallback(self) -> None:
        self.assertEqual(validate_backup_algorithm_v01("dqn"), "dqn")
        self.assertEqual(
            validate_backup_algorithm_v01("double_dqn"), "double_dqn"
        )
        with self.assertRaisesRegex(ValueError, "backup_algorithm"):
            validate_backup_algorithm_v01("ddqn")

    def test_dqn_and_ddqn_selection_evaluation_delta(self) -> None:
        result_dqn = compute_next_backup_comparison_v01(
            online_next_q_values=[[1.0, 2.0, 5.0]],
            target_next_q_values=[[1.0, 4.0, 3.0]],
            dones=[False],
            next_action_masks=[[True, True, True]],
            backup_algorithm="dqn",
        )
        result_ddqn = compute_next_backup_comparison_v01(
            online_next_q_values=[[1.0, 2.0, 5.0]],
            target_next_q_values=[[1.0, 4.0, 3.0]],
            dones=[False],
            next_action_masks=[[True, True, True]],
            backup_algorithm="double_dqn",
        )
        self.assertEqual(int(result_dqn.dqn_action_ids.numpy()[0]), 1)
        self.assertEqual(int(result_ddqn.ddqn_action_ids.numpy()[0]), 2)
        self.assertEqual(float(result_dqn.selected_bootstrap.numpy()[0]), 4.0)
        self.assertEqual(float(result_ddqn.selected_bootstrap.numpy()[0]), 3.0)
        self.assertEqual(float(result_ddqn.backup_gap.numpy()[0]), 1.0)

    def test_online_equals_target_makes_targets_identical(self) -> None:
        q = [[0.5, 2.0, -1.0], [3.0, 1.0, 2.0]]
        masks = [[True, True, False], [True, False, True]]
        dqn = compute_next_backup_comparison_v01(
            online_next_q_values=q,
            target_next_q_values=q,
            dones=[False, False],
            next_action_masks=masks,
            backup_algorithm="dqn",
        )
        ddqn = compute_next_backup_comparison_v01(
            online_next_q_values=q,
            target_next_q_values=q,
            dones=[False, False],
            next_action_masks=masks,
            backup_algorithm="double_dqn",
        )
        np.testing.assert_array_equal(
            dqn.dqn_action_ids.numpy(), ddqn.ddqn_action_ids.numpy()
        )
        np.testing.assert_array_equal(
            dqn.selected_bootstrap.numpy(), ddqn.selected_bootstrap.numpy()
        )

    def test_same_mask_excludes_invalid_raw_max(self) -> None:
        result = compute_next_backup_comparison_v01(
            online_next_q_values=[[2.0, 1.0, 999.0]],
            target_next_q_values=[[3.0, 4.0, 888.0]],
            dones=[False],
            next_action_masks=[[True, True, False]],
            backup_algorithm="double_dqn",
        )
        self.assertEqual(int(result.dqn_action_ids.numpy()[0]), 1)
        self.assertEqual(int(result.ddqn_action_ids.numpy()[0]), 0)
        self.assertEqual(float(result.selected_bootstrap.numpy()[0]), 3.0)

    def test_terminal_all_false_is_exact_zero_and_placeholder_independent(self) -> None:
        first = compute_next_backup_comparison_v01(
            online_next_q_values=[[1.0, 2.0, 3.0]],
            target_next_q_values=[[4.0, 5.0, 6.0]],
            dones=[True],
            next_action_masks=[[False, False, False]],
            backup_algorithm="double_dqn",
        )
        second = compute_next_backup_comparison_v01(
            online_next_q_values=[[-30.0, 200.0, -10.0]],
            target_next_q_values=[[700.0, -500.0, 60.0]],
            dones=[True],
            next_action_masks=[[False, False, False]],
            backup_algorithm="double_dqn",
        )
        self.assertEqual(float(first.selected_bootstrap.numpy()[0]), 0.0)
        self.assertEqual(float(second.selected_bootstrap.numpy()[0]), 0.0)
        self.assertEqual(float(first.backup_gap.numpy()[0]), 0.0)
        self.assertEqual(float(second.backup_gap.numpy()[0]), 0.0)

    def test_nonterminal_all_false_is_rejected(self) -> None:
        import tensorflow as tf

        with self.assertRaises(tf.errors.InvalidArgumentError):
            compute_next_backup_comparison_v01(
                online_next_q_values=[[1.0, 2.0, 3.0]],
                target_next_q_values=[[4.0, 5.0, 6.0]],
                dones=[False],
                next_action_masks=[[False, False, False]],
                backup_algorithm="double_dqn",
            )

    def test_tie_is_deterministic_first_valid_action(self) -> None:
        result = compute_next_backup_comparison_v01(
            online_next_q_values=[[5.0, 5.0, 0.0]],
            target_next_q_values=[[7.0, 7.0, 1.0]],
            dones=[False],
            next_action_masks=[[True, True, True]],
            backup_algorithm="double_dqn",
        )
        self.assertEqual(int(result.dqn_action_ids.numpy()[0]), 0)
        self.assertEqual(int(result.ddqn_action_ids.numpy()[0]), 0)

    def test_next_q_paths_are_detached_from_gradient(self) -> None:
        import tensorflow as tf

        online = tf.Variable([[1.0, 2.0, 3.0]])
        target = tf.Variable([[3.0, 2.0, 1.0]])
        with tf.GradientTape() as tape:
            result = compute_next_backup_comparison_v01(
                online_next_q_values=online,
                target_next_q_values=target,
                dones=[False],
                next_action_masks=[[True, True, True]],
                backup_algorithm="double_dqn",
            )
            objective = tf.reduce_sum(result.selected_bootstrap)
        gradients = tape.gradient(objective, [online, target])
        self.assertEqual(gradients, [None, None])

    def test_nonfinite_next_q_is_rejected(self) -> None:
        import tensorflow as tf

        with self.assertRaises(tf.errors.InvalidArgumentError):
            compute_next_backup_comparison_v01(
                online_next_q_values=[[1.0, np.nan, 3.0]],
                target_next_q_values=[[1.0, 2.0, 3.0]],
                dones=[False],
                next_action_masks=[[True, True, True]],
                backup_algorithm="double_dqn",
            )

    def test_action_order_regression(self) -> None:
        self.assertEqual(
            [encode_action_v01(name) for name in ("Hold", "Buy", "Sell")],
            [0, 1, 2],
        )


class DoubleDQNLearnerIntegrationTest(unittest.TestCase):
    def _learner(
        self, *, telemetry: bool, backup_algorithm: str = "dqn"
    ) -> _E0LearnerV01:
        recorder = E0ObservabilityRecorderV01(
            run_identity={
                "run_id": f"test-{backup_algorithm}-{telemetry}",
                "gamma": 0.9962,
                "backup_algorithm": backup_algorithm,
            }
        )
        learner = _E0LearnerV01(
            variant="B0",
            seed=20260815,
            gamma=0.9962,
            recorder=recorder,
            backup_algorithm=backup_algorithm,
            mechanism_telemetry_enabled=telemetry,
        )
        state = np.full(59, 0.01, dtype=np.float32)
        next_state = np.full(59, 0.02, dtype=np.float32)
        mask = np.ones(59, dtype=bool)
        for index in range(1024):
            learner.replay.add(
                RepresentationReplayTransitionV01(
                    variant="B0",
                    state_schema_id=learner.contract.state_schema_id,
                    q_input_schema_id=learner.contract.q_input_schema_id,
                    episode_id=f"fixture-{index // 32}",
                    step_index=index % 32,
                    state=state,
                    state_valid_mask=mask,
                    action_id=index % 3,
                    reward=0.01,
                    next_state=next_state,
                    next_state_valid_mask=mask,
                    done=False,
                    action_mask=np.ones(3, dtype=bool),
                    next_action_mask=np.ones(3, dtype=bool),
                )
            )
        learner.admission.stats.replay_committed_count = 1024
        return learner

    def test_h0_telemetry_on_off_learning_parity_and_online_only_update(self) -> None:
        off = self._learner(telemetry=False)
        on = self._learner(telemetry=True)
        np.testing.assert_array_equal(
            off.q_network.get_weights()[0], on.q_network.get_weights()[0]
        )
        off_target_before = _network_weight_hash_v01(off.target_network)
        on_target_before = _network_weight_hash_v01(on.target_network)
        off_result = off.train_one(epsilon=1.0)
        on_result = on.train_one(epsilon=1.0)
        self.assertEqual(off_result["loss"], on_result["loss"])
        for left, right in zip(
            off.q_network.get_weights(), on.q_network.get_weights(), strict=True
        ):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(
            _network_weight_hash_v01(off.target_network), off_target_before
        )
        self.assertEqual(
            _network_weight_hash_v01(on.target_network), on_target_before
        )
        self.assertEqual(len(off.mechanism_telemetry_rows), 0)
        self.assertEqual(len(on.mechanism_telemetry_rows), 1)
        self.assertEqual(on.mechanism_telemetry_rows[0]["pre_update_index"], 0)

    def test_200_update_hard_sync_regression(self) -> None:
        learner = self._learner(telemetry=True, backup_algorithm="double_dqn")
        learner.optimizer_updates = 199
        learner.last_target_sync_update = 0
        result = learner.train_one(epsilon=1.0)
        self.assertTrue(result["target_updated"])
        self.assertEqual(learner.last_target_sync_update, 200)
        self.assertEqual(
            _network_weight_hash_v01(learner.q_network),
            _network_weight_hash_v01(learner.target_network),
        )

    def test_checkpoint_saves_online_target_and_sync_identity(self) -> None:
        learner = self._learner(telemetry=False, backup_algorithm="double_dqn")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _save_checkpoint_v01(
                learner.q_network,
                root,
                250,
                gamma=0.9962,
                target_network=learner.target_network,
                backup_algorithm="double_dqn",
                last_target_sync_update=200,
                metadata={
                    "run_id": "checkpoint-test",
                    "seed": 20260815,
                    "source_identity": "a" * 64,
                    "config_hash": "b" * 64,
                    "backup_steps": 1,
                    "target_sync_interval": 200,
                    "state_schema": learner.contract.state_schema_id,
                    "q_input_schema": learner.contract.q_input_schema_id,
                    "action_order": ["Hold", "Buy", "Sell"],
                },
            )
            checkpoint = root / "checkpoints/update_0250"
            receipt = json.loads(
                (checkpoint / "receipt.json").read_text(encoding="utf-8")
            )
            self.assertTrue((checkpoint / "q_weights.npz").is_file())
            self.assertTrue((checkpoint / "target_q_weights.npz").is_file())
            self.assertEqual(receipt["backup_algorithm"], "double_dqn")
            self.assertEqual(receipt["last_target_sync_update"], 200)
            self.assertEqual(receipt["sync_age_at_checkpoint"], 50)
            self.assertEqual(receipt["action_order"], ["Hold", "Buy", "Sell"])
            reloaded = self._learner(
                telemetry=False, backup_algorithm="double_dqn"
            )
            for filename, network in (
                ("q_weights.npz", reloaded.q_network),
                ("target_q_weights.npz", reloaded.target_network),
            ):
                with np.load(checkpoint / filename, allow_pickle=False) as payload:
                    keys = sorted(
                        payload.files,
                        key=lambda name: int(name.rsplit("_", 1)[1]),
                    )
                    network.set_weights([payload[key] for key in keys])
            self.assertEqual(
                _network_weight_hash_v01(reloaded.q_network),
                receipt["online_network_identity_sha256"],
            )
            self.assertEqual(
                _network_weight_hash_v01(reloaded.target_network),
                receipt["target_network_identity_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
