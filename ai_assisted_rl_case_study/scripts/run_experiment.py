#!/usr/bin/env python3
"""非公開datasetを指定してI12 DDQNの学習経路を起動する。性能再評価ではない。"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

def load_config(path):
    config = json.loads(Path(path).read_text())
    canonical = json.dumps(config, sort_keys=True, separators=(',', ':')).encode()
    if hashlib.sha256(canonical).hexdigest() != '75daf29cadc2ce26684ba1b9ff58ad057c6e13095d40ae6dddddd7bd1528216c':
        raise ValueError('この入口では公開固定configと異なるsemanticを許可しません')
    return config

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/i12_explicit_history_experiment.json')
    parser.add_argument('--dataset-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    variant = parser.add_mutually_exclusive_group(required=True)
    variant.add_argument('--variant', choices=['B0', 'L1', 'L2', 'L3'])
    variant.add_argument('--history-depth', type=int, choices=range(4))
    parser.add_argument('--seed', type=int, default=20260815)
    parser.add_argument('--smoke', action='store_true', help='PUBLIC_BUILD_E0_1_SMOKE、50更新で停止')
    parser.add_argument('--trusted-dataset', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    name = args.variant or ['B0', 'L1', 'L2', 'L3'][args.history_depth]
    if args.seed not in config['seeds'] or (args.smoke and args.seed != 20260815):
        raise ValueError('seedが固定条件と異なります')
    if args.output_root.exists():
        raise ValueError('出力先は新しいdirectoryを指定してください')
    args.output_root.mkdir(parents=True)
    os.environ.setdefault('MPLCONFIGDIR', str(args.output_root / '.mpl'))
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    from training.public_dataset import PublicDataset
    dataset = PublicDataset(args.dataset_root, config, trusted=args.trusted_dataset)
    import numpy as np
    import tensorflow as tf
    devices = tf.config.list_physical_devices('GPU')
    if len(devices) != 1:
        raise RuntimeError('GPUが一意に利用可能ではありません。CPU fallbackは禁止です')
    details = tf.config.experimental.get_device_details(devices[0])
    gpu_name = details.get('device_name', '')
    if 'RTX 5070 Ti' not in gpu_name:
        raise RuntimeError('承認されたGPUモデルではありません')
    tf.config.experimental.set_memory_growth(devices[0], True)
    tf.config.set_soft_device_placement(False)
    from training.rl_representation_e0_runner_v01 import _E0LearnerV01, _run_training_episode_v01
    from training.rl_representation_observability_v01 import E0ObservabilityRecorderV01
    from training.epsilon_scheduler_v01 import compute_epsilon_v01
    stage = 'PUBLIC_BUILD_E0_1_SMOKE' if args.smoke else 'PUBLIC_BUILD_TRAINING'
    cap = 50 if args.smoke else config['optimizer_updates']
    internal_variant = 'B0' if name == 'B0' else 'LAG' + name[-1]
    recorder = E0ObservabilityRecorderV01(run_identity={
        'run_id': f'{stage}_{name}_{args.seed}', 'phase': stage, 'variant': internal_variant,
        'gamma': config['gamma'], 'backup_algorithm': 'double_dqn', 'rl_seed': args.seed,
    })
    learner = _E0LearnerV01(variant=internal_variant, seed=args.seed, gamma=config['gamma'],
                           recorder=recorder, backup_algorithm='double_dqn', mechanism_telemetry_enabled=True)
    learner.target_action_mask_source = config['target_action_mask_source']
    model = config['variants'][name]
    assert learner.q_network.count_params() == model['parameters']
    assert learner.contract.state_dim == model['state_dimension']
    assert learner.contract.q_input_dim == model['q_input_dimension']
    started = time.monotonic()
    episodes = 0
    for item in dataset.episodes:
        if args.smoke and learner.optimizer_updates >= cap:
            break
        artifact = dataset.load(item)
        _run_training_episode_v01(learner=learner, artifact=artifact, date=item['date'],
                                 symbol=item['symbol'], run_phase=stage)
        episodes += 1
        burst = 0
        while learner.optimizer_updates < cap and learner.pending_train_credit > 0 and len(learner.replay) >= 1024 and burst < 256:
            epsilon = compute_epsilon_v01(learner.global_env_step, 1.0, 0.1, 20000)
            with tf.device('/GPU:0'):
                learner.train_one(epsilon=epsilon)
            if 'GPU:0' not in learner.actual_device:
                raise RuntimeError('actual training tensorがGPU上にありません。停止します')
            burst += 1
        recorder.write_json(args.output_root / 'observability.json')
        print(json.dumps({'variant': name, 'episodes': episodes, 'updates': learner.optimizer_updates}), flush=True)
    if learner.optimizer_updates != cap or int(learner.optimizer.iterations.numpy()) != cap:
        raise RuntimeError('必要更新数に未到達。自動retryはしません')
    if learner.admission.stats.training_invalid_episode_count or recorder.health['fatal_finite_gate_count']:
        raise RuntimeError('training health gate failed')
    assert learner.target_update_count == 1 + cap // 200
    assert learner.last_target_sync_update == (cap // 200) * 200
    if not np.isfinite(np.asarray([v for w in learner.q_network.get_weights() for v in w.ravel()])).all():
        raise RuntimeError('nonfinite model weights')
    result = {
        'stage': stage, 'variant': name, 'seed': args.seed, 'updates': learner.optimizer_updates,
        'optimizer_iteration': int(learner.optimizer.iterations.numpy()), 'parameter_count': learner.q_network.count_params(),
        'state_dimension': learner.contract.state_dim, 'q_input_dimension': learner.contract.q_input_dim,
        'actual_device': 'GPU:0', 'gpu_model': gpu_name, 'tensorflow_version': tf.__version__,
        'cpu_fallback': False, 'finite_gates': 'PASS', 'target_sync_logic': 'PASS',
        'target_sync_count_including_initial': learner.target_update_count, 'last_target_sync_update': learner.last_target_sync_update,
        'replay_size': len(learner.replay), 'replay_committed': learner.admission.stats.replay_committed_count,
        'episodes': episodes, 'elapsed_seconds': time.monotonic() - started,
        'dataset_identity_sha256': dataset.identity, 'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
        'status': 'PASS', 'performance_evaluation': False,
    }
    recorder.write_json(args.output_root / 'observability.json')
    (args.output_root / 'run_health.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    from training.public_observability import persist_final_observability
    persist_final_observability(json.loads((args.output_root / 'observability.json').read_text()), result, args.output_root)
    print(json.dumps(result, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
