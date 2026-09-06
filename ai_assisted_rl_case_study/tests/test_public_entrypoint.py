from pathlib import Path
import importlib.util
import json
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
spec = importlib.util.spec_from_file_location('public_entrypoint', ROOT / 'scripts/run_experiment.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)

def test_config_fixed_historical_identity():
    c = entry.load_config(ROOT / 'config/i12_explicit_history_experiment.json')
    assert c['gamma'] == .9962 and c['target_sync_updates'] == 200
    assert [v['parameters'] for v in c['variants'].values()] == [32131, 33667, 35203, 36739]

def test_config_rejects_silent_semantic_override(tmp_path):
    c = json.loads((ROOT / 'config/i12_explicit_history_experiment.json').read_text())
    c['gamma'] = .99
    p = tmp_path / 'changed.json'
    p.write_text(json.dumps(c))
    with pytest.raises(ValueError): entry.load_config(p)

def test_dataset_requires_explicit_trust(tmp_path):
    from training.public_dataset import PublicDataset
    with pytest.raises(ValueError, match='trusted-dataset'):
        PublicDataset(tmp_path, {}, trusted=False)

def test_dataset_rejects_nontrain_date(tmp_path):
    from training.public_dataset import PublicDataset
    p = tmp_path / 'dataset_manifest.json'
    p.write_text(json.dumps({'schema': 'i12_prepared_market_dataset_v1', 'episodes': [
        {'date': '2026-06-30', 'symbol': 'fixture', 'role': 'train', 'sha256': {}}
    ]}))
    with pytest.raises(ValueError, match='train-role'):
        PublicDataset(tmp_path, {'train_dates': ['2026-06-24']}, trusted=True)

def test_public_imports_have_no_private_fallback():
    import training.rl_representation_e0_runner_v01
    import trading_env.env_v01
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] in {'training', 'trading_env', 'features'} and getattr(module, '__file__', None):
            assert Path(module.__file__).resolve().is_relative_to(ROOT / 'src'), name

def test_final_writer_uses_measured_runtime_not_default_counters(tmp_path):
    from training.public_observability import persist_final_observability
    payload = {'learning_updates': [{}], 'episodes': [{'env_step_count': 7}],
               'health': {'fatal_finite_gate_count': 0, 'nonfinite_count': 0, 'replay_committed_count': 0}}
    health = {'updates': 1, 'replay_committed': 1234, 'replay_size': 1234,
              'target_sync_count_including_initial': 1, 'finite_gates': 'PASS',
              'actual_device': 'GPU:0', 'cpu_fallback': False, 'gpu_model': 'fixture', 'elapsed_seconds': 1.0}
    persist_final_observability(payload, health, tmp_path)
    saved = json.loads((tmp_path / 'observability.json').read_text())
    assert saved['health']['replay_committed_count'] == 1234
    assert saved['runtime']['actual_training_tensor_device'] == 'GPU:0'
    assert 'replay_staged_count' not in saved['health']
