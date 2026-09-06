"""公開wrapperの最終観測writer。未収集の累積counterを0として報告しない。"""
from pathlib import Path
import json

def persist_final_observability(payload, run_health, output_root):
    if len(payload['learning_updates']) != run_health['updates']:
        raise ValueError('観測行数とsuccessful update数が一致しません')
    old = payload['health']
    health = {key: old[key] for key in ['fatal_finite_gate_count', 'nonfinite_count']}
    health.update(
        optimizer_update_count=run_health['updates'],
        replay_committed_count=run_health['replay_committed'],
        replay_size=run_health['replay_size'],
        target_network_update_count=run_health['target_sync_count_including_initial'],
        finite_gates=run_health['finite_gates'],
    )
    payload['health'] = health
    payload['runtime'] = {
        'actual_training_tensor_device': run_health['actual_device'],
        'cpu_fallback': run_health['cpu_fallback'],
        'gpu_model': run_health['gpu_model'],
        'elapsed_seconds': run_health['elapsed_seconds'],
        'optimizer_updates': run_health['updates'],
        'env_step_count': sum(e['env_step_count'] for e in payload['episodes']),
        'scope': 'training tensors; Env/data pipeline includes CPU work',
    }
    payload['public_writer_note'] = '上記以外の未収集counterは省略。未観測を0へ補完しない。'
    (Path(output_root) / 'observability.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
