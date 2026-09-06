#!/usr/bin/env python3
"""非公開の観測sourceを指定し、smokeまたはhistorical report graphを生成する。"""
from pathlib import Path
import argparse
import json
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['smoke', 'historical'])
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--phase1-reference-csv', type=Path)
    args = parser.parse_args()
    if args.output_root.exists(): raise ValueError('出力先は新しいdirectoryを指定してください')
    args.output_root.mkdir(parents=True)
    os.environ.setdefault('MPLCONFIGDIR', str(args.output_root / '.mpl'))
    from training import public_graphs as writer
    writer.OUTPUT_ROOT = args.output_root
    if args.mode == 'smoke':
        obs = json.loads((args.source_root / 'observability.json').read_text())
        health = json.loads((args.source_root / 'run_health.json').read_text())
        result = writer.smoke_graphs(obs, health, args.output_root)
        print(json.dumps({'status': 'PASS', 'graph_count': len(result)}))
        return
    import pandas as pd
    from training import historical_graphs as renderer
    renderer.OUT = args.output_root
    def read(name): return pd.read_csv(args.source_root / ('lag23_' + name + '.csv'))
    learning = read('G06').merge(read('G07'), on=['variant', 'update_end'], validate='one_to_one')
    result = renderer.create_graphs(
        memory=read('memory_depth_comparison'), seed=read('seed_level_comparison'),
        checkpoint=read('checkpoint_comparison'), behavior=read('trading_behavior_comparison'),
        learning=learning, tail=read('tail_risk_comparison'), cdf=read('G11'),
        time_combined=read('G12'), window=read('window_drop_comparison'), history=read('history_health_comparison'))
    phase_count = 0
    if args.phase1_reference_csv:
        from training.phase1_graphs import plot_checkpoint_reward, plot_time_since_close
        output = args.output_root / 'phase1'
        output.mkdir()
        reward = pd.read_csv(args.source_root / 'diagnostic_reward_progression_by_checkpoint.csv', dtype={'seed': str})
        time = pd.read_csv(args.source_root / 'diagnostic_time_since_close_behavior_q.csv', dtype={'seed': str})
        reference = pd.read_csv(args.phase1_reference_csv, dtype={'seed': str})
        plot_checkpoint_reward(reward, output)
        plot_time_since_close(time, reference, output)
        phase_count = 2
    print(json.dumps({'status': 'PASS', 'graph_count': len(result['graphs']) + phase_count}))

if __name__ == '__main__': main()
