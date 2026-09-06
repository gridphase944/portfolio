"""公開graph writer。実行時のsource dataと軸・seriesを保存し、空画像を拒否する。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_ROOT = None

def save_figure(figure, path, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    axes = []
    for ax in figure.axes:
        series = []
        for line in ax.lines:
            xy = np.asarray(line.get_xydata(), dtype=float)
            if not len(xy) or not np.isfinite(xy).all():
                raise ValueError('empty/nonfinite plotted line')
            series.append({'kind': 'line', 'label': line.get_label(), 'xy': xy.tolist()})
        for container in ax.containers:
            if hasattr(container, 'datavalues'):
                values = np.asarray(container.datavalues, dtype=float)
                if not len(values) or not np.isfinite(values).all():
                    raise ValueError('empty/nonfinite plotted bars')
                series.append({'kind': 'bar', 'label': container.get_label(), 'values': values.tolist()})
        if not series:
            raise ValueError('axisにdata seriesがありません')
        axes.append({'xlabel': ax.get_xlabel(), 'ylabel': ax.get_ylabel(),
                     'title': ax.get_title(), 'xscale': ax.get_xscale(), 'yscale': ax.get_yscale(), 'series': series})
    if not axes:
        raise ValueError('figureにaxisがありません')
    figure.savefig(path, **kwargs)
    image = plt.imread(path)
    if image.size == 0 or float(image.std()) < .001:
        raise ValueError('empty image')
    path.with_suffix('.plot.json').write_text(json.dumps({'axes': axes, 'png_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2) + '\n')

def save_graph(manifest, *, graph_id, figure, source, purpose, axis, unit, series, denominator, limitations, seed_checkpoint='3 seeds / update_2000'):
    if OUTPUT_ROOT is None or source.empty:
        raise ValueError('output root未指定、またはsourceが空です')
    root = Path(OUTPUT_ROOT)
    png = root / 'graphs' / (graph_id + '.png')
    data = root / 'graph_source_data' / (graph_id + '.csv')
    data.parent.mkdir(parents=True, exist_ok=True)
    source.to_csv(data, index=False, lineterminator='\n')
    figure.tight_layout()
    save_figure(figure, png, dpi=160)
    plt.close(figure)
    manifest.append({'graph_id': graph_id, 'purpose': purpose, 'axis': axis,
                     'unit': unit, 'series': list(series), 'denominator': denominator,
                     'source_rows': len(source), 'source_path': str(data.relative_to(root)),
                     'source_sha256': hashlib.sha256(data.read_bytes()).hexdigest(),
                     'png_path': str(png.relative_to(root)), 'png_sha256': hashlib.sha256(png.read_bytes()).hexdigest(),
                     'limitations': limitations, 'seed_checkpoint': seed_checkpoint,
                     'smoothing': 'non-overlapping 50-update arithmetic mean' if graph_id in ['G06', 'G07'] else 'none'})

def smoke_graphs(observability, run_health, output):
    global OUTPUT_ROOT
    OUTPUT_ROOT = Path(output)
    rows = observability['learning_updates']
    if len(rows) != run_health['updates']:
        raise ValueError('learning row/update count mismatch')
    frame = pd.DataFrame([{
        'update': r['optimizer_update_index'], 'loss': r['loss'],
        'td_abs_mean': r['td_error']['abs_mean'], 'epsilon': r['epsilon'],
        'q_hold': r['q_by_action']['Hold']['mean'], 'q_buy': r['q_by_action']['Buy']['mean'],
        'q_sell': r['q_by_action']['Sell']['mean'], 'q_margin': r['q_margin']['mean'],
        'replay_size': r['replay_size'], 'gradient_norm': r['gradient_global_norm_before_clip'],
    } for r in rows])
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError('nonfinite smoke graph source')
    manifest = []
    for graph_id, columns, ylabel in [
        ('S01', ['loss', 'td_abs_mean'], 'Loss / absolute TD error'),
        ('S02', ['epsilon'], 'Epsilon'),
        ('S03', ['q_hold', 'q_buy', 'q_sell', 'q_margin'], 'Q / Q margin'),
        ('S04', ['replay_size'], 'Replay transitions'),
        ('S05', ['gradient_norm'], 'Gradient global norm'),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for col in columns:
            ax.plot(frame['update'], frame[col], label=col)
        ax.set_xlabel('Optimizer update'); ax.set_ylabel(ylabel); ax.legend()
        save_graph(manifest, graph_id=graph_id, figure=fig, source=frame[['update'] + columns],
                   purpose='PUBLIC_BUILD_E0_1_SMOKE health only', axis={'x': 'optimizer update', 'y': ylabel},
                   unit='health units', series=columns, denominator='128 replay samples/update',
                   limitations='50-update implementation health; no performance comparison', seed_checkpoint='seed 20260815 / 50 updates')
    counts = {a: sum(e['action_counts'].get(a, 0) for e in observability['episodes']) for a in ['Hold', 'Buy', 'Sell']}
    total = sum(counts.values())
    actions = pd.DataFrame([{'action': a, 'count': count, 'rate': count / total} for a, count in counts.items()])
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.bar(actions.action, actions.rate)
    ax.set_xlabel('Action'); ax.set_ylabel('Policy action fraction'); ax.set_ylim(0, 1)
    save_graph(manifest, graph_id='S06', figure=fig, source=actions, purpose='Smoke policy action observation',
               axis={'x': 'action', 'y': 'fraction'}, unit='fraction', series=list(counts), denominator=str(total) + ' observed policy actions',
               limitations='Epsilon-greedy smoke; not historical performance', seed_checkpoint='seed 20260815 / 50 updates')
    (OUTPUT_ROOT / 'graph_manifest.json').write_text(json.dumps({'status': 'PASS', 'graphs': manifest, 'runtime_health': run_health}, indent=2) + '\n')
    return manifest
