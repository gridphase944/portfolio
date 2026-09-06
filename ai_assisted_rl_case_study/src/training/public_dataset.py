"""明示的なmanifestから、信頼できる非公開の前処理済み市場datasetを読む。

市場feature、共通readiness、約定sidecarは入力契約の一部であり再定義しない。
pickleを扱うため、作成者とhashを確認したdatasetだけを利用すること。
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

from training.rl_representation_e0_runner_v01 import _EpisodeArtifactV01
from trading_env.execution_closingrisk_v02 import (
    prepare_closingrisk_decision_rows_v02,
    load_execution_closingrisk_sidecar_v02,
    attach_execution_closingrisk_sidecar_v02,
)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def contained(root, name):
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("dataset参照が指定rootの外側です")
    return path

class PublicDataset:
    def __init__(self, root, config, *, trusted=False):
        if not trusted:
            raise ValueError("信頼できる前処理済みdatasetのみ --trusted-dataset で指定してください")
        self.root = Path(root).resolve(strict=True)
        self.manifest = json.loads((self.root / 'dataset_manifest.json').read_text())
        if self.manifest.get('schema') != 'i12_prepared_market_dataset_v1':
            raise ValueError('dataset schemaが異なります')
        self.episodes = self.manifest['episodes']
        if not self.episodes or len({(x['date'], x['symbol']) for x in self.episodes}) != len(self.episodes):
            raise ValueError('episodeが空、または重複しています')
        for item in self.episodes:
            if item['role'] != 'train' or item['date'] not in config['train_dates']:
                raise ValueError('許可されたtrain-role日付ではありません')
            for name, expected in item['sha256'].items():
                if digest(contained(self.root, name)) != expected:
                    raise ValueError('dataset file hash不一致')
        self.identity = digest(self.root / 'dataset_manifest.json')

    def load(self, item):
        for field in ['episode', 'readiness', 'readiness_reasons']:
            if item[field] not in item['sha256']:
                raise ValueError('必要datasetのhashが未指定です')
        sidecar = contained(self.root, item['execution_sidecar'])
        for filename in ['manifest.json', 'sidecar.pkl']:
            name = str(Path(item['execution_sidecar']) / filename)
            if name not in item['sha256']:
                raise ValueError('execution sidecarのhashが未指定です')
        rows = pd.read_pickle(contained(self.root, item['episode']))
        rows = prepare_closingrisk_decision_rows_v02(rows)
        count = len(rows)
        with np.load(contained(self.root, item['readiness']), allow_pickle=False) as arrays:
            ready = np.asarray(arrays['ready'][:count], dtype=bool)
            embeddings = np.asarray(arrays['embeddings'][:count], dtype=np.float32)
        reasons = json.loads(contained(self.root, item['readiness_reasons']).read_text())[:count]
        if len(ready) != count or len(reasons) != count or embeddings.shape != (count, 96):
            raise ValueError('readiness/row alignment不一致')
        timestamps = pd.to_datetime(rows['grid_timestamp'])
        if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
            raise ValueError('decision timestamp順序不一致')
        if set(timestamps.dt.strftime('%Y-%m-%d')) != {item['date']}:
            raise ValueError('manifestと実データの日付不一致')
        execution, _ = load_execution_closingrisk_sidecar_v02(directory=sidecar, decision_rows=rows)
        attached = attach_execution_closingrisk_sidecar_v02(rows, execution)
        return _EpisodeArtifactV01(attached, ready, embeddings,
                                   tuple(tuple(x) for x in reasons), {'dataset_sha256': self.identity})
