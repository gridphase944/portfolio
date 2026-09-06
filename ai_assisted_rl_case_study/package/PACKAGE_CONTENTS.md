# このpackageについて

完了済みの「DDQN Lag-1〜3 explicit agent history E0」を、Technical Experiment Report・Human-readable Experiment Report・必要なsourceとともに読むためのpackageです。

## 最初に読むもの

- [Human-readable Experiment Report](../docs/human_readable_experiment_report.html)：画像を内包しているため、単体で開けます。
- [Technical Experiment Report](../docs/technical_experiment_report.md)：仮説、条件、数値、判断、限界を記録しています。
- [実装確認の結果](SMOKE_TEST_REPORT.md)：今回の公開build確認であり、historical実験の性能再評価ではありません。

この比較はモデル全体のend-to-end比較です。履歴を増やすと入力次元・parameter数も増え、L3はB0比約14%増です。history単独の因果効果は分離していません。使用済みdevelopment data上のE0結果であり、fresh OOS、本番利用、利益モデル成立、E1昇格を示しません。

## 構成

以下のpathと実行commandはpackage rootを基準とします。

- `docs/`：[I12 methodology](../docs/i12_experiment_methodology.md)、Technical Experiment Report、Human-readable Experiment Report、公開プロセス仕様3冊。
- `docs/ai_assisted_development/`：[ChatGPT Project Instructions](../docs/ai_assisted_development/chatgpt_project_instructions.md)と[Codex Repository Instructions](../docs/ai_assisted_development/codex_repository_instructions.md)。役割分担・Context / Harness / Loop運用を示す公開用copyであり、portfolio repositoryのactive agent instructionではありません。
- [Dockerfile](../Dockerfile)：NVIDIA NGC base imageを用いた公開runtime / focused tests環境。
- [.dockerignore](../.dockerignore)：dataset・生成artifact・local secret等をbuild contextから除外。
- [.gitignore](../.gitignore)：dataset・生成artifact・local secret等をGit管理から除外。`outputs/`は実行時生成先であり、空directoryは同梱しません。
- `config/i12_explicit_history_experiment.json`：historical主要条件をまとめた固定config。
- `scripts/run_experiment.py`：config・dataset root・variantまたはhistory depth・seed・output rootを指定する入口。
- `src/features/`・`src/trading_env/`：状態、history、約定、rewardの必要な実装。
- `src/training/`：Q input、network、DDQN、Replay、更新、観測、graph生成。
- `tests/`：history、DDQN target、公開入口・保存処理のfocused tests。
- `report/figures/`：必要な24 PNG。原本の画像内容を保持しています。
- `package/SOURCE_PROVENANCE.json`：公開sourceの出典の記録。

実データ、集計CSV、学習済みweights、実行時checkpoint、Replay dump、private ML/RL完全仕様、内部実行記録は含めていません。

## 実行環境

検証環境はPython3.12.3、TensorFlow2.17.0、NumPy1.26.4、pandas2.2.2です。GPU smokeではRTX5070 Tiを使用しました。環境のインストール・driver調整はこのpackageの自動処理に含めません。

`requirements.txt`は必要なPython packageの一覧です。GPUを要求する実行でCPUへfallbackすることはありません。Envやdata pipelineにはCPU処理が含まれ、全処理をGPU化したという意味ではありません。

Dockerfileは`nvcr.io/nvidia/tensorflow:25.01-tf2-py3`を使用し、TensorFlowはNGC内蔵版を保持します。requirementsからTensorFlowを除外して他のPython dependencyを導入し、公開`src/`・`scripts/`・`config/`・`tests/`だけを配置します。NodeSource・Node.js・npm・Codex CLIおよび不要なCodex/sandbox用dependencyの追加install処理はありません。datasetはimageに含めず、利用者が外部path / read-only mount等で与える前提です。

ユーザー手動WSL環境でのDocker build / GPU検証と、container testsの43 PASS / 1件の数値非決定性、およびblockerとしないPM判断は[実装確認の結果](SMOKE_TEST_REPORT.md)に出典を分けて記録しています。

## 非公開datasetの入力契約

入口が受け取るのは、**前処理済み市場feature、共通readiness、1秒latency／closing-risk約定sidecar**を持つdatasetです。vendor形式のraw CSVだけから全前処理を行う汎用loaderではありません。既存の市場feature・readiness・約定の意味を変えずにtraining部分を切り出すため、この境界を明示しています。

dataset rootに`dataset_manifest.json`を置きます。

```json
{
  "schema": "i12_prepared_market_dataset_v1",
  "episodes": [
    {
      "date": "2026-06-24",
      "symbol": "DATASET_SYMBOL",
      "role": "train",
      "episode": "episode.pkl",
      "readiness": "readiness.npz",
      "readiness_reasons": "readiness_reasons.json",
      "execution_sidecar": "execution",
      "sha256": {
        "episode.pkl": "REPLACE_WITH_SHA256",
        "readiness.npz": "REPLACE_WITH_SHA256",
        "readiness_reasons.json": "REPLACE_WITH_SHA256",
        "execution/manifest.json": "REPLACE_WITH_SHA256",
        "execution/sidecar.pkl": "REPLACE_WITH_SHA256"
      }
    }
  ]
}
```

episodeは5秒decision gridのDataFrameで、state builderが使うmarket feature・timestamp・session・板・前日終値を持ちます。agentのposition/action/historyはEnv内で生成し、将来のactionを入力へ入れません。readiness NPZには既存契約の`ready`と`embeddings`、理由JSONにはrowごとの理由配列を保持します。I12ではembedding値をQ inputへ接続しません。共通readinessを別の条件に置き換えてはいけません。

sidecarは公開された`execution_closingrisk_v02.py`のschema／identity／row alignment検証に通るものを用意します。manifestのepisode順をそのまま使い、許可されたtrain日付以外を拒否します。必要columnと検証条件は同梱のstate builder・sidecar validatorを正本とします。

pickleは信頼できない入力で任意コード実行の危険があります。作成者・由来・hashを確認した自分のdatasetだけを、`--trusted-dataset`で明示的に許可してください。hash一致だけでは第三者のpickleを安全にしません。datasetは公開・Git管理しないでください。

```bash
python scripts/run_experiment.py --help
python scripts/run_experiment.py --config config/i12_explicit_history_experiment.json --dataset-root PRIVATE_PREPARED_DATASET --variant L3 --seed 20260815 --output-root NEW_RUN_DIRECTORY --smoke --trusted-dataset
```

`--history-depth 0/1/2/3`も指定できます。`--smoke`は50更新に制限します。上記の大文字pathは利用者が置き換えるplaceholderです。

configはsemantic固定です。異なる値をsilentに受け入れず拒否します。通常経路は同じ更新則で最大2,000更新を行い、manifest内の全train episodeを処理します。historical runは192 train episodesでした。今回実行したのはtrain-roleの1 episode・各50更新だけです。通常の2,000更新経路を今回再実行したという意味ではありません。

## Testsとgraph

focused testsは`python -m pytest -q tests`で実行します。実行時の一時fileを公開物へ混入させないよう、`TMPDIR`、pytestの`--basetemp`、cache先を検証用directoryへ指定してください。

```bash
python scripts/generate_graphs.py --help
python scripts/generate_graphs.py smoke --source-root PRIVATE_SMOKE_OUTPUT --output-root NEW_GRAPH_DIRECTORY
python scripts/generate_graphs.py historical --source-root PRIVATE_COMPACT_DATA --phase1-reference-csv PRIVATE_PHASE1_REFERENCE --output-root NEW_GRAPH_DIRECTORY
```

historical modeは非公開の集計CSVを別途必要とします。G01〜G22と技術レポート参照のPhase-I 2図を対象とし、分母・軸・seriesを保持します。source mappingは`GRAPH_VALIDATION.json`、描画data・axis確認は実行時のgraph manifestとplot metadataに保存します。集計CSVが含まれないため、ZIPだけからhistorical graphを再計算できるとは主張しません。

## 再現性の範囲

historical sourceの必要部分を用いた公開training経路であり、全baselineのsource freeze・dataset・重み・software imageを含むexact historical rerunではありません。source subsetのsemantic bodyと今回のfocused tests／GPU smoke／graph検証を区別して確認してください。詳細な未使用判定ができないclass methodやregistry経由の参照は、推測で削除せず保持しています。

公開copy以外のsourceやGit履歴は変更していません。公開repositoryでのcommit・pushは本package作成作業の対象外です。
