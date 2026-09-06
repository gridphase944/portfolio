# AI-assisted ML/RL Development Case Study

本プロジェクトは、**AI-assistedでML/RL研究開発をどのように統制・実行するか**を示すケーススタディです。2026年8月に実施したHistorical Experiment「DDQN Lag-1〜3 Explicit Agent History E0」を実例として、実験ガバナンス、報告、AIとの役割分担、技術的証拠を公開しています。

## 主要ドキュメント

1. [ML/RL Experiment Governance](docs/ml_rl_experiment_governance_spec.md)に基づき、実験段階、promotion gate、evidence boundary、失敗時の停止条件を管理しています。
2. 読者に応じて[Human-readable Experiment Report](docs/human_readable_experiment_report.html)と[Technical Experiment Report](docs/technical_experiment_report.md)を分けています。
3. User・ChatGPT・Codexの役割と権限を分け、成果物と判断根拠を追跡するAI-assisted developmentを行っています。

> 本プロジェクトはHistorical Snapshotです。掲載するprocess文書は2026年8月時点の運用を示す公開用snapshotであり、現在のモデル・agent挙動・推奨設定に対する最新ガイドではありません。

## 1. Experiment Governance

本ケーススタディでは、**実験をどの証拠で次段階へ進め、どの条件で停止するか**をExperiment Governanceとして定義しています。

[ML/RL Experiment Governance](docs/ml_rl_experiment_governance_spec.md)では、E0（実装・観測・限定的な仮説確認）、E1（固定条件での正式比較）、E2（採用候補の正式統合）を分離し、promotion gate、data role、metricと分母、artifact、failure/retry、human approvalを定義しています。

Historical DDQN実験ではL3に一部改善signalが見られましたが、seed間の一貫性、tail risk、absolute profitability、parameter増加との交絡を踏まえ、E1へpromotionしませんでした。詳細な結果と判断根拠は[Technical Experiment Report](docs/technical_experiment_report.md)に記載しています。

## 2. Experiment Reports

- [Human-readable Experiment Report](docs/human_readable_experiment_report.html)：ML経験が浅い読者の認知負荷を下げるため、視覚的な階層、図、用語、数式の意味を段階的に説明した自己完結型HTMLです。
- [Technical Experiment Report](docs/technical_experiment_report.md)：実験条件、metricと分母、数式、評価、evidence boundary、limitations、最終判断を詳細に残した技術レビュー・再検証用資料です。

実験設計と公開範囲の要約は[I12 Experiment Methodology](docs/i12_experiment_methodology.md)、固定条件は[experiment config](config/i12_explicit_history_experiment.json)で確認できます。

## 3. AI-assisted Development

生成AIの利用を隠さず、設計権限と実行権限を分離しています。

| Role | 主な責任 |
|---|---|
| User | 問題設定への関与、研究方針、candidate採否、最終意思決定 |
| ChatGPT | PM、ML/RL設計、実験設計、結果解釈、採否案、外部調査 |
| Codex | repository調査、実装、test、training/evaluation、artifact生成、一次解析、実装品質監査 |

当時の詳細な役割・権限は[ChatGPT Project Instructions](docs/ai_assisted_development/chatgpt_project_instructions.md)と[Codex Repository Instructions](docs/ai_assisted_development/codex_repository_instructions.md)に保存しています。Historical source、依存コード、公開用adapterの区分は[Source Provenance](package/SOURCE_PROVENANCE.json)に記録しています。

### Context / Harness / Loop Engineering

| 項目 | このケーススタディでの扱い |
|---|---|
| Context | 正本とevidenceを分け、current/candidate/historical/frozenを混同せず、必要最小限のcontextとhandoffを使います。 |
| Harness | tool権限、filesystem、Git、CPU/GPU、test、artifact、停止条件を実行前に固定します。 |
| Loop | evaluator、retry上限、stop state、human approval gateを定義し、無制限な自動反復を避けます。 |

関連資料：[ChatGPT Project Instructions](docs/ai_assisted_development/chatgpt_project_instructions.md) / [Codex Repository Instructions](docs/ai_assisted_development/codex_repository_instructions.md) / [Handoff Bundle Policy](docs/handoff_bundle_policy.md) / [Git・GitHub Security Policy](docs/git_github_security_and_operations_spec.md)

## 4. Historical DDQN Experiment

- 実施時期：2026年8月
- Algorithm：Double DQN（DDQN）
- 比較：baseline B0とL1 / L2 / L3
- 変更点：過去1〜3 decisionsのagent position/action historyをstateへ追加
- 判断：L3に限定的な改善signalはあったが、E1非昇格・正式採用なし

これはHistorical Experimentであり、現在のprivate RL systemそのものではありません。State、Reward、TradingEnv、feature engineering、current private system全体を公開するものでもありません。

詳細：[I12 Experiment Methodology](docs/i12_experiment_methodology.md) / [Technical Experiment Report](docs/technical_experiment_report.md) / [experiment config](config/i12_explicit_history_experiment.json)

## 5. Technology and Technical Evidence

使用技術：Python、TensorFlow、WSL2、Docker、NVIDIA GPU、NVIDIA TensorFlow Container（NVIDIA NGC）。実験環境はWSL2上のDockerコンテナで構築し、NVIDIA GPUを用いてTensorFlowの学習・評価を実行しています。

source、tests、config、Dockerは、実験が実際に実装・検証・実行されたことを追跡するための**technical evidence**です。

- Entry point：[training](scripts/run_experiment.py) / [graph generation](scripts/generate_graphs.py)
- Config：[I12 fixed experiment config](config/i12_explicit_history_experiment.json)
- 代表実装：[explicit history state](src/features/state_vector_v01.py) / [DDQN backup](src/training/ddqn_backup_v01.py) / [environment](src/trading_env/env_v01.py)
- Tests：[DDQN backup](tests/test_ddqn_backup_v01.py) / [Lag-1](tests/test_lag1_agent_transition_context_v01.py) / [Lag-2/3](tests/test_lag2_lag3_agent_transition_context_v01.py) / [public entry point](tests/test_public_entrypoint.py)
- Runtime：[Dockerfile](Dockerfile)
- 検証記録：[Smoke Test Report](package/SMOKE_TEST_REPORT.md) / [Graph Validation](package/GRAPH_VALIDATION.json)
- Figures：[report/figures](report/figures/)

focused tests、GPU smoke、graph validationの確認範囲と数値上の注意点は[Smoke Test Report](package/SMOKE_TEST_REPORT.md)を参照してください。testの成功をmodel性能やproduction readinessの証拠とは扱っていません。

## 6. Dataset, Reproducibility, and Public Boundary

- 実市場datasetとraw/processed dataは非公開で、本プロジェクトに含めません。
- weights、checkpoint、Replay dump、graph-source CSVも含めません。
- 第三者によるexact historical rerunはできません。
- 必要な前処理済みdatasetを別途指定すれば、公開training pathは起動できます。入力契約は[Package Contents](package/PACKAGE_CONTENTS.md)を参照してください。
- current private RL system全体や、current LSTM/recurrent研究を公開するものではありません。
- development diagnosticはfresh clean OOSではありません。
- profitable trading system、production-ready、history単独のcausal effectを主張しません。

本プロジェクトが公開するのは、Historical Experimentの完全な再実行環境ではなく、**AI-assisted ML/RL研究開発の統制、実験判断、報告、技術的証拠を追跡できるケーススタディ**です。
