# Handoff Bundle Policy

> **Historical snapshot: 2026年8月時点**  
> 本文書は当時のAI-assisted開発運用を示すための公開用snapshotであり、現在のモデル・agent挙動・推奨設定に対する最新ガイドではない。  
> 公開copy：研究開発プロセスの意味を保持し、非公開文書へのリンクと環境固有のpath表記のみ公開向けに整理した。参照する非公開文書は本packageに含めない。

## 1. 目的

この文書は、`daytrading_rl_system`において、CodexからChatGPTへ作業結果・実験結果を渡す方法を定める。

目的は、ユーザーが複数ファイルを個別に探して添付したり、巨大なCSVやログを手作業で整理したりせず、1つのzipをChatGPTへ添付するだけでPM判断を進められる状態にすることである。

## 2. 基本原則

Codexは、巨大なCSV、JSON、ログ、checkpoint、実験artifactをそのままChatGPTへ渡さない。

Codexがローカル環境で一次解析を行い、ChatGPTがPM・ML／RL設計責任者として判断するために必要な情報だけを、小型の要約、比較表、設定snapshot、図表、manifestへ圧縮する。

ChatGPTへ1つ以上のファイルを渡す場合は、ファイル数にかかわらず、原則として1つのhandoff bundle zipにまとめる。

最終報告の本文だけで十分な作業では、zipを作成しなくてよい。

## 3. Codexによる一次解析

実験や評価で巨大な成果物が生成された場合、Codexはbundle作成前に少なくとも次を行う。

* 実行healthの確認
* 主要metricの集計
* baselineとcandidateの比較
* 異常値、欠損、nonfinite値、実行失敗の確認
* 必要に応じた銘柄、日付、symbol-day等への結果集中の確認
* 設定、split、seed、run ID、commit等の再現条件の整理
* ChatGPTが判断すべき問題と未解決事項の抽出

Codexは、巨大成果物を未分析のままbundleへ入れ、ChatGPTへ第1分析を委ねない。

Codexの一次解析は、事実確認、集計、比較、異常検出、原因候補の整理までとする。実験の正式採否、仕様変更、次の研究方針はChatGPT PMとユーザーが判断する。

## 4. 最終報告

Codexの最終報告本文には、少なくとも次を含める。

1. 目的と結論
2. 完了した範囲と未完了の範囲
3. 実行・変更した内容
4. 主要な結果と根拠
5. テストおよび実行health
6. 原因候補と未解決事項
7. 次に必要なPM判断
8. 変更ファイルとGit状態
9. handoff bundleのpathと内容

該当しない項目は、`対象外`または`なし`と明記すること。

巨大ログ、巨大CSV、diff全文を最終報告本文へ貼らない。

## 5. bundleを必須とする条件

次の場合はhandoff bundleを必須とする。

* ChatGPTへ1つ以上のファイルを渡す
* model trainingまたは重要なevaluationの結果を渡す
* 複数候補またはbaselineを比較した結果を渡す
* 正式なhandoffまたはcloseoutを行う
* 編集後の仕様書snapshotをChatGPTへ渡す
* ChatGPTまたはユーザーがbundle作成を明示した

本文だけで判断でき、ChatGPTへ渡すファイルがない作業ではbundleは不要である。

## 6. bundleに含めるもの

bundleには、ChatGPTのPM判断に必要な小型成果物だけを含める。

例：

* `decision_summary.md`
* `condensed_summary.md`
* `run_config_snapshot.json`
* `split_snapshot.json`
* `baseline_comparison.csv`
* 小型の集計CSV
* 小型のグラフまたは画像
* `reproducibility_summary.json`
* `artifact_manifest.json`
* `spec_sync_check.md`
* 必要な仕様書の編集後snapshot
* bundle内ファイルを説明する`README.md`

ファイル一覧は固定しない。作業内容に必要なものだけを作る。

summaryには、結論だけでなく、その結論を確認できる主要数値、比較条件、分母、既知の制約を含める。

## 7. bundleに含めないもの

次はbundleへ含めない。

* raw data本体
* processed data全量
* 巨大CSV・JSON・Markdown
* 巨大raw log
* checkpoint
* model weights
* replay buffer
* TensorBoard event
* cache
* full transition dump
* full episode／step metrics
* full action／Q-value／state dump
* Docker image
* secret、token、API key
* local config
* 証券口座情報
* 個人情報
* 不要な中間生成物

再検証に必要な巨大artifactは削除せず、Git対象外の正式保存場所へ残す。

bundleには、必要に応じて次を記録する。

* artifact path
* run ID
* schema
* row、episode、transition数
* configおよびsplitの識別情報
* manifestまたはhash
* 既知の制約

## 8. bundle作成失敗

bundle必須の作業でzipを作成できない場合、実装や実験本体の成否とhandoffの成否を分けて報告する。

例：

```text
作業本体：完了
handoff：BLOCKED
```

最終報告には次を含める。

* 失敗したbundle path
* 失敗理由
* 作成済みの個別ファイル
* 個別ファイルが有効か
* ユーザーまたはPMに必要な対応

bundle作成に失敗したことだけを理由に、正常に完了した学習や実装まで失敗扱いにしない。

## 9. 命名と保存場所

bundle名は次を基本とする。

```text
<YYYYMMDD>_daytrading_rl_system_<topic>_<version>_chatgpt_handoff.zip
```

保存場所は原則として次とする。

```text
outputs/<experiment-or-topic>/
```

bundle、summary、manifestを一時directory、`docs/`、source code、tests、config、Git管理対象へ置かない。

個別依頼で名前または保存場所が指定された場合は、その指定を優先する。

## 10. 正式仕様との関係

確定した設計や仕様は、handoff bundleだけに残さず、関連する正式仕様書へ反映する。

`docs/`には長期参照する正式仕様と設計判断を置く。

`outputs/`には個別作業・実験の結果、集計、diagnostics、handoff bundleを置く。

Codexの最終報告、ChatGPTへ渡すファイル、handoff bundleは、この文書を正本とする。個別依頼では、タスク固有の成果物や確認事項だけを追加し、独自の返却形式を新設しない。
