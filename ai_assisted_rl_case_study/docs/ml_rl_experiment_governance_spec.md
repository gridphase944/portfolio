# ML・RL実験ガバナンス仕様

> **Historical snapshot: 2026年8月時点**  
> 本文書は当時のAI-assisted開発運用を示すための公開用snapshotであり、現在のモデル・agent挙動・推奨設定に対する最新ガイドではない。  
> 公開copy：研究開発プロセスの意味を保持し、非公開文書へのリンクと環境固有のpath表記のみ公開向けに整理した。参照する非公開文書は本packageに含めない。

本書は、`daytrading_rl_system`におけるML・RL実験の段階、promotion gate、比較条件、計算性能、metric、可視化、artifact、failure handlingを定める共通仕様である。

本書の作成または承認だけでは、個別のformal training、formal evaluation、production candidate、paper trading、live trading、broker API接続、実注文を承認しない。

## 1. 目的・対象・正本

### 1.1 目的

- 長時間学習の前に、実装、観測、artifact、計算性能の不備を小型runで検出する。
- E0、E1、E2の目的と必要な証拠を分離する。
- baselineとcandidateを公平かつ再実行可能な条件で比較する。
- 学習量不足、設計、実装、data、評価、計算性能の問題を分けて判断する。
- CodexがPM判断に必要なmetric、graph、artifactを生成できるようにする。
- 正常完走、health、計算性能、model性能、production candidate、正式採用を混同しない。

### 1.2 対象と非対象

対象：supervised／self-supervised／representation learning／RL、candidate比較、E0-1／E0-2／E1／E2、experiment plan、run identity、data role、metric、checkpoint、graph、計算性能、artifact、handoff、failure handling。

非対象：個別model architecture、feature、state、action、reward、loss、data split、metric threshold等の採否。これらはproject instruction、各正式仕様、個別experiment plan、ユーザーとChatGPTのPM判断を正本とする。

### 1.3 優先関係

- 権限、設計権、承認、禁止事項は、現在のユーザー指示とproject instructionを正本とする。
- model、data、state、reward、runtime等の意味は、ユーザー承認済みの各正式仕様を正本とする。
- 本書は、実験段階、promotion、比較、計測、可視化、artifact、failure handlingの共通手順を正本とする。
- candidate固有仕様は本書より厳しい条件を追加できるが、PM承認なしに本書を暗黙に緩和しない。

### 1.4 関連仕様

- handoff：[Handoff Bundle Policy](handoff_bundle_policy.md)
- data role／leakage：Data Inventory（本公開packageには含めない）
- feature／利用可能時点：Feature Engineering Specification（本公開packageには含めない）
- state：State Specification（本公開packageには含めない）
- environment：TradingEnv Specification（本公開packageには含めない）
- reward：Reward Specification（本公開packageには含めない）
- runtime／parallel／artifact：Parallel Rollout Architecture（本公開packageには含めない）
- limited formal route：Limited Formal Training Manifest Specification（本公開packageには含めない）
- unresolved status：Unresolved Items Register（本公開packageには含めない）

## 2. 実験段階

| stage | purpose | formal adoption |
|---|---|---|
| E0-1 | 実装、観測、artifact、計算性能のhealth確認 | 不可 |
| E0-2 | 限定実dataでE1へ進む合理性を確認 | 不可 |
| E1 | 固定条件でbaselineとcandidateを正式比較 | 採用候補を選定可能 |
| E2 | 採用候補を正式実装、仕様、artifact、runtimeへ統合 | 完了後に別途承認 |

E0の結果だけで正式採用を判断しない。E2中に新しい設計仮説が必要になった場合はE0またはE1へ戻す。

## 3. 共通experiment plan

実験は実行前にexperiment planを持つ。小型E0-1 fixtureでは簡略化できるが、E1では必要項目を固定する。

### 3.1 必須項目

- experiment ID、version、stage、目的、仮説
- baseline、candidate、変更対象、非対象、関連仕様
- data source、role、split、fit範囲
- model、state、target、loss、metric等のidentity
- primary／secondary／health metric、denominator、aggregation、not-applicable条件
- training量、batch size、seed、run matrix
- runtime mode、device、worker、environment
- config、override、checkpoint、selection、early stopping
- required graph set、artifact、handoff条件
- telemetry、計算予算、retry、stop、cleanup、promotion gate、approval state

planとartifactには、runを再構成できるcommit、config、split、seed、schema、software、hardware、runtime、checkpoint、stop reasonを保存する。

### 3.2 仮説

仮説は、変更内容、期待するmechanism、影響するmetric／subset、E0で成立すべき必要条件、棄却・再設計条件、追加学習で判断が変わり得る条件を記録する。

一度に複数の独立設計要素を変更しないことを原則とする。不可分の場合は理由と比較不能になる範囲を明示する。

## 4. E0-1：実装・観測・性能health

### 4.1 目的

長時間run前に、実装、観測可能性、artifact、計算実行性が成立していることを確認する。model性能は判断しない。

### 4.2 必須確認

**実装health**

- 入出力、shape、dtype、forward、loss、gradient、optimizer update
- mask、target、label、normalizer、episode／batch境界
- NaN、Inf、missing gradient等の異常
- checkpoint保存、strict reload、必要な場合のresume
- CPU実行、GPU要求時のactual GPU execution
- data供給、shutdown、timeout、error propagation

**観測・artifact health**

- run ID、commit、config、split、seed、schema、actual runtime
- loss／metric、denominator、invalid count、gradient norm、learning rate
- checkpoint選定理由、停止理由、verdict、telemetry
- required graph、graph manifest、compact summary、handoff

fieldの存在だけでなく、値、型、意味、runtimeとの一致を確認する。

**計算性能health**

- batch処理、actual device、data取得、environment step、train update
- samples／秒またはsteps／秒、worker数、queue／input wait、peak memory
- E1の想定wall-clockと計算予算

完全な最適化は求めない。明らかな供給詰まり、GPU待機、memory超過、非現実的なE1推定時間は次段階前に確認する。

### 4.3 Hardcodeとverdict

fixture、synthetic、極小scopeのdiagnostic hardcodeは許容する。ただしformal／E1 pathから分離し、artifact scopeへ記録し、E1前に必要なconfig化とwiring testを行う。

verdict：`E0_1_PASS`、`E0_1_REPAIR_REQUIRED`、`E0_1_BLOCKED`、`PM_DECISION_REQUIRED`。

## 5. E0-2：小規模な仮説確認

### 5.1 目的と比較条件

限定実dataと短時間学習で、E1の計算予算を投入する合理性を判断する。「フル学習しても絶対に改善しない」とは断定しない。

baselineとcandidateでdata role、metric、denominator、主要runtime条件を合わせ、candidateだけに有利なdata、fit、threshold、overlayを使わない。小型scopeとformal条件との差、diagnostic dataの再利用による独立性低下を記録する。

### 5.2 確認事項

- 学習信号と必要parameter更新
- lossまたは目的metricの方向性
- baselineとの差、train／validation乖離
- target、prediction、error分布の退化
- Head、component、symbol、date等への集中
- nonfinite、gradient explosion／vanishing
- 計算性能、E1推定時間、learning curve
- 学習量不足か、設計・data・評価修正が先か

### 5.3 Reportとverdict

reportには、仮説、必要条件、観測結果、学習量不足と設計問題の切り分け、追加計算で結論が変わる可能性、次段階へ進む／停止／再設計する理由、判断を覆す追加証拠を含める。

verdict：`PROMOTE_TO_E1`、`EXTEND_E0`、`REDESIGN_REQUIRED`、`STOP_CANDIDATE`、`BLOCKED`、`PM_DECISION_REQUIRED`。

`PROMOTE_TO_E1`は正式採用を意味しない。

## 6. E1：正式比較

### 6.1 開始前promotion gate

- E0-1 healthが成立し、E0-2からE1へ進む理由が記録されている
- baseline、candidate、data role、split、fit範囲が固定されている
- metric、denominator、selection、verdictが固定されている
- training量、seed、run matrix、計算予算が承認されている
- 主要値がconfig化され、runtimeへのwiring testがある
- required graph、artifact、handoff、高速実行経路が成立している
- formal trainingが必要な場合はユーザーの明示承認がある

### 6.2 正式比較条件

- baselineとcandidateで同じdata、sampling、training量、evaluation条件を使う。
- 差を意図したfield以外は同一にする。
- hardware／runtime差がある場合、model性能と計算性能を分ける。
- stochasticなcandidateで単一seedを十分な再現性証拠とみなさない。
- checkpoint selectionは許可されたvalidation dataだけを使う。
- evaluation、holdout、backtest、fresh clean OOSをcandidate調整へ使わない。

E1開始後にmodel、data role、metric、seed、training量、baseline／candidate定義、性能へ影響するruntime条件を変更した場合、同一比較として継続しない。意味を変えないbug修正でもrun identityを分け、再実行範囲をPM判断する。

### 6.3 結果とverdict

runtime completion、implementation health、training health、計算性能、model performance、seed variability、subgroup concentration、production candidate eligibilityを区別する。

verdict：`ADOPT_CANDIDATE_FOR_E2`、`REJECT_CANDIDATE`、`PARTIAL_REDESIGN`、`ADDITIONAL_EVIDENCE_REQUIRED`、`RUN_INVALID`、`PM_DECISION_REQUIRED`。

## 7. E2：正式採用・統合

E1の採用候補をcurrent正式実装へ統合し、仕様、runtime、artifact、文書を同期する。

必須確認：承認designとの一致、regression test、schema identity、checkpoint strict reload、必要なresume、artifact round-trip、configとactual runtime、必要なmode parity、限定再現run、正式仕様・README・unresolved register、legacy／rejected artifact境界、承認状態。

E2中に新architecture、feature、loss、metric等を試さない。必要ならE0またはE1へ戻す。

verdict：`E2_COMPLETE`、`E2_BLOCKED`、`ROLLBACK_REQUIRED`、`RETURN_TO_E0`、`RETURN_TO_E1`、`PM_DECISION_REQUIRED`。

## 8. Data role・fit・leakage

| role | boundary |
|---|---|
| train | gradient、normalizer fit、train-only baseline等、planで学習用と明示されたdata |
| validation | checkpoint／candidate selectionに使えるがfitへ使わないdata |
| evaluation | 固定artifactの評価用。学習、fit、candidate調整へ使わない |
| holdout／backtest | 事前定義した確認用。結果を見て変更した後は独立評価ではない |
| consumed holdout | 過去に判断へ使用済みであるprovenanceを保持するdata |
| fresh clean OOS | model、feature、threshold、選択、diagnosticに未使用のdata |
| diagnostic | health／原因分析用。formal性能証拠へ暗黙昇格しない |

normalizer、statistics、feature selection、threshold、baseline、calibration等は、planで許可されたtrain dataだけでfitする。

future data、future label、future PnL、evaluation専用score、teacher／oracle結果をcurrent input／policyへ暗黙注入しない。report-only、counterfactual、oracle、eval-only overlayをactual policy performanceと同一視しない。

role、split、fit範囲を変更する場合は、新しいversioned split／manifestとPM承認を必要とする。directoryへのdata追加だけで自動採用しない。

## 9. Config・hardcode・runtime binding

E0-1のfixture、synthetic、極小scopeではdiagnostic hardcodeを許容するが、formal／E1 pathから分離し、artifact scopeへ記録する。

E1前には、実験条件へ影響するepoch、step、sample数、batch size、seed、data scope、model、optimizer、worker、evaluation頻度、checkpoint、runtime mode、device、output root等をconfigまたは引数から変更可能にする。

指定値がactual runtimeとartifactへ反映されることをfocused wiring testで確認する。hidden hardcode、silent fallback、未記録overrideを許容せず、CLI、environment、config fileの優先順位とactual normalized configを保存する。

## 10. 計算性能とhardware

高速化はE0から重要な設計要件とする。長時間学習candidateは、E1前に利用可能なhardwareを有効利用する高速実行経路を持つ。single-process／eager経路はcorrectness確認用として維持できるが、それだけを正式長時間経路としない。

GPUを要求するtraining runは、全experiment stageでCodex sandbox外で実行し、起動前にユーザーの明示承認を得る。Codex sandbox内でGPUを利用できない場合も、CPUへfallbackしてtrainingを継続しない。

batch化、GPU、parallel data loading、multiprocessing、複数environment、bounded queue、prefetch、`tf.data`、`tf.function`、XLA、mixed precision、multi-GPU等から、処理と実測に適した方式を選ぶ。

通常runでは、throughput、data取得、environment step、train update、wall-clock、actual device、worker数、memory、queue／input wait等のlightweight telemetryを保存する。

GPU利用不足、供給詰まり、計算予算超過、worker scaling不足、memory、timeout、deadlock等が疑われる場合だけ詳細profilingを行う。

高速化前後でsample、transition、target、metric、seed、data order、数値安定性、stop condition、checkpoint、artifact、比較公平性が変わっていないか確認する。効果がない場合は過剰な最適化を続けない。

## 11. Metric・denominator・evaluation

各metricは、名称・version・purpose、formula／aggregation、denominator、applicable条件、aggregation単位、改善方向、baseline、gate、data role、checkpoint／verdict利用可否を定める。

optimization lossとPM判断用のperformance metricを分ける。loss低下だけで目的性能が改善したと判断しない。nonfinite、missing gradient、invalid reward、discarded episode、failure、timeout、artifact eligibility等のhealth metricも分離する。

RLではlossだけでpolicy性能を判断せず、planに応じてreward、PnL、action、episode、holding、forced exit、execution／valuation failure、Replay、Q値、TD error、exploration、symbol／date／session集中等を確認する。

formal evaluationはsaved artifact、fitted normalizer、固定data、固定policy、固定metricを使い、training、optimizer update、Replay add、normalizer fit、candidate調整を行わない。

同一条件でsequential／parallelの意味とreportを比較する。bitwise一致を保証できない場合は、保証範囲と比較単位をplanで定める。

## 12. 可視化

Codexはrequired graph setと一次解析に必要なgraphを生成し、全graphと再生成可能な元dataをGit対象外の`outputs/`へ保存する。

handoff bundleには、PM判断・教育説明に必要なgraph、graph manifest、compact comparison table、主要数値と制約を含める。full CSV、raw prediction、full telemetry、大量の再生成dataは原則として含めず、relative path、schema、件数、SHA-256で参照する。

各graph manifestには、graph ID、path、purpose、stage、必要性、axis、series、metric、denominator、raw／smoothed、smoothing条件、比較条件、source data identity、limitationsを記録する。

axis、unit、denominator、sample数を明示し、baselineとcandidateは可能な範囲で同じscale、frequency、smoothingを使う。raw値を保持し、truncated axisや恣意的な範囲でcandidateを有利に見せない。empty画像、series欠落、条件不一致を拒否する。

required graph setはplanで定める。候補にはloss、primary metric、baseline比較、seedばらつき、分布、subgroup集中、throughput、failure、RL reward／PnL、action、Replay、Q／TD error等を含む。

ChatGPTは原則としてgraphを再生成せず、bundle内の重要graphを選んでPM判断と初学者向け説明に使用する。

## 13. Artifact・handoff

run outputは必要に応じて、config、split／data role、run identity、metric、health、telemetry、graph、checkpoint、normalizer、weights、source manifest、failure summaryを持つ。巨大artifactはGit対象外の正式保存場所へ残す。

Codexはbundle前に、execution health、主要metric、baseline比較、異常、結果集中、config／split／seed／commit、required graph、PM判断事項を整理し、巨大成果物を未分析のままChatGPTへ渡さない。

handoffの内容、命名、保存場所、禁止物、validationは[Handoff Bundle Policy](handoff_bundle_policy.md)を正本とする。実験本体とhandoffの成否は分けて報告する。

## 14. Failure・retry・cleanup

stop stateは最低限、`PASS`、`FAIL`、`BLOCKED`、`PM_DECISION_REQUIRED`、`CANCELLED`を区別する。stage固有verdictは併記できる。

retryは原因、上限、計算予算を定める。environment failure、data／artifact corruption、implementation bug、nonfinite、timeout／deadlock、計算性能不足、model性能不足を区別し、model性能不足をblind retryで解決しない。

failed／fatal／aborted／partial runをformal success artifactまたはregistry latest／bestへ昇格させない。

failure時は新規assignment、training update、artifact publishを止め、process、worker、queue、temporary artifactをcleanupする。existing valid artifactを上書きせず、forced terminationは最後の手段として理由を記録する。

## 15. Test・導入・既存run

適用範囲に応じて、plan／manifest parse、config wiring、run identity、data role／fit、checkpoint reload、metric denominator、graph／manifest、telemetry、promotion gate、failure eligibility、handoff validation、Markdown／JSON／CSV parse、`git diff --check`を確認する。

unit testやsmokeのPASSをmodel性能、formal training、production candidateの証拠にしない。full trainingを通常のtest gateとして要求しない。

本仕様は承認後の新規experiment planへ適用する。既存historical run、artifact、仕様のprovenanceを書き換えず、欠損fieldを現在の情報で偽装しない。

`AGENTS.md`はML・RL experiment taskで本書を読むようroutingし、candidate固有仕様は共通項目を必要以上に複製しない。machine-readable schema、default threshold、seed数、graph形式、CI validator等の未決定事項はUnresolved Items Register（本公開packageには含めない）で管理する。
