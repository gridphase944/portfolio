# I12 explicit agent history：公開実験methodology

## 1. Scopeと証拠の範囲

本文書は、I12 historical case study（DDQN Lag-1〜3 explicit agent history E0）の技術レポートを理解するための公開用methodologyである。current private RL systemの完全仕様ではなく、市場データは非公開である。exact historical rerunを保証する文書ではない。current LSTM/recurrent researchはscope外とする。

根拠は2026-08-27のhistorical technical reportとbeginner HTML、I12の保存済みexperiment plan／run config、当時のsource／spec snapshot、report manifest／provenanceである。current private ML/RL仕様の要約ではない。同梱のプロセス仕様3冊は公開準備時点の最新版であり、当時の実験条件を遡って変更するものではない。

実験結果・判断・図の正本は[Technical Experiment Report](technical_experiment_report.md)、平易な解説は[Human-readable Experiment Report](human_readable_experiment_report.html)を参照する。source出典と再構成上の限界は[公開source provenance](../package/SOURCE_PROVENANCE.json)に記録されている。

## 2. Data / Featureの概要

日本株の板・価格snapshotから作成したmarket stateと、agent自身のposition等の内部状態を使用した。decision時点以前に観測可能な情報だけをpolicy inputに用い、future informationを入力しない。5秒gridへのsnapshot割当は同一session内のlatest-priorであり、将来のrowを遡って使わない。約定評価に使う1秒後の情報とdecision時点のpolicy inputを区別する。

trainとdevelopment diagnosticは分離されている。trainは2026-06-24／25／26／29の192 episodes、development diagnosticは2026-06-30／07-01／07-03の144 episodesである。ただしdevelopment populationは過去の研究判断に使用済みであり、未使用の独立評価dataではない。

dataset本体、market featureの全列一覧・全算出式、raw schemaは掲載しない。公開入口が必要とする前処理済みdatasetの入力契約は[package利用説明](../package/PACKAGE_CONTENTS.md)を参照する。

## 3. Stateとexplicit history

| Variant | State dim | Q input dim | Parameters |
|---|---:|---:|---:|
| B0 | 59 | 118 | 32,131 |
| L1 | 65 | 130 | 33,667 |
| L2 | 71 | 142 | 35,203 |
| L3 | 77 | 154 | 36,739 |

B0の59次元を基礎とし、L1／L2／L3は直前1／2／3 decisionsのagent transitionを追加する。市場snapshotの履歴全体を複製する設計ではない。各lagは次の順の6値で、全lagに同じencodingを使う。

| 3値のgroup | 意味 | one-hotの順序 |
|---|---|---|
| position-before | 過去decisionでactionを実行する前のposition | Flat=(1,0,0)、Long=(0,1,0)、Short=(0,0,1) |
| effective action | mask／gate／fallback後、実際にEnvへ渡した最終semantic action | Hold=(1,0,0)、Buy=(0,1,0)、Sell=(0,0,1) |

現在decisionを$t$とすると、prev1は$t-1$、prev2は$t-2$、prev3は$t-3$のposition-beforeとeffective actionを表す。新しいlagから順に連結し、L1はprev1、L2はprev1・prev2、L3はprev1・prev2・prev3を使う。raw Q argmaxそのものを保存するわけではない。現在のactionを決定前のstateへ入れず、transition後のnext stateでqueueを進める。

履歴は同一session内に限定し、午前・午後の最初は全lag unavailableとする。履歴resetとは別に、position自体は当時のenvironment契約に従い昼休みを越えて維持される。未観測lagの6値はNaN、対応するvalidity maskはfalseであり、実際に観測したFlat／Holdのone-hotと区別する。Q inputでは無効値を0にしたstateと同次元のvalidity maskを連結するため、入力次元はstate次元の2倍となる。

parameter matchingは行っていない。hidden widthは全variantで128×128に固定され、history depthの増加に伴いinputとparameter数も増える。L3はB0比4,608 parameters、約14.34%多い。したがってモデル全体の比較であり、history単独のcausal effectとは断定できない。59 base stateの完全な列一覧は掲載しない。

## 4. Q network / DDQN

Q networkはDense 128 ReLU → Dense 128 ReLU → output 3 linearである。Q出力とReplayのaction順序はHold／Buy／Sell。利用可能なactionに対してonline networkが次actionを選び、target networkが選ばれたactionのQ値を評価するDouble DQN（DDQN）を使用した。backupはone-step、gammaは0.9962である。

次action selection、Bellman target、TD errorの数式は[Technical Experiment Reportの数学的条件](technical_experiment_report.md#6-mathematical-contract)を参照する。

## 5. Reward

当時のrewardは、100株固定のliquidation equity差分を前日終値基準の想定元本で正規化した値である。

$$
r_t =
\frac{E_t - E_{t-1}}
{100 \times P_{\mathrm{prevclose}}}
$$

- $r_t$：step $t$の正規化reward。
- $E_t$：action実行後のliquidation equity（円）。cashの売買収支と、残るpositionを清算したときの価値を合わせたもの。Flatではcashの売買収支に等しい。
- $E_{t-1}$：Envが保持する直前のreward-validなliquidation equity（円）。episode開始時は0。
- $P_{\mathrm{prevclose}}$：対象銘柄の前日終値（PreviousClose、円／株）。
- 100：固定取引数量の100株。分母は「100株×前日終値」の円建て想定元本であり、rewardは無次元となる。

通常stepでは1秒latency後のaction実行・清算評価に基づく。reward-valid episodeでは差分がtelescopingし、step reward和とfinal normalized PnLが一致する。この定義を超えるfailure／Replay admissionの全契約や現行の設計候補は扱わない。

## 6. Environment

| 項目 | I12比較の条件 |
|---|---|
| Action | Hold / Buy / Sell |
| Position | Flat / Long / Short |
| 取引数量 | 100株固定 |
| Decision interval | 5秒 |
| Execution latency | 1秒 |
| Variant間の条件 | B0／L1／L2／L3で同一のlatency-1s / closing-risk environmentを使用 |

履歴depth以外のenvironment条件をcandidateごとに変更した比較ではない。environmentの完全仕様やI12理解に不要な詳細段階は掲載しない。

## 7. Training条件

以下はhistorical I12の固定条件であり、公開build確認用smokeの設定とは区別する。

| 項目 | Historical条件 |
|---|---|
| Algorithm | DDQN（最終B0／L1／L2／L3 depth比較） |
| Backup / gamma | one-step / 0.9962 |
| Optimizer / learning rate | Adam / 0.001 |
| Loss | Huber、delta 1 |
| Gradient clip | global norm 10 |
| Batch | 128 |
| Replay capacity | 50,000 |
| Replay sampling | uniform without replacement（batch内で重複なしの一様抽出） |
| Target sync | 200 optimizer updatesごとのhard sync |
| Epsilon | 1.0 → 0.1 |
| Epsilon decay | 20,000 global environment stepsで線形減衰 |
| Optimizer updates | 2,000 / run |
| Seeds | 20260815 / 20260821 / 20260822 |
| Primary checkpoint | 事前固定のupdate_2000。結果を見てbest checkpointを選ばない |

Lag-1の初期研究にはDQN比較も含まれるが、本書のdepth比較はDDQNである。公開buildの既存50-update smokeは実行health確認であり、上表のhistorical学習量や性能結果を置き換えない。公開configは[固定実験条件](../config/i12_explicit_history_experiment.json)を参照する。

## 8. Evaluation boundary / limitations

- 比較は3 seeds。development diagnosticはfresh clean OOSではない。
- 集約の432 outcomesは144 market episodes × 3 trained seedsであり、432独立market episodesではない。
- L3の中心指標には改善方向があるが、seed variabilityが残る。p01、worst 5% mean、maximum lossにはtail risk悪化がある。
- absolute profitabilityは成立しておらず、E1へpromotionしていない。
- parameter capacity増加とhistory効果は分離できない。history単独の因果効果や正式採用を主張しない。

performance resultの詳細と分母は[Technical Experiment Report](technical_experiment_report.md)を正本とし、本書では全結果を重複しない。

## 9. Public source / testとの対応

| 対象 | 公開package内の実装・test |
|---|---|
| History encoding / state | [state_vector_v01.py](../src/features/state_vector_v01.py)・[state_builder_v01.py](../src/trading_env/state_builder_v01.py) |
| History queue / environment | [env_v01.py](../src/trading_env/env_v01.py) |
| Reward | [reward_v01.py](../src/trading_env/reward_v01.py) |
| MaskとQ input | [q_input_adapter_v01.py](../src/training/q_input_adapter_v01.py) |
| Q/model builder | [q_network_v01.py](../src/training/q_network_v01.py) |
| DDQN backup | [ddqn_backup_v01.py](../src/training/ddqn_backup_v01.py)・[test_ddqn_backup_v01.py](../tests/test_ddqn_backup_v01.py) |
| Replay | [replay_buffer_v01.py](../src/training/replay_buffer_v01.py) |
| Training entrypoint | [run_experiment.py](../scripts/run_experiment.py) |
| Lag focused tests | [Lag-1 tests](../tests/test_lag1_agent_transition_context_v01.py)・[Lag-2／3 tests](../tests/test_lag2_lag3_agent_transition_context_v01.py) |

公開sourceはhistorical実装の必要部分と公開用入口から構成される。全baselineの実行時source・dataset・weights・環境を完全に保存したclosureではない。利用条件は[package利用説明](../package/PACKAGE_CONTENTS.md)、既存のpublic-build確認範囲は[実装確認記録](../package/SMOKE_TEST_REPORT.md)を参照する。
