# 公開buildの実装確認

Status: PASS。これは`PUBLIC_BUILD_E0_1_SMOKE`であり、historical I12の性能再評価ではありません。

| Variant | Seed | Successful updates | Training tensor | Parameters | Finite / health |
|---|---:|---:|---|---:|---|
| B0 | 20260815 | 50 | GPU:0 | 32,131 | PASS |
| L1 | 20260815 | 50 | GPU:0 | 33,667 | PASS |
| L2 | 20260815 | 50 | GPU:0 | 35,203 | PASS |
| L3 | 20260815 | 50 | GPU:0 | 36,739 | PASS |

合計200更新。NVIDIA GeForce RTX5070 Ti、TensorFlow2.17.0。CPU fallbackなし。Env/data処理を含む全処理がGPUだったという意味ではありません。

historical train-roleの同一1 episodeを4 variantへ使用しました。state/Q形状、Replay mutation、finite reward/Q/target/TD/loss、nonempty gradients、optimizer iteration、initial target syncと200更新前の非sync、artifact writerを確認しました。周期200更新の境界はfocused testで確認しています。

## Testsとgraph

- 保存済みhistorical receipt：Lag1 13 PASS、Lag2/Lag3 12 PASS、DDQN backup 13 PASS。
- 今回の公開build：focused tests 44 PASS（historical 38件とpublic adapter 6件）。
- 新smoke graph：各variant 6枚、計24枚。sourceとaxis/seriesの存在・finite・非emptyを確認。
- historical graph：G01〜G22と参照Phase-I 2枚、計24枚を再生成。24枚すべて原本PNGとbyte一致。非公開source CSVは同梱しません。

## 限界と修正記録

学習済み性能、収益性、fresh OOSを今回判断していません。通常の2,000更新経路を実行したという意味でもありません。

初回smokeの最終観測JSONで、未接続の累積counterが初期値のまま残っていました。公開adapterのwriterだけを修正し、保存済みの実測run receiptとlearning rowsから最終writerを再実行しました。historical実装・学習結果に影響せず、追加trainingは0更新です。writer修正後にGPU training全体を再実行したとは主張しません。

初回focused testの1 fixtureがOS標準一時directoryを使用しました。context終了時に削除済みで、最終testsでは一時保存先を専用検証directoryへ固定しました。

公開sourceのsemantic bodyはhistorical実装と照合しています。不確実な未使用functionの削除は行っていません。dataset、smoke観測data、一時画像、weights等はpublic packageに含めません。

## Docker / GPU 手動検証

`validation_executor = user_manual_wsl`。以下はユーザーがWSL上で実行し、最終ZIP化の依頼で提供した手動検証結果です。CodexがDocker build・container tests・GPU probeを実行した結果ではありません。前節までの44 PASSは先行する公開build検証の記録であり、本節のcontainer testsとは別の結果です。

| 確認項目 | ユーザー報告の結果 |
| --- | --- |
| Docker image build | PASS |
| NVIDIA NGC Release | 25.01-tf2 |
| TensorFlow | 2.17.0 |
| GPU visible | GPU:0 |
| GPU model | NVIDIA GeForce RTX 5070 Ti |
| TensorFlow GPU tensor computation | PASS |
| finite result | true |
| `scripts/run_experiment.py --help` | PASS |
| Public focused tests inside container | 43 PASS / 1 strict bitwise equality不一致 |

不一致の対象は`tests/test_ddqn_backup_v01.py::DoubleDQNLearnerIntegrationTest::test_h0_telemetry_on_off_learning_parity_and_online_only_update`です。

| 観測差 | 値 |
| --- | ---: |
| mismatched elements | 11 / 128 |
| max absolute difference | 1.1641532e-10 |
| max relative difference | 1.16741504e-7 |

ユーザー提供のPM判断では、これはGPU floating-point execution orderによる既知タイプの数値非決定性として扱い、functional / ML semantic mismatchとは扱いません。Docker public validationのblockerにはしませんが、strict equality test自体がPASSしたとは記録しません。test sourceは変更していません。この判断は当該観測差に限定し、他のtest failureを許容する一般ルールではありません。

本節は公開environmentの実行healthの記録であり、I12 historical run identityの完全一致、historical performanceの再評価、exact rerunを証明するものではありません。最終ZIP化ではDocker / GPU / trainingを再実行していません。ユーザー手動環境のimage / container cleanupの実施状況は確認していません。
