# DQNによる株式売買判断モデル

> **Historical Project / 開発時期：2024年頃**
>
> 強化学習を実際の意思決定問題へ適用することを目的に、株式売買を題材として作成したDQNです。  
> 本プロジェクトでは、当時のシステム全体のうち **DQNの学習・Replay Memory・BacktestなどML/RL部分** を公開しています。

## このProjectで示すもの

このProjectは、現在のRL実装のベストプラクティスや収益性能を示すものではなく、**当時本人がDQNを学びながら、TensorFlowのcustom training loop、Replay Memory、RL環境のState / Action / Rewardなどを実際に実装していたこと**を示すHistorical Projectとして公開しています。

株式売買は強化学習を実践するための題材として使用しました。  
公開コードでは、OCR等で構造化済みの市場データを入力として扱います。

## 開発・実装範囲

当時はWeb上の資料等を参照しながら、本人中心で以下を実装しました。

- DQNの学習ループ
- Q Network / target network
- Experience Replay
- ε-greedyによる行動選択
- `tf.GradientTape`を用いたcustom training
- State生成
- Reward / position管理
- 学習状態・Replay Memoryの保存 / 再開処理
- 学習済みモデルを用いたBacktest

当時の開発では生成AIも一部利用しています。  
また、GitHub公開時のコード監査・公開安全性確認・README整理にはChatGPT / Codexを利用しています。

## 処理の流れ

公開版では、OCR等の前段処理でNumPy形式へ構造化されたepisodeデータから学習を開始します。

```text
1秒間隔の市場データ
    ↓
約定値・現在値・買い板・売り板を読み込み
    ↓
Stateを生成
    ↓
Q Network
    ↓
ε-greedyで Hold / Sell / Buy を選択
    ↓
Reward・positionを更新
    ↓
Replay Memoryへtransitionを保存
    ↓
ReplayからsampleしてQ Networkを更新
    ↓
target networkを定期同期
```

Backtestでは学習用とは別のepisodeデータを使用し、学習済みモデルのWeightを固定した状態でgreedyにActionを選択します。

## 入力データ

1 stepは、**約1秒間隔で取得した1時点の市場状態**です。

各episode directoryでは、次のNumPyファイルを読み込みます。

| ファイル | 内容 |
|---|---|
| `value.npy` | 約定値系列 |
| `nowPriceIndex.npy` | 現在値に対応するindex |
| `price.npy` | 価格系列 |
| `buy.npy` | 買い板 |
| `sell.npy` | 売り板 |

episodeの長さはデータ収集時期・用途によって異なるため、固定長を前提としていません。

## RL環境の設計

### State

Stateは主に次の情報から構成しています。

- 直近の約定値変化
- 現在値周辺の買い板・売り板
- position（flat / long / short）
- 保有中の損益

約定値は最新値との差分として扱い、価格帯ごとに設定したscalingを用いて値を調整しています。  
これは、価格水準の異なる銘柄を同一のNetworkで扱うことを意図した当時の設計です。

買い板・売り板は現在値周辺からそれぞれ5要素を切り出し、一定範囲へscalingしてStateへ加えています。  
positionは `flat=0 / long=1 / short=-1` として表現し、保有中の損益もStateへ含めています。

### Action

Actionは3種類です。

| Action | 意味 |
|---|---|
| `0` | Hold |
| `1` | Sell（short entry / long close） |
| `2` | Buy（long entry / short close） |

学習時はε-greedyで探索します。

episodeの最終decisionでは、未決済positionを残さないようActionを制限します。

- flat → Hold
- long → Sell
- short → Buy

### Reward / Position

Rewardは実現損益を基準として計算します。

- 利益確定時：scaling後のgainを2倍してRewardへ反映
- 損失確定時：scaling後のgainをそのままRewardへ反映
- 売買が成立しないAction：基本的にRewardは0

positionはlong / short / flatを管理し、episode終了時には未決済positionを清算する構成です。

## DQNの実装

Q NetworkはTensorFlow / Kerasで実装しています。

```text
Input
  ↓
Dense(64, tanh)
  ↓
Dense(32, tanh)
  ↓
Dense(3, linear)
```

主な学習構成は次のとおりです。

- optimizer: Adam
- loss: Huber Loss
- discount factor: `γ = 0.99`
- Replay Memory: 最大1,200 transition
- Replay sampling: 30 transition
- main network / target networkを分離
- selected Actionに対応するQ値をTD targetへ近づけるcustom training

Replay transitionは次の5要素です。

```text
(state, action, reward, next_state, done)
```

terminal transitionでは `done=True` とし、未来価値をbootstrapしません。

## 当時入れた学習上の工夫

### 微小変動時のdecisionを間引く

position保有中に価格変動が一定範囲内の場合、同様の市場状態でdecisionが大量に発生することを避けるため、raw stepを一部skipしています。

### Replay Memoryへの追加を制御する

Holdや追加売買が成立しない類似経験ばかりでReplay Memoryが占有されることを避けるため、`addStateFlag`でReplayへ追加する経験を制御しています。

これらは当時、学習データの偏りや無意味なdecisionの増加を抑える目的で入れた設計です。  
一方で、decision間隔やReplay内の経験分布を人為的に変える設計でもあるため、**現在はその影響も考慮すべきHistoricalな設計判断**として位置づけています。

## Backtest / 評価

`backtest.py`では、学習用とは分離した `OCR_Result_Numpy_Backtest` のデータを使用します。

- 学習済みモデルを読み込む
- Weightを固定する
- ε-greedyではなくgreedyにActionを選択する
- 学習時と同じState生成・Reward / position処理を使用する
- episodeごとの損益を集計する

当時保存していた損益結果は、公開時にML/RLロジックを再確認した現在のコードと同一条件の評価ではないため、**現在の公開版の性能指標としては扱っていません**。  
また、README作成のためだけの再学習・再評価は行っていません。

## 主なファイル

| ファイル | 役割 |
|---|---|
| [`DQN.py`](./DQN.py) | 学習ループ、episode処理、Experience Replay、target同期、保存 / 再開 |
| [`TRAIN.py`](./TRAIN.py) | State生成、ε-greedy、Reward / position処理、TD更新 |
| [`Q_NETWORK.py`](./Q_NETWORK.py) | Q Network定義 |
| [`REMOTEMEMORY.py`](./REMOTEMEMORY.py) | Replay Memory |
| [`backtest.py`](./backtest.py) | 学習済みモデルのBacktest |

## 公開範囲・制約

- 公開しているのはML/RL部分のソースコードです。
- 学習・評価に使用した市場データは公開していません。
- 学習済みモデル / Weightは公開していません。
- データ取得・OCR等の前段システムは本Projectの公開範囲に含めていません。
- そのため、公開しているファイルだけでは当時の学習・Backtestをそのまま再現できません。
- 当時のコードでは乱数seedを固定しておらず、完全な再現性を保証する構成ではありません。
- 使用ライブラリのversionを固定した当時の実行環境も保存していません。

## 現在の位置づけ

このProjectは、現在開発中のRLシステムとは分けて、**過去に本人がDQN・TensorFlow custom training・RL環境設計を実際に実装していたことを確認できるHistorical Project**として公開しています。

公開にあたっては、個人情報・環境依存情報の除去や重要な不整合の確認など、公開に必要な整理を行っています。  
一方で、READMEを修正履歴としては扱わず、当時の設計・実装経験が分かる形を優先しています。
