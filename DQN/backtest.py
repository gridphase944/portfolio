import Q_NETWORK
import TRAIN
import tensorflow as tf
import numpy as np
import glob
import time
import os

path_name=os.path.dirname(os.path.abspath(__file__))
episode_dir=path_name+"/OCR_Result_Numpy_Backtest"#backtest専用エピソードフォルダ
episode_dir = sorted(glob.glob(episode_dir+"/*"))
train=TRAIN.Train()#トレーニングclass
gainCurve=np.array([],dtype=int)#利益曲線
#共通パラメータ
alpha=0.0025#学習率
fullboard_count=5#現在値から何個分の売り買いデータを取るか決める。0.5円間隔で表示されている場合は＝4で2円分の情報を取る

#モデルの初期化
print("modelディレクトリ以下から使用するモデル名または識別子を入力してください:",end="")
num=input()
f=path_name+"/model/"+num
NN=tf.keras.models.load_model(f)
main_network =Q_NETWORK.QNetwork(alpha)
state = np.zeros(46)
state=state.reshape(1,-1)
#set_weights前にnetworkをbuild
main_network(state).numpy()
main_network.set_weights(NN.get_weights())
#テスト開始
for a in range(0,len(episode_dir),+1):#エピソード
    start = time.time()
    #各エピソードの初期化
    totalGain=0#各エピソードでの合計利益
    purchase_price=0#買った時の株価を保存
    #状態の読み出し
    #歩み値
    value_OCR=np.load(episode_dir[a]+"/value.npy")#約定値
    #現在値
    nowPriceIndex_OCR=np.load(episode_dir[a]+"/nowPriceIndex.npy")#現在値インデックス
    price_OCR=np.load(episode_dir[a]+"/price.npy")#現在値
    #全板
    buy_OCR=np.load(episode_dir[a]+"/buy.npy")#買い
    sell_OCR=np.load(episode_dir[a]+"/sell.npy")#売り
    if price_OCR.shape[0]<2:
        continue
    terminal_state_flag=True
    for terminal_b in [price_OCR.shape[0]-2,price_OCR.shape[0]-1]:
        if buy_OCR.shape[1]<nowPriceIndex_OCR[terminal_b]+fullboard_count:
            terminal_state_flag=False
        elif nowPriceIndex_OCR[terminal_b]-fullboard_count+1<0:
            terminal_state_flag=False
    if not terminal_state_flag:
        continue
    if nowPriceIndex_OCR[0]<1:
        #閾値計算で末尾要素を誤参照しないよう、初期indexが0のエピソードはスキップする
        continue
    #fullboard_count=5では、売り板を5件取得するため現在値indexが4以上必要
    #売買価格の閾値（現在値が購入値±閾値を超えていない場合は売却しない）
    buysell_threshold=(price_OCR[0][nowPriceIndex_OCR[0]-1]-price_OCR[0][nowPriceIndex_OCR[0]])*3
    for b in range(price_OCR.shape[0]):#ステップ
        #状態の作成
        #全板買い売りを必要分のみ取り出し
        new_buy_OCR=np.zeros((fullboard_count),dtype=int)
        if buy_OCR.shape[1]<nowPriceIndex_OCR[b]+fullboard_count:
            continue
        else:
            ndim=0
            for aa in range(int(nowPriceIndex_OCR[b]), int(nowPriceIndex_OCR[b]+fullboard_count),+1):
                new_buy_OCR[ndim]=buy_OCR[b][aa]
                ndim+=1
        #売り
        new_sell_OCR=np.zeros((fullboard_count),dtype=int)
        if nowPriceIndex_OCR[b]-fullboard_count+1<0:
            #要素数オーバー
            continue
        else:
            ndim=0
            for aa in range(int(nowPriceIndex_OCR[b]),int(nowPriceIndex_OCR[b]-fullboard_count),-1):
                new_sell_OCR[ndim]=sell_OCR[b][aa]
                ndim+=1
        #状態作成
        state,scaling=train.create_state(
            value_OCR[b],
            price_OCR[b][nowPriceIndex_OCR[b]],
            new_buy_OCR,
            new_sell_OCR,
            purchase_price,
            buysell_threshold
        )
        if b ==price_OCR.shape[0]-1:
            #N-1はterminal stateのため、行動を選択しない
            break
        #Q値を計算
        values=main_network(state).numpy()
        #最終decision step（N-2）はAction Maskで未決済ポジションを解消する
        if b ==price_OCR.shape[0]-1 -1:
            if purchase_price==0:
                valid_actions=[0]
            elif purchase_price>0:
                valid_actions=[1]
            elif purchase_price<0:
                valid_actions=[2]
            masked_values=np.full_like(values,-np.inf,dtype=float)
            masked_values[:,valid_actions]=values[:,valid_actions]
            action=np.argmax(masked_values)
        else:
            action=np.argmax(values)
        #報酬の設定
        reward,purchase_price,totalGain,_=train.get_reward(action,price_OCR[b][nowPriceIndex_OCR[b]],purchase_price,scaling,totalGain,buysell_threshold)
        #最終decision stepでは閾値以下でも未決済ポジションを精算する
        if b ==price_OCR.shape[0]-1 -1 and purchase_price !=0:
            if purchase_price>0:
                gain=price_OCR[b][nowPriceIndex_OCR[b]]-purchase_price#評価額
            elif purchase_price<0:
                gain=-(price_OCR[b][nowPriceIndex_OCR[b]]+purchase_price)
            totalGain+=gain*100
            purchase_price=0
            if gain>0:
                reward=(gain/scaling)*2
            else:
                reward=gain/scaling
    print(totalGain,end="")
    gainCurve=np.append(gainCurve,totalGain)
    end = time.time()
    print("処理時間:",end-start)
print("終了！")
print(f)
print("最終利益：",np.sum(gainCurve))
if gainCurve.size>0:
    print("最大利益：",np.max(gainCurve))
    print("最小利益：",np.min(gainCurve))
input()
