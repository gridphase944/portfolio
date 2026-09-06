import Q_NETWORK as Q_NETWORK
import TRAIN as TRAIN
import REMOTEMEMORY
import datetime
import os
import glob
import time
import tensorflow as tf
import numpy as np

try:
    a=-1
    b=-1
    filename=None
    epsilon_Down=0
    """
    空売りも学習させたモデル
    最終decision step（N-2）で未決済ポジションを精算する
    """
    #=======================================================================================================
    #初期設定
    #=======================================================================================================
    #日付取得
    now=datetime.datetime.now()
    dt_now=now.strftime('%Y%m%d_%H%M%S')
    train=TRAIN.Train()#トレーニングclass
    remotememory=REMOTEMEMORY.RemoteMemory()
    rewardCurve=np.array([],dtype=int)#報酬曲線(合計報酬)
    lossCurve=np.array([])#損失曲線
    gainCurve=np.array([],dtype=int)#利益曲線
    # NumPy形式のエピソードデータが保存されているディレクトリを指定
    path_name=os.path.dirname(os.path.abspath(__file__))
    filename=path_name+"/"+dt_now
    episode_dir = path_name+"/OCR_Result_Numpy"#エピソードが保存されているディレクトリ
    episode_dir = sorted(glob.glob(episode_dir+"/*"))# ディレクトリ内の全てのフォルダ名を取得してリスト化しソート
    #保存先生成
    print("保存フォルダを生成しますか？[はい:1][いいえ:2]",end="")
    v=input()
    if v=="1":
        os.mkdir(path_name+"/"+dt_now)#保存フォルダ生成
        print("保存先フォルダを生成しました！")
    else:
        print("保存先フォルダを作成しません。1エピソード終了後エラーが発生します")
    #共通パラメータ
    gamma=0.99#割引率（0に近い値：即時報酬（現在の報酬）のみ重視、1に近い値：将来の報酬も重視）
    alpha=0.0025#学習率
    fullboard_count=5#現在値から何個分の売り買いデータを取るか決める。0.5円間隔で表示されている場合は＝4で2円分の情報を取る
    #専用パラメータ
    action_dim = 3  # 行動の次元数
    batch_size=30#RemoteMemoryから1回にサンプリングする経験数
    epsilon_Down=0#探索率の減衰カウント
    weight_copy_count=0#target network同期用カウント
    #モデルの初期化
    main_network =Q_NETWORK.QNetwork(alpha)
    target_q_network =Q_NETWORK.QNetwork(alpha)
    print("過去のモデルを使用して学習しますか？[はい:1][いいえ:2]",end="")
    v=input()
    if v=="1":
        NN=tf.keras.models.load_model(path_name+"/TrainingModel/model.keras")
        print("過去のモデルを使用して学習します。続行するにはEnterキーを押してください",end="")
        input()
        state = np.zeros(46)
        state=state.reshape(1,-1)
        #set_weights前にtarget networkをbuild
        target_q_network(state).numpy()
        main_network=NN
        target_q_network.set_weights(main_network.get_weights())
        #リモートメモリーの呼び出し
        epsilon_Down=np.load(path_name+"/TrainingModel/epsilon_Down.npy")
        epsilon_Down=epsilon_Down[0]
        state_item=np.load(path_name+"/TrainingModel/state_item.npy")
        action_item=np.load(path_name+"/TrainingModel/action_item.npy")
        reward_item=np.load(path_name+"/TrainingModel/reward_item.npy")
        next_state_item=np.load(path_name+"/TrainingModel/next_state_item.npy")
        done_item=np.load(path_name+"/TrainingModel/done_item.npy")
        #格納
        for a in range(len(action_item)):
            remotememory.add(state_item[a].reshape(1,-1),action_item[a],reward_item[a],next_state_item[a].reshape(1,-1),done_item[a])
    else:
        print("新規で学習を開始します。続行するにはEnterキーを押してください",end="")
        input()
        state = np.zeros(46)
        state=state.reshape(1,-1)
        #main / target networkをbuild
        main_network(state).numpy()
        target_q_network(state).numpy()
        target_q_network.set_weights(main_network.get_weights())
    print("何番目のエピソードからスタートしますか")
    startEpisode=int(input())

    #行動：0[無し]1[売る]2[買う]

    for a in range(startEpisode,len(episode_dir)):#エピソード
        #各エピソードの初期化
        totalReward=0#各エピソードでの合計報酬
        totalGain=0#各エピソードでの合計利益
        purchase_price=0#買った時の株価を保存
        now_state=None#現在の状態を格納
        now_action=None#現在の行動を格納
        now_reward=None#現在の報酬を格納
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
            print("e:(",a,"/",len(episode_dir),"),s:(",b,"/",price_OCR.shape[0]-1,")",end="")
            start = time.time()
            #最終decision step（N-2）は閾値によるraw step skipを適用しない
            if not b ==price_OCR.shape[0]-1 -1:
                #ポジション持っていて閾値以下の利益なら次に進む
                if purchase_price>0:#買い
                    gain=price_OCR[b][nowPriceIndex_OCR[b]]-purchase_price#評価額
                    if gain> buysell_threshold or gain<-buysell_threshold:
                        pass#利益が大きい
                    else:
                        end = time.time()
                        print(",time:",round(end-start,1))
                        continue
                elif purchase_price<0:#空売り
                    gain=-(price_OCR[b][nowPriceIndex_OCR[b]]+purchase_price)#評価額
                    if gain> buysell_threshold or gain<-buysell_threshold:
                        pass#利益が大きい
                    else:
                        end = time.time()
                        print(",time:",round(end-start,1))
                        continue
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
                #N-1をterminal stateとして保存し、bootstrapを無効にする
                terminal_state=state
                remotememory.add(now_state, now_action, now_reward, terminal_state, True)
                break
            #行動の選択
            epsilon = 1 * (0.9999 ** epsilon_Down)#Experience Replay実行回数に応じて探索率を下げる
            valid_actions=None
            #最終decision step（N-2）はAction Maskで未決済ポジションを解消する
            if b ==price_OCR.shape[0]-1 -1:
                if purchase_price==0:
                    valid_actions=[0]
                elif purchase_price>0:
                    valid_actions=[1]
                elif purchase_price<0:
                    valid_actions=[2]
            values,action=train.epsilon_greedy(main_network,state,action_dim,epsilon,valid_actions)
            print(",Qvar:[",round(values[0][0],3),round(values[0][1],3),round(values[0][2],3),"]",end="")
            #報酬の設定
            reward,purchase_price,totalGain,addStateFlag=train.get_reward(action,price_OCR[b][nowPriceIndex_OCR[b]],purchase_price,scaling,totalGain,buysell_threshold)
            #最終decision stepでは閾値以下でも未決済ポジションを精算する
            if b ==price_OCR.shape[0]-1 -1 and purchase_price !=0:
                if purchase_price>0:
                    gain=price_OCR[b][nowPriceIndex_OCR[b]]-purchase_price#評価額
                elif  purchase_price<0:
                    gain=-(price_OCR[b][nowPriceIndex_OCR[b]]+purchase_price)#評価額
                totalGain+=gain*100
                purchase_price=0
                if gain>0:
                    reward=(gain/scaling)*2
                else:
                    reward=gain/scaling
            #各エピソードでの合計報酬
            totalReward+=reward
            #行動後のポジション確認
            if purchase_price==0:
                poss=0
            elif purchase_price>0:
                poss=1
            elif purchase_price<0:
                poss=-1
            print(",Tgain:",totalGain,end="")
            print(",Treward:",round(totalReward,3),end="")
            print(",poss:",poss,end="")
            # 経験バッファに追加
            #閾値以下の損益で取引が成立しなかった状態は原則として保存しない
            #最終decision stepはterminal transitionへつなぐため保存対象にする
            if b ==price_OCR.shape[0]-1 -1:
                addStateFlag=True
            ExperienceReplayFlag=False#このstepでExperience Replayを実行するか
            if addStateFlag:
                if now_state is None:
                    now_state=state#現在の状態に状態を入力
                    now_action=action#現在の行動に行動を入力
                    now_reward=reward#現在の報酬に報酬を入力
                else:
                    next_state=state#次の状態に状態を入力
                    remotememory.add(now_state, now_action, now_reward, next_state, False) #経験バッファに追加
                    ExperienceReplayFlag=True
                    now_state=state#次の状態を現在の状態に入力
                    now_action=action#次の行動を現在の行動に入力
                    now_reward=reward
            #経験数がbatch_sizeを超えたらExperience Replayを実行
            if len(remotememory.memory)>batch_size and ExperienceReplayFlag:
                #Experience Replayを実行する場合のみ重みの同期カウントと探索率の減衰カウントを加算
                weight_copy_count+=1
                epsilon_Down+=1
                batch = remotememory.sample(batch_size)
                for d in batch:
                    states_remotememory=d[0]
                    actions_remotememory=d[1]
                    rewards_remotememory=d[2]
                    next_states_remotememory=d[3]
                    dones_remotememory=d[4]
                    #選択したactionのQ値を更新
                    main_network,loss=train.update_weights(
                        main_network,
                        target_q_network,
                        states_remotememory,
                        actions_remotememory,
                        rewards_remotememory,
                        next_states_remotememory,
                        dones_remotememory,
                        gamma
                    )
                    lossCurve=np.append(lossCurve,loss)
                print(",loss:",round(loss,5),end="")
            #重みの同期
            if weight_copy_count>300:
                target_q_network.set_weights(main_network.get_weights())
                weight_copy_count=0

            end = time.time()
            print(",time:",round(end-start,1))
        #エピソード終了処理（主にデータの保存）
        rewardCurve=np.append(rewardCurve,totalReward)#報酬
        gainCurve=np.append(gainCurve,totalGain)#利益
        #最後に処理したエピソード情報（進捗情報）を上書き保存
        with open(filename+"/episord.txt",'w', encoding='utf-8') as f:
            f.write("エピソード:"+str(a)+"終了(エピソードファイル名："+episode_dir[a]+")")
    #学習終了処理
    #リモートメモリーのデータを追加学習の為に保存する
    state_item=np.empty((0,46))
    action_item=np.empty((0),dtype=int)
    reward_item=np.empty((0))
    next_state_item=np.empty((0,46))
    done_item=np.empty((0),dtype=bool)
    flag=True
    while True:
        item=remotememory.get()
        if item is None:
            break
        else:
            #初回のみ初期化処理を入れる
            if flag:
                flag=False
                state_item=np.empty((0,item[0].shape[1]))
                action_item=np.empty((0),dtype=int)
                reward_item=np.empty((0))
                next_state_item=np.empty((0,item[0].shape[1]))
                done_item=np.empty((0),dtype=bool)
            state_item=np.append(state_item,item[0],axis=0)
            action_item=np.append(action_item,item[1])
            reward_item=np.append(reward_item,item[2])
            next_state_item=np.append(next_state_item,item[3],axis=0)
            done_item=np.append(done_item,item[4])
    np.save(filename+"/state_item.npy",state_item)
    np.save(filename+"/action_item.npy",action_item)
    np.save(filename+"/reward_item.npy",reward_item)
    np.save(filename+"/next_state_item.npy",next_state_item)
    np.save(filename+"/done_item.npy",done_item)
    arr=np.array([epsilon_Down],dtype=int)
    np.save(filename+"/epsilon_Down.npy",arr)
    np.save(filename+"/lossCurve.npy",lossCurve)#損失関数の結果
    main_network.save(filename+"/model.keras", save_format='keras')#学習終了時のメインネットワークを保存
    np.save(filename+"/rewardCurve.npy",rewardCurve)#合計報酬の結果
    np.save(filename+"/gainCurve.npy",gainCurve)#合計利益の結果
except Exception as e:
    print("エラーメッセージ")
    print(e)
    if filename is not None and os.path.isdir(filename):
        with open(filename+"/NG.txt",'w', encoding='utf-8') as f:
            f.write("エピソード:"+str(a)+",ステップ："+str(b)+",エラー内容："+str(e)+"\n")
        if remotememory is not None:
            state_item=np.empty((0,46))
            action_item=np.empty((0),dtype=int)
            reward_item=np.empty((0))
            next_state_item=np.empty((0,46))
            done_item=np.empty((0),dtype=bool)
            flag=True
            while True:
                item=remotememory.get()
                if item is None:
                    break
                else:
                    #初回のみ初期化処理を入れる
                    if flag:
                        flag=False
                        state_item=np.empty((0,item[0].shape[1]))
                        action_item=np.empty((0),dtype=int)
                        reward_item=np.empty((0))
                        next_state_item=np.empty((0,item[0].shape[1]))
                        done_item=np.empty((0),dtype=bool)
                    state_item=np.append(state_item,item[0],axis=0)
                    action_item=np.append(action_item,item[1])
                    reward_item=np.append(reward_item,item[2])
                    next_state_item=np.append(next_state_item,item[3],axis=0)
                    done_item=np.append(done_item,item[4])
            np.save(filename+"/state_item.npy",state_item)
            np.save(filename+"/action_item.npy",action_item)
            np.save(filename+"/reward_item.npy",reward_item)
            np.save(filename+"/next_state_item.npy",next_state_item)
            np.save(filename+"/done_item.npy",done_item)
            arr=np.array([epsilon_Down],dtype=int)
            np.save(filename+"/epsilon_Down.npy",arr)
