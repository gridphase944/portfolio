import tensorflow as tf
import numpy as np
class Train():
    #状態作成
    def create_state(self,value_OCR,nowPrice_OCR,buy_OCR,sell_OCR,purchase_price,buysell_threshold):
        #現在値からスケーリングを計算
        match nowPrice_OCR:
            case nowPrice_OCR if nowPrice_OCR<100:
                scaling=3#報酬の部分で価格差により値上がりできる率が違うのでスケーリング
            case nowPrice_OCR if nowPrice_OCR<200:
                scaling=5
            case nowPrice_OCR if nowPrice_OCR<500:
                scaling=8
            case nowPrice_OCR if nowPrice_OCR<700:
                scaling=10
            case nowPrice_OCR if nowPrice_OCR<1000:
                scaling=15
            case nowPrice_OCR if nowPrice_OCR<1500:
                scaling=30
            case nowPrice_OCR if nowPrice_OCR<2000:
                scaling=40
            case nowPrice_OCR if nowPrice_OCR<3000:
                scaling=50
            case nowPrice_OCR if nowPrice_OCR<5000:
                scaling=70
            case nowPrice_OCR if nowPrice_OCR<7000:
                scaling=100
            case nowPrice_OCR if nowPrice_OCR<10000:
                scaling=150
            case nowPrice_OCR if nowPrice_OCR<15000:
                scaling=300
            case nowPrice_OCR if nowPrice_OCR<20000:
                scaling=400
            case nowPrice_OCR if nowPrice_OCR<30000:
                scaling=500
            case nowPrice_OCR if nowPrice_OCR<50000:
                scaling=700
            case _:
                scaling=1000
        # 約定値：最新の約定値を基準に増減を計算しスケーリングする
        update_value_OCR=np.zeros((len(value_OCR)-1),dtype=float)#-1は最後の値を基準にして過去からの価格変動を入力する為、最後の要素は必ず0になるから不要
        for aa in range(len(update_value_OCR)):
            #まず約定値が現在値より5倍以上離れているなら/10する。(OCRミスによる)
            if value_OCR[aa]>nowPrice_OCR*5:
                value_OCR[aa]=value_OCR[aa]/10
            scalingvalue=(value_OCR[aa]-value_OCR[len(value_OCR)-1])/(scaling*10)#*10はスケーリングが報酬と状態で重みづけが違うから（状態では-1から1の範囲でスケーリングしたい）
            #-1から1の範囲に収める
            if -1>scalingvalue:
                scalingvalue=-1
            elif 1<scalingvalue:
                scalingvalue=1
            update_value_OCR[aa]=scalingvalue
        update_buy_OCR=np.zeros((len(buy_OCR)),dtype=float)
        update_sell_OCR=np.zeros((len(sell_OCR)),dtype=float)
        # 必要範囲に切り出された全板の買い・売りデータをスケーリングする
        #買い
        for aa in range(len(buy_OCR)):
            scalingvalue=buy_OCR[aa]/10000
            #1を超えていたら強制的に1にする
            if 1<scalingvalue:
                scalingvalue=1
            update_buy_OCR[aa]=scalingvalue
        #売り
        for aa in range(len(sell_OCR)):
            scalingvalue=sell_OCR[aa]/10000
            #1を超えていたら強制的に1にする
            if 1<scalingvalue:
                scalingvalue=1
            update_sell_OCR[aa]=scalingvalue
        #ポジション状態を-1、0、1で表す
        if purchase_price==0:#ポジションを持っていない
            possession=np.array([0])
            rate=np.array([0])
        elif purchase_price>0:#買い
            possession=np.array([1])
            gain=nowPrice_OCR-purchase_price
            #閾値以下の損益なら0にする
            if gain> buysell_threshold or gain<-buysell_threshold:
                rate=gain/(scaling*10)
                if -1>rate:
                    rate=-1
                elif 1<rate:
                    rate=1
            else:
                rate=0
            rate=np.array([rate])
        elif purchase_price<0:#空売り
            possession=np.array([-1])
            gain=-(nowPrice_OCR+purchase_price)
            #閾値以下の損益なら0にする
            if gain> buysell_threshold or gain<-buysell_threshold:
                rate=gain/(scaling*10)
                if -1>rate:
                    rate=-1
                elif 1<rate:
                    rate=1
            else:
                rate=0
            rate=np.array([rate])
        if rate.ndim==2:
            rate=rate.flatten()
        #状態の作成
        state=np.concatenate([
            update_value_OCR,
            update_buy_OCR,
            update_sell_OCR,
            possession,
            rate
        ])
        state=state.reshape(1,-1)
        return state,scaling
    #ε-greedy法（valid_actions指定時はAction Maskを適用）
    def epsilon_greedy(self,main_network,state,action_num,epsilon,valid_actions=None):
        values=main_network(state).numpy()
        if valid_actions is None:
            valid_actions=np.arange(action_num)
        if np.random.uniform() < epsilon:
            action=np.random.choice(valid_actions)
        else:
            masked_values=np.full_like(values,-np.inf,dtype=float)
            masked_values[:,valid_actions]=values[:,valid_actions]
            action = np.argmax(masked_values)
        return values,action
    #報酬の設定
    def get_reward(
            self,
            action,
            nowPrice_OCR,
            purchase_price,
            scaling,
            totalGain,
            buysell_threshold
            ):
        addStateFlag=True
        #行動[無し]の場合
        if action==0 :
            if purchase_price==0:#持っていない
                pass
            elif purchase_price>0:#買いから入ってる
                gain=nowPrice_OCR-purchase_price#評価額
                if gain> buysell_threshold or gain<-buysell_threshold:
                    pass
                else:
                    addStateFlag=False
            elif purchase_price<0:#売りから入ってる
                gain=-(nowPrice_OCR+purchase_price)#評価額
                if gain> buysell_threshold or gain<-buysell_threshold:
                    pass
                else:
                    addStateFlag=False
            reward=0
        #行動[売る]の場合
        elif action==1:
            #「空売り」購入
            if purchase_price==0:
                reward=0
                purchase_price=-nowPrice_OCR#購入価格
            #「買い」売却
            elif purchase_price>0:
                gain=nowPrice_OCR-purchase_price#評価額
                if gain> buysell_threshold or gain<-buysell_threshold:#閾値を超える損益
                    totalGain+=gain*100
                    purchase_price=0
                    #損益により報酬を変える
                    if gain> buysell_threshold:#利益
                        reward=(gain/scaling)*2#利益が出たら2倍の報酬を与える
                    else:#損失
                        reward=gain/scaling
                #閾値以下の価格変動
                else:
                    addStateFlag=False
                    reward=0
            #「空売り」追加購入
            elif purchase_price<0:
                addStateFlag=False
                reward=0
        #行動[買う]の場合
        elif action==2:
            #「買い」購入
            if purchase_price==0:
                reward=0
                purchase_price=nowPrice_OCR#購入価格
            #「買い」追加購入
            elif purchase_price>0:
                addStateFlag=False
                reward=0
            elif purchase_price<0:#「空売り」売却
                gain=-(nowPrice_OCR+purchase_price)#評価額
                if gain> buysell_threshold or gain<-buysell_threshold:#閾値を超える損益
                    totalGain+=gain*100
                    purchase_price=0
                    #損益により報酬を変える
                    if gain> buysell_threshold:#利益
                        reward=(gain/scaling)*2#利益が出たら2倍の報酬を与える
                    else:#損失
                        reward=gain/scaling
                #閾値以下の価格変動
                else:
                    addStateFlag=False
                    reward=0
        return reward,purchase_price,totalGain,addStateFlag
    #重みの更新
    def update_weights(self, main_network,target_q_network,now_state, action, reward , next_state , done , gamma):
        next_q_value = target_q_network(next_state)
        max_next_q_value = tf.reduce_max(next_q_value, axis=1)
        done = tf.cast(done, max_next_q_value.dtype)
        #terminal transitionではbootstrapしないTD targetを使用する
        target_q_value = reward + (1 - done) * gamma * max_next_q_value
        with tf.GradientTape() as tape:
            q_value = main_network(now_state)
            #選択したactionのQ値だけをHuber LossでTD targetへ近づける
            current_q_value = tf.gather(q_value,action,axis=1)
            loss = main_network.loss(target_q_value, current_q_value)
        grads = tape.gradient(loss, main_network.trainable_variables)
        main_network.optimizer.apply_gradients(zip(grads, main_network.trainable_variables))
        return main_network,loss.numpy()


