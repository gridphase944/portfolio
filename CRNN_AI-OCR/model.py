# model.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
import collections

# data_utilsから辞書情報をインポート
from data_utils import char_to_num, num_to_char, PAD_TOKEN, IMG_HEIGHT, IMG_WIDTH

# --- グローバル設定（configから読み込まれる）---
CNN_FILTERS = []
RNN_UNITS = []
DROPOUT = 0.0
L2 = 0.0

def build_base_model():
    """予測値(logits)を出力する責務を持つベースモデル"""
    input_img = keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1), name="image", dtype="float32")

    x = input_img 
    
    l2_reg = keras.regularizers.l2(L2)
    
    for filters in CNN_FILTERS:
        x = keras.layers.Conv2D(filters, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", kernel_regularizer=l2_reg)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
    
    w, h, f = x.shape[1], x.shape[2], x.shape[3]
    x = keras.layers.Reshape((w, h * f))(x)
    x = keras.layers.Dense(64, activation="relu", kernel_regularizer=l2_reg)(x)

    for units in RNN_UNITS:
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(units, return_sequences=True, dropout=DROPOUT, kernel_regularizer=l2_reg)
        )(x)

    x = keras.layers.Dense(len(char_to_num.get_vocabulary()), activation=None, name="output")(x)
    
    return keras.models.Model(inputs=input_img, outputs=x, name="crnn_base_simple")

class CTCTrainer(keras.Model):
    """CTC損失の計算と訓練ステップを管理する責務を持つ訓練用モデル"""
    def __init__(self, base_model, all_train_labels_for_weighting=None, **kwargs):
        super().__init__(**kwargs)
        self.model = base_model
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.val_loss_tracker = keras.metrics.Mean(name="val_loss")

        # float32で計算するように統一
        if all_train_labels_for_weighting is not None:
            print("\n--- クラス重みを計算します（訓練モード） ---")
            all_labels = "".join(all_train_labels_for_weighting) 
            char_counts = collections.Counter(all_labels)
            total_chars = sum(char_counts.values())
            class_weights = {char: np.log(total_chars / count).astype(np.float32) for char, count in char_counts.items()}
            
            self.loss_weight_table = np.ones(char_to_num.vocabulary_size(), dtype=np.float32)
            for char, weight in class_weights.items():
                idx = char_to_num(char)
                self.loss_weight_table[idx] = weight
            
            for char, count in sorted(char_counts.items()):
                weight = class_weights.get(char, 1.0)
                print(f"  '{char}': Count={count}, Weight={weight:.2f}")
            print("---------------------------------")
        else:
            print("\n--- クラス重みは計算しません（評価モード） ---")
            self.loss_weight_table = np.ones(char_to_num.vocabulary_size(), dtype=np.float32)

    def call(self, inputs):
        return self.model(inputs)

    def calculate_loss(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        logit_length = tf.cast(tf.shape(y_pred)[1], dtype="int64") * tf.ones(shape=(batch_len,), dtype="int64")
        # パディングトークン（PAD_TOKEN）のIDを取得
        pad_token_idx = char_to_num(PAD_TOKEN)
        
        # y_trueの中で、パディングではない要素（本物の文字）の数を数える
        # tf.not_equal でパディング以外の場所を True にし、それを合計して長さを得る
        label_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true, pad_token_idx), tf.int64), axis=1)

        loss = tf.nn.ctc_loss(
            labels=tf.cast(y_true, dtype=tf.int32),
            logits=y_pred,
            label_length=label_length,
            logit_length=logit_length,
            logits_time_major=False,
            blank_index=-1
        )
        
        # 各ラベルの文字重みを取得
        label_weights = tf.gather(self.loss_weight_table, tf.cast(y_true, dtype=tf.int32))
        mask = tf.cast(tf.not_equal(y_true, pad_token_idx), dtype=tf.float32)
        
        label_weights *= mask
        # 0除算を避けるため、文字数0を1として扱う
        safe_label_length = tf.cast(label_length, tf.float32)
        safe_label_length = tf.where(tf.equal(safe_label_length, 0.0), 1.0, safe_label_length)
        
        avg_weights = tf.reduce_sum(label_weights, axis=1) / safe_label_length
        
        weighted_loss = loss * avg_weights
        return tf.reduce_mean(weighted_loss)

    def train_step(self, data):
        x, y = data['image'], data['label']
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.calculate_loss(y, y_pred)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x, y = data['image'], data['label']
        y_pred = self.model(x, training=False)
        loss = self.calculate_loss(y, y_pred)
        self.val_loss_tracker.update_state(loss)
        return {"loss": self.val_loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.val_loss_tracker]

def ctc_decode(y_pred):
    y_pred_transposed = tf.transpose(y_pred, perm=[1, 0, 2])
    input_len = tf.ones(tf.shape(y_pred)[0], dtype=tf.int32) * tf.shape(y_pred_transposed)[0]
    decoded_sparse, _ = tf.nn.ctc_beam_search_decoder(
        inputs=y_pred_transposed,
        sequence_length=input_len,
        beam_width=10,
        top_paths=1
    )
    decoded_dense = tf.sparse.to_dense(decoded_sparse[0], default_value=-1)
    return [tf.strings.reduce_join(num_to_char(tf.gather(res, tf.where(res != -1)))).numpy().decode("utf-8") for res in decoded_dense]

class PredictionCallback(keras.callbacks.Callback):
    def __init__(self, base_model, validation_dataset):
        super().__init__()
        self.base_model = base_model
        self.validation_dataset = validation_dataset
        self.padding_value = char_to_num(PAD_TOKEN)
    def on_epoch_end(self, epoch, logs=None):
        for batch in self.validation_dataset.take(1):
            preds = self.base_model.predict(batch["image"], verbose=0)
            pred_texts = ctc_decode(preds)
            orig_texts = []
            for l in batch["label"]:
                label = tf.gather(l, tf.where(l != self.padding_value))
                text = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
                orig_texts.append(text)
            print("-" * 60)
            print(f"Epoch {epoch+1} Prediction Samples:")
            for i in range(min(5, len(pred_texts))):
                print(f"  True: {orig_texts[i]:<20} | Pred: {pred_texts[i]}")
            print("-" * 60)
