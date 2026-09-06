# predict.py
import os
import sys
import json
import tensorflow as tf
from tensorflow import keras

# モジュールから関数とクラスをインポート
import data_utils
import model as model_lib

def load_config_for_predict(config_path):
    """予測に必要なconfigのみを読み込み、グローバル変数を設定する"""
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # data_utilsモジュールの変数を設定
    data_utils.IMG_HEIGHT = 40
    data_utils.IMG_WIDTH = 155
    # 予測時はダミーの値で良い
    data_utils.AUG_PARAMS = {
        "brightness_delta": 0.0, "contrast_lower": 1.0, 
        "contrast_upper": 1.0, "noise_stddev": 0.0
    }
    data_utils.SCALE_MAX = 1.0
    data_utils.SCALE_MIN = 1.0

    # modelモジュールの変数を設定
    model_lib.CNN_FILTERS = config['model_params']['cnn_filters']
    model_lib.RNN_UNITS = config['model_params']['rnn_units']
    model_lib.DROPOUT = 0.0 # 予測時はDropoutは自動でオフになるが、明示的に0
    model_lib.RANDOM_ROTATION = 0.0
    model_lib.RANDOM_TRANSLATION = [0.0, 0.0]
    model_lib.RANDOM_ZOOM = [0.0, 0.0]
    model_lib.L2 = 0.0 # 予測時はL2は影響しない

def main():
    # --- 0. 引数の確認 ---
    if len(sys.argv) != 4:
        print("エラー: 3つの引数が必要です。")
        print("使用法: python predict.py [config.jsonパス] [model.kerasパス] [画像パス]")
        return
    
    config_path = sys.argv[1]
    model_path = sys.argv[2]
    image_path = sys.argv[3]

    if not os.path.exists(config_path): print(f"エラー: {config_path} が見つかりません。"); return
    if not os.path.exists(model_path): print(f"エラー: {model_path} が見つかりません。"); return
    if not os.path.exists(image_path): print(f"エラー: {image_path} が見つかりません。"); return

    # --- 1. 設定とモデルの読み込み ---
    print(f"Loading config from {config_path}...")
    load_config_for_predict(config_path)
    
    keras.backend.clear_session()
    
    print(f"Loading model from {model_path}...")
    base_model = model_lib.build_base_model()
    # 評価モードでCTCTrainerを構築
    model = model_lib.CTCTrainer(base_model, all_train_labels_for_weighting=None)
    model.compile(optimizer=keras.optimizers.Adam())
    model.load_weights(model_path)
    
    # --- 2. 単一画像の前処理 ---
    print(f"Processing image: {image_path}...")
    # 学習時のis_training=Falseと同じイージーモードで前処理を実行
    img_tensor = data_utils.preprocess_image_easy(image_path, "")["image"]
    # バッチの次元を追加 (1, width, height, 1)
    img_tensor = tf.expand_dims(img_tensor, axis=0)

    # --- 3. 予測の実行 ---
    print("Running prediction...")
    # 予測はベースモデルで行う (CTCTrainerを通す必要はない)
    preds = base_model.predict(img_tensor)
    
    # --- 4. 結果のデコード ---
    decoded_texts = model_lib.ctc_decode(preds)
    
    print("\n--- Prediction Result ---")
    print(f"  Predicted Text: {decoded_texts[0]}")
    print("---------------------------")

if __name__ == "__main__":
    main()
