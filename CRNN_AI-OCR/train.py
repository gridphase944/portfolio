# train.py
import os
import json
import traceback
from datetime import datetime
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import difflib # 文字列類似度計算用

# モジュールから関数とクラスをインポート
import data_utils
import model as model_lib

def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    
    data_utils.IMG_HEIGHT = 40
    data_utils.IMG_WIDTH = 155
    data_utils.AUG_PARAMS = config['augmentation_params']
    data_utils.SCALE_MAX = config["image_processing"]["scale_max"]
    data_utils.SCALE_MIN = config["image_processing"]["scale_min"]
    data_utils.ROTATION_RANGE = config["random_params"]["rotation"]

    model_lib.CNN_FILTERS = config['model_params']['cnn_filters']
    model_lib.RNN_UNITS = config['model_params']['rnn_units']
    model_lib.DROPOUT = config['model_params']['dropout']
    model_lib.L2 = config["L2"]
    
    return config

def calculate_sequence_similarity(true_text, pred_text):
    """SequenceMatcher.ratio()による文字列類似度を計算"""
    matcher = difflib.SequenceMatcher(None, true_text, pred_text)
    return matcher.ratio()

def main():
    # --- 0. 設定の読み込み ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("config名を指定してください (例: config_v1):", end="")
    config_name = input()
    config_path = os.path.join(current_dir, f"{config_name}.json")
    
    if not os.path.exists(config_path):
        print(f"エラー: {config_path} が見つかりません。")
        return
        
    config = load_config(config_path)
    config_basename = config_name
    
    # --- グローバル設定の読み込み ---
    LEARNING_RATE = config['learning_rate']
    EPOCHS = config['epochs']
    BATCH_SIZE = config['batch_size']

    # --- パス設定 ---
    try:
        with open(os.path.join(current_dir, "TrainImagePath.txt"), 'r', encoding='utf-8') as f:
            content = f.read().strip()
    except FileNotFoundError:
        print("エラー: TrainImagePath.txt が見つかりません。")
        return

    TRAIN_IMG_DIR = os.path.join(content, "OCRTrainImage")
    TEST_IMG_DIR = os.path.join(content, "OCRTestImage")
    MODEL_SAVE_PATH = os.path.join(current_dir, f"{config_basename}.keras")
    ERROR_LOG_FILE = os.path.join(current_dir, "training_error_log.txt")
    GRAPH_SAVE_PATH = os.path.join(current_dir, f"{config_basename}_graph.png")
    ERROR_LIST_PATH = os.path.join(current_dir, f"{config_basename}_errors.txt") # エラーリストのパス

    # =======================================================
    # 終了メッセージを必ず表示するため、全体をtry...finallyで囲む
    # =======================================================
    try:
        # --- ステージ1：データセットの準備 ---
        print("Loading Stage 1 data (OCRTrainImage)...")
        all_train_paths, all_train_labels = data_utils.load_image_paths_and_labels(TRAIN_IMG_DIR)
        print("Loading Stage 2 data (OCRTestImage)...")
        final_test_paths, final_test_labels = data_utils.load_image_paths_and_labels(TEST_IMG_DIR)
        
        val_split_size = 0.1
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            all_train_paths, all_train_labels, test_size=val_split_size, random_state=42
        )
        
        print(f"\nTotal training images (Stage 1): {len(train_paths)}")
        print(f"Total validation images (Stage 1): {len(val_paths)}")
        print(f"Total test images (Stage 2): {len(final_test_paths)}")

        train_dataset = data_utils.create_dataset(train_paths, train_labels, BATCH_SIZE, is_training=True)
        validation_dataset = data_utils.create_dataset(val_paths, val_labels, BATCH_SIZE, is_training=False)

        # --- ステージ1：モデルの学習 ---
        base_model = model_lib.build_base_model()
        training_model = model_lib.CTCTrainer(base_model, all_train_labels_for_weighting=train_labels)
        training_model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))
        
        checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath=MODEL_SAVE_PATH, save_best_only=True, monitor="val_loss", mode="min", verbose=1)
        reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
        prediction_cb = model_lib.PredictionCallback(base_model, validation_dataset)
        early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=False)

        history = None
        final_test_loss_value = None
        final_accuracy = None
        final_sequence_similarity = None

        print("\nStarting Stage 1 training...")
        try:
            history = training_model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=EPOCHS,
                callbacks=[checkpoint_cb, prediction_cb, reduce_lr_cb, early_stopping_cb]
            )
            print("\nStage 1 training completed.")
            
        except Exception as e:
            print(f"\n{'*'*20} An error occurred during training {'*'*20}")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            error_details = traceback.format_exc()
            with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"---\nTimestamp: {timestamp}\n{error_details}\n")
            print(f"Error details have been saved to {ERROR_LOG_FILE}")
            return
        
        # --- ステージ2：最終試験と正解率計算 ---
        print("\n" + "="*50)
        print("Starting Stage 2: Final Evaluation on unknown data")
        print("="*50)
        
        if not os.path.exists(MODEL_SAVE_PATH):
            print(f"エラー: モデルファイル {MODEL_SAVE_PATH} が見つかりません。")
            return

        try:
            keras.backend.clear_session()
            print(f"Loading best model from {MODEL_SAVE_PATH}...")
            
            best_base_model = model_lib.build_base_model()
            
            # 入力shapeを指定してビルド
            trainer_input_shape = {
                "image": (None, data_utils.IMG_WIDTH, data_utils.IMG_HEIGHT, 1),
                "label": (None, None)
            }
            evaluation_model = model_lib.CTCTrainer(best_base_model, all_train_labels_for_weighting=None)
            evaluation_model.compile(optimizer=keras.optimizers.Adam()) 
            evaluation_model.build(input_shape=trainer_input_shape)
            best_base_model.build(input_shape=(None, data_utils.IMG_WIDTH, data_utils.IMG_HEIGHT, 1))
            
            # 重み読み込み
            evaluation_model.load_weights(MODEL_SAVE_PATH)
            
            # データセット作成
            final_test_dataset = data_utils.create_dataset(final_test_paths, final_test_labels, BATCH_SIZE, is_training=False)

            # 1. Lossの評価
            print("Calculating Test Loss...")
            final_results = evaluation_model.evaluate(final_test_dataset, verbose=0)
            final_test_loss_value = final_results.numpy()

            # 2. 正解率の計算
            print("Calculating Accuracy...")
            correct_count = 0
            total_sequence_similarity_sum = 0
            error_list = []
            total_samples = len(final_test_paths)
            
            processed_count = 0
            for batch in final_test_dataset:
                images = batch['image']
                preds = best_base_model.predict(images, verbose=0)
                pred_texts = model_lib.ctc_decode(preds)
                
                current_batch_size = len(pred_texts)
                start_idx = processed_count
                end_idx = processed_count + current_batch_size
                true_labels_batch = final_test_labels[start_idx:end_idx]
                true_paths_batch = final_test_paths[start_idx:end_idx]

                for i in range(current_batch_size):
                    true_text = true_labels_batch[i]
                    pred_text = pred_texts[i]
                    
                    if true_text == pred_text:
                        correct_count += 1
                    else:
                        error_list.append({
                            "path": os.path.basename(true_paths_batch[i]),
                            "true": true_text,
                            "pred": pred_text
                        })
                    total_sequence_similarity_sum += calculate_sequence_similarity(true_text, pred_text)
                processed_count += current_batch_size

            final_accuracy = (correct_count / total_samples) * 100
            final_sequence_similarity = (total_sequence_similarity_sum / total_samples) * 100
            
            print("\n--- Final Test Results ---")
            print(f"  Test Loss: {final_test_loss_value:.4f}")
            print(f"  Accuracy : {final_accuracy:.2f}%")
            print(f"  Sequence Similarity: {final_sequence_similarity:.2f}%")
            print("----------------------------")

            # 間違いリストの保存
            with open(ERROR_LIST_PATH, "w", encoding="utf-8") as f:
                f.write(f"Evaluation Report for {config_name}\n")
                f.write(f"Accuracy: {final_accuracy:.2f}%\n")
                f.write("-" * 50 + "\n")
                f.write(f"{'Image File':<20} | {'True Label':<15} | {'Prediction':<15}\n")
                f.write("-" * 50 + "\n")
                for err in error_list:
                    f.write(f"{err['path']:<20} | {err['true']:<15} | {err['pred']:<15}\n")
            print(f"Mistake list saved to: {ERROR_LIST_PATH}")
        
        except Exception as e:
            print(f"\nAn error occurred during final evaluation: {e}")
            traceback.print_exc()

        # --- グラフの保存 ---
        if history is not None:
            print(f"\nグラフを {GRAPH_SAVE_PATH} に保存します...")
            plt.figure(figsize=(12, 5))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Stage 1 Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # 結果を右上に表示
            if final_test_loss_value is not None:
                result_text = (
                    f"Test Loss: {final_test_loss_value:.4f}\n"
                    f"Accuracy : {final_accuracy:.2f}%\n"
                    f"Sequence Similarity: {final_sequence_similarity:.2f}%"
                )
                # 右上 (x=0.98, y=0.98), 縦位置合わせ=top
                plt.text(0.98, 0.98, result_text,
                         fontsize=10,
                         horizontalalignment='right',
                         verticalalignment='top',
                         transform=plt.gca().transAxes,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)) # 読みやすいようにalphaを上げる
            
            plt.grid(True)
            plt.savefig(GRAPH_SAVE_PATH)
            print("グラフを保存しました。")
    
    finally:
        print("\n--- メイン処理が終了しました ---")

if __name__ == "__main__":
    main()
