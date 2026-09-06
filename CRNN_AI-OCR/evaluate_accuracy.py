import os
import json
from tensorflow import keras
import data_utils
import model as model_lib

# 標準ライブラリのdifflibを使って文字列類似度を計算
import difflib

def calculate_sequence_similarity(true_text, pred_text):
    """SequenceMatcher.ratio()による文字列類似度を計算"""
    matcher = difflib.SequenceMatcher(None, true_text, pred_text)
    return matcher.ratio()

def main():
    # --- 0. 設定 (input方式) ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("評価に使用するconfig名を指定してください (例: config_v1):", end="")
    config_name = input()
    config_path = os.path.join(current_dir, f"{config_name}.json")
    
    # モデルファイル名は自動推測（train.pyの保存規則に従う）
    model_path = os.path.join(current_dir, f"{config_name}.keras")
    
    if not os.path.exists(config_path): print(f"エラー: {config_path} が見つかりません。"); return
    if not os.path.exists(model_path): print(f"エラー: {model_path} が見つかりません。"); return

    # --- 1. 設定読み込み ---
    print(f"\nLoading config...")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # data_utilsへ設定反映
    data_utils.IMG_HEIGHT = 40
    data_utils.IMG_WIDTH = 155
    data_utils.AUG_PARAMS = config['augmentation_params'] # 評価時は使われないが念のため
    data_utils.SCALE_MAX = config["image_processing"]["scale_max"]
    data_utils.SCALE_MIN = config["image_processing"]["scale_min"]
    # model_libへ設定反映
    model_lib.CNN_FILTERS = config['model_params']['cnn_filters']
    model_lib.RNN_UNITS = config['model_params']['rnn_units']
    model_lib.DROPOUT = config['model_params']['dropout']
    model_lib.L2 = config["L2"]

    # --- 2. テストデータの読み込み ---
    try:
        with open(os.path.join(current_dir, "TrainImagePath.txt"), 'r', encoding='utf-8') as f:
            content = f.read().strip()
        TEST_IMG_DIR = os.path.join(content, "OCRTestImage")
    except FileNotFoundError:
        print("エラー: TrainImagePath.txt が見つかりません。")
        return

    print(f"Loading Test Data from: {TEST_IMG_DIR}")
    test_paths, test_labels = data_utils.load_image_paths_and_labels(TEST_IMG_DIR)
    total_samples = len(test_paths)
    print(f"合計テストデータ数: {total_samples} 枚")

    # 評価用データセット作成 (シャッフルなし、拡張なし)
    batch_size = 32
    test_dataset = data_utils.create_dataset(test_paths, test_labels, batch_size, is_training=False)

    # --- 3. モデルの構築とロード ---
    keras.backend.clear_session()
    print("Building Model...")
    base_model = model_lib.build_base_model()
    
    # 入力shapeを指定してビルド（ロードエラー防止）
    trainer_input_shape = {
        "image": (None, data_utils.IMG_WIDTH, data_utils.IMG_HEIGHT, 1),
        "label": (None, None)
    }
    # CTCTrainer経由でビルド
    evaluation_model = model_lib.CTCTrainer(base_model, all_train_labels_for_weighting=None)
    evaluation_model.compile(optimizer=keras.optimizers.Adam())
    evaluation_model.build(input_shape=trainer_input_shape)
    base_model.build(input_shape=(None, data_utils.IMG_WIDTH, data_utils.IMG_HEIGHT, 1))
    
    print(f"Loading Weights from {model_path}...")
    evaluation_model.load_weights(model_path)
    print("Model Loaded Successfully.")

    # --- 4. 予測と評価の実行 ---
    print("\n--- 評価を開始します (時間がかかる場合があります) ---")
    
    correct_count = 0
    total_sequence_similarity = 0
    error_list = []

    # バッチごとに処理
    processed_count = 0
    for batch in test_dataset:
        images = batch['image']
        # 予測
        preds = base_model.predict(images, verbose=0)
        pred_texts = model_lib.ctc_decode(preds)
        
        # 正解ラベルを元のリストから取得 (バッチ内のインデックスと全体のリストを対応させる)
        current_batch_size = len(pred_texts)
        start_idx = processed_count
        end_idx = processed_count + current_batch_size
        
        true_labels_batch = test_labels[start_idx:end_idx]
        true_paths_batch = test_paths[start_idx:end_idx]

        for i in range(current_batch_size):
            true_text = true_labels_batch[i]
            pred_text = pred_texts[i]
            
            # 完全一致のチェック
            if true_text == pred_text:
                correct_count += 1
            else:
                # 間違いリストに追加
                error_list.append({
                    "path": os.path.basename(true_paths_batch[i]), # ファイル名のみ
                    "true": true_text,
                    "pred": pred_text
                })
            
            # 文字列類似度の加算
            total_sequence_similarity += calculate_sequence_similarity(true_text, pred_text)

        processed_count += current_batch_size
        print(f"\rProcessing: {processed_count}/{total_samples}", end="")

    # --- 5. 結果の集計と表示 ---
    accuracy = (correct_count / total_samples) * 100
    avg_sequence_similarity = (total_sequence_similarity / total_samples) * 100

    print("\n\n" + "="*50)
    print("   【最終評価結果】")
    print("="*50)
    print(f"データ総数: {total_samples}")
    print(f"完全正解数: {correct_count}")
    print("-" * 30)
    print(f"★ 完全一致率 (Accuracy) : {accuracy:.2f}%")
    print(f"★ 文字列類似度 (SequenceMatcher): {avg_sequence_similarity:.2f}%")
    print("="*50)

    # --- 6. 間違いレポートの保存 ---
    report_file = "evaluation_errors.txt"
    with open(os.path.join(current_dir, report_file), "w", encoding="utf-8") as f:
        f.write(f"Evaluation Report for {config_name}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Image File':<20} | {'True Label':<15} | {'Prediction':<15}\n")
        f.write("-" * 50 + "\n")
        for err in error_list:
            f.write(f"{err['path']:<20} | {err['true']:<15} | {err['pred']:<15}\n")
    
    print(f"\n間違えたデータの一覧を '{report_file}' に保存しました。")
    print("このリストを確認して、許容できるミスかどうか判断してください。")

if __name__ == "__main__":
    main()
