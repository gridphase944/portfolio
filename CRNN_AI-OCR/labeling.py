import cv2
import numpy as np
import os
from pathlib import Path
import re
from PIL import Image
import OCRST
import io
"""
価格専門のラベル作成
"""
def input_value_judgment(key,text_buffer):
    is_break=False
    # Enterキー (ASCII: 13)
    if key == 13:
        if not text_buffer == "":
            is_break=True
        else:
            print(f"数値が入力されていません")
    # Backspaceキー (ASCII: 8)
    elif key == 8:
        text_buffer = text_buffer[:-1] # 末尾から1文字削除
    # 数字キー (ASCII: '0' -> 48, '9' -> 57)
    elif (48 <= key <= 57) or (key==46):
        # テンキーからの入力も ASCII コードでは通常数字キーと同じ範囲
        # ただし、一部の環境ではテンキーの特殊なコードがある可能性も考慮するが、
        # 一般的にはこの範囲で十分
        text_buffer += chr(key) # ASCIIコードを文字に変換して追加
    # その他のキーが押された場合
    elif key != -1: # 何もキーが押されていない場合は-1が返される
        print(f"認識されないキーが押されました。コード: {key}")
    return text_buffer,is_break
def natural_sort_key(s):
    """
    自然順序でソート
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
def get_next_jpg_number(folder_path: str) -> int:
    """
    指定されたフォルダ直下にある数値名のJPEGを確認し、次に安全な番号を返します。
    """
    image_numbers = []
    # os.listdir() でフォルダ内のすべてのファイルとフォルダの名前を取得
    for filename in os.listdir(folder_path):
        # ファイルのフルパスを作成
        file_path = os.path.join(folder_path, filename)
        # それがファイルであり、かつ拡張子が .jpg または .jpeg かどうかをチェック
        # os.path.isfile() でディレクトリではなくファイルであることを確認
        if os.path.isfile(file_path):
            # 拡張子を小文字にして比較することで、大文字・小文字を区別しない
            stem, extension = os.path.splitext(filename)
            if extension.lower() in ('.jpg', '.jpeg') and stem.isdigit():
                image_numbers.append(int(stem))
    return max(image_numbers) + 1 if image_numbers else 1
#--------------------#初期処理#--------------------#
path_name=os.path.dirname(os.path.abspath(__file__))
ocr=OCRST.OCR()
#ラベル付けするフォルダを取得
episode_dir = os.path.join(path_name, "LabelingSourceImage")
train_image_dir = os.path.join(path_name, "OCRTrainImage")
episode_list=list(Path(episode_dir).glob('**/*.jpg'))
episode_list= [str(p) for p in episode_list]#文字列に変換
#自然順序でソート
episode_list.sort(key=natural_sort_key)
#--------------------#画像の前処理#--------------------#
previous_text=None
for a in range(len(episode_list)):
    #画像読み込み
    with open(episode_list[a], 'rb') as f:file_data = f.read()# ファイルのバイナリデータを読み込む
    nparr = np.frombuffer(file_data, np.uint8)# NumPy配列に変換
    bgr_mstimg = cv2.imdecode(nparr, cv2.IMREAD_COLOR)#画像としてデコード
    bgr_img=bgr_mstimg.copy()
    all_area_coordinates=ocr.get_area_coordinates(bgr_img)#各エリアのトリミング用の座標情報を取得
    if all_area_coordinates is None:
        continue
    th_img=ocr.threshold(bgr_img)#二値化
    #--------------------#各エリアに分けて処理#--------------------#
    for area_idx,area_coordinate in enumerate(all_area_coordinates):
        is_auto_text=False
        #--------------------#各銘柄ごとに輪郭を分類する#--------------------#
        x1, y1, x2, y2 = area_coordinate#エリア座標抜き取り
        area_mstimg=th_img[y1:y2, x1:x2].copy()
        output_img=bgr_mstimg[y1:y2, x1:x2].copy()#カラー画像からエリアを切り出す
        area_img=area_mstimg.copy()
        area_img=cv2.bitwise_not(area_img)#白黒反転※これ必須
        contours_classification=ocr.get_cell_coordinates(area_img)
        if contours_classification is None:
            continue
        #--------------------#各セルに対して処理#--------------------#
        for b in range(0,len(contours_classification),2):
            if b==2:
                continue
            if b==4:
                is_auto_text=False#自動入力解除
            for c in range(len(contours_classification[b])):
                is_skip=False#列ごとスキップ
                is_continue=False#その回のみスキップ
                print(f"エピソード:{a},{area_idx}-{b}-{c}")
                #輪郭から左上右下の位置を取得
                x=contours_classification[b][c][:,0]
                y=contours_classification[b][c][:,1]
                if len(x)==0:#輪郭がなければスキップする
                    continue
                x1=min(x)
                x2=max(x)
                y1=min(y)
                y2=max(y)

                cell_img=output_img[y1-2:y2+2, x1-2:x2+2].copy()#セル内の文字でトリミング(少し余裕を見てトリミングする)
                cell_img=ocr.add_margin(cell_img)#ここでニューラルネットワークへの入力サイズへ変更
                save_img=cell_img.copy()
                cell_img=cv2.resize(cell_img,(0,0),fx=5,fy=5)
                #--------------------##--------------------#
                #-----------#ここまではOCRと同じ。以下はラベリング専用#----------#
                #--------------------##--------------------#
                h, w, _ = cell_img.shape # 拡大後の画像の高さと幅を取得
                img_white = np.full((h, w, 3), 255, dtype=np.uint8)#表示画像と同じサイズの白紙の画像を生成 (3チャンネルのBGR画像、白色)
                combined_img = np.hstack((cell_img, img_white))
                # テキスト表示用の設定
                text_buffer = "" # 入力されたテキストを保持するバッファ
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                font_thickness = 2
                text_color = (0, 0, 0) # 黒色
                text_start_x = w + 10 # 元画像の幅 + 少しのパディング
                text_start_y = 50 # 上からの開始位置
                window_name = "Image with Text Input"
                cv2.imshow(window_name, combined_img)
                while True:
                    # 現在の結合画像をコピーしてテキストを描画（元の画像に影響を与えないため）
                    display_img = combined_img.copy()
                    #自動入力ON
                    if is_auto_text:
                        if previous_text.find(".")>=0 or correction_value.find(".")>=0:#小数点あり
                            text_buffer=str(float(previous_text)-float(correction_value))
                        else:#小数点なし
                            text_buffer=str(int(previous_text)-int(correction_value))
                    # 白紙画像部分に現在のテキストを描画
                    # cv2.putText(画像, テキスト, 座標, フォント, スケール, 色, 太さ, 線種)
                    cv2.putText(display_img, text_buffer, (text_start_x, text_start_y),
                                font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                    cv2.imshow(window_name, display_img)
                    key = cv2.waitKey(0) # キーが押されるまで無限に待機
                    #自動入力値判定
                    if key==97:#(a)
                        if not is_auto_text:#自動入力ON起動
                            if previous_text is None:
                                print("自動入力にはひとつ前の入力値が必要です")
                                continue
                            correction_value=""#補正値初期化
                            is_auto_text=True
                            while True:
                                display_img = combined_img.copy()
                                cv2.putText(display_img, "("+correction_value+")", (text_start_x, text_start_y),
                                        font, font_scale,(255,0,0), font_thickness, cv2.LINE_AA)
                                cv2.imshow(window_name, display_img)
                                key = cv2.waitKey(0) # キーが押されるまで無限に待機
                                correction_value,is_break=input_value_judgment(key,correction_value)
                                if is_break:
                                    break
                        else:
                            correction_value=""#ひとつ前の入力値から減算する値
                            text_buffer=""
                            is_auto_text=False
                        continue
                    elif key==115:#(s)
                        is_skip=True
                        break
                    elif key==99:
                        is_continue=True
                        break
                    #入力値判定
                    text_buffer,is_break=input_value_judgment(key,text_buffer)
                    if is_break:#入力値判定側の終了フラグを確認
                        break
                
                if is_skip:
                    break
                if is_continue:
                    continue
                previous_text=text_buffer#前回値を記録
                text_buffer=ocr.format_number_string_with_commas_format(text_buffer)#カンマ付き数値の文字列に変換

                #ラベルごとの保存先ディレクトリを作成
                label_dir = os.path.join(train_image_dir, text_buffer)
                os.makedirs(label_dir, exist_ok=True)
                #既存の数値ファイル名と重複しない番号を取得
                file_count=get_next_jpg_number(label_dir)

                processed_image_rgb = cv2.cvtColor(save_img, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(processed_image_rgb)
                buffer = io.BytesIO()
                pil_image.save(buffer, format='JPEG')
                buffer.seek(0)
                save_path = os.path.join(label_dir, str(file_count)+".jpg")
                try:
                    # 'xb' で既存ファイルへの上書きを防止
                    with open(save_path, 'xb') as f:
                        f.write(buffer.getvalue())
                    print(f"画像を保存しました: {save_path}")

                except Exception as e:
                    print(f"ファイルの書き込みに失敗しました: {e}")
                    print("保存先のパスと書き込み権限を確認してください。")
                print(f"Enterキーが押されました。入力された数値: {text_buffer}")

