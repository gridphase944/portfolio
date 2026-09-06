import cv2
import os
import numpy as np
import random
class OCR():
    def get_area_coordinates(self,bgr_img):
        """
        全体の画像から各エリアの座標情報を取得
        bgr_img=必ずbgrカラーで入力
        """
        path_name=os.path.dirname(os.path.abspath(__file__))
        #--------------------#基準画像#--------------------#
        reference_img=["buy","sell","over","undr"]
        reference_img_shape=[[]]*len(reference_img)
        for a in range(len(reference_img)):
            reference_img[a]=cv2.imread(path_name+"/image/"+reference_img[a]+".jpg")#画像[a]
            reference_img_shape[a]=reference_img[a].shape[:2]#height,width
        #--------------------#マッチング#--------------------#
        # ※必要な座標のみ取得#buy=x,sell=x,over=y,undr=y
        matching_result=[]
        for a in range(len(reference_img)):
            coordinates_list=self.find_non_overlapping_matches(bgr_img,reference_img[a],0.7,)#検出座標フィルタリング付きテンプレートマッチング
            if not len(coordinates_list)==36:
                return None#万が一すべての位置を特定できていないならこの回はスキップする
            temp_matching_coordinates = [
                x if a == 0 or a == 1 else y
                for x, y, _ in coordinates_list
            ]
            matching_result.append(self.unique_with_tolerance(temp_matching_coordinates))#許容範囲を考慮したユニーク化
        #--------------------#すべての要素が6でない場合はこの回はスキップする。#--------------------#
        if not all(len(sublist) == 6 for sublist in matching_result):
            return None
        #--------------------#X座標#--------------------#
        coordinate_x=[0]*(len(matching_result[0])+1)
        for a in range(len(matching_result[0])-1):
            buy_center=matching_result[0][a]+(reference_img_shape[0][1]//2)
            sell_center=matching_result[1][a+1]+(reference_img_shape[1][1]//2)
            coordinate_x[a+1]=((buy_center+sell_center)//2)-4
        bgr_img_h,bgr_img_w=bgr_img.shape[:2]
        coordinate_x[len(coordinate_x)-1]=bgr_img_w-1
        #--------------------#Y座標#--------------------#
        coordinate_y=[0]*(len(matching_result[2])+len(matching_result[3]))
        for a in range(len(matching_result[2])):
            coordinate_y[a*2]=matching_result[2][a]+reference_img_shape[2][0]+7#上線
            coordinate_y[a*2+1]=matching_result[3][a]-7#下線
        #--------------------#実際の座標を[x1,y1,x2,y2]の形に変換#--------------------#
        area_coordinates=[]#輪郭エリア[x1,y1,x2,y2]
        for a in range(len(coordinate_x)-1):
            for b in range(0,len(coordinate_y),2):
                area_coordinates.append((coordinate_x[a],coordinate_y[b],coordinate_x[a+1],coordinate_y[b+1]))
        return area_coordinates
    def get_cell_coordinates(self,th_img):
        """
        一つの銘柄の板情報をセル分けする
        ※全エリア画像を引数に設定しないで1エリアの画像を引数に設定してください
        """
        coordinate_threshold=5#座標値がn離れていたら別のセル値として認識させる閾値
        contours,_=cv2.findContours(th_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)#一つの銘柄の板情報に対して輪郭抽出
        if not contours:
            return None
        contours=np.array(self.convert_4d_to_2d(contours))#二次元のnumpy配列に変換
        #--------------------#X軸基準#--------------------#
        contours_x_classification=[]#x軸を基準として左から順に輪郭を分類する
        contours,over_threshold_idx=self.find_coordinates_beyond_distance_idx(contours,"x",8)#既定の距離座標が離れた位置のインデックス番号を取得
        for aa in range(len(over_threshold_idx)-1):
            contours_x_classification.append(contours[over_threshold_idx[aa]:over_threshold_idx[aa+1]])
        if not len(contours_x_classification)==5:
            return None
        #--------------------#Y軸基準#--------------------#
        contours_classification=[[],[],[],[],[]]#最終的な座標を登録
        for aa in range(len(contours_x_classification)):
            contours,over_threshold_idx=self.find_coordinates_beyond_distance_idx(contours_x_classification[aa],"y",5)
            for bb in range(len(over_threshold_idx)-1):
                contours_classification[aa].append(contours[over_threshold_idx[bb]:over_threshold_idx[bb+1]])
        if not len(contours_classification[2])==33:
            return None
        #売買のy軸は1セル飛ばしていたり要素が全然足りないので空の配列を補完する#--------------------
        #各項目の各セルの平均y軸座標を取得
        price_y_average=np.zeros(len(contours_classification[2]))#各値段の座標yの平均値
        sell_y_average=np.zeros(len(contours_classification[0]))#各売りの座標yの平均値
        buy_y_average=np.zeros(len(contours_classification[4]))#各買いの座標yの平均値       
        for aa in range(price_y_average.shape[0]):
            price_y_average[aa]=self.UnidirectionalAverage(contours_classification[2][aa],"y")#平均し、切り上げしたのを整数値に変換
        for aa in range(sell_y_average.shape[0]):
            sell_y_average[aa]=self.UnidirectionalAverage(contours_classification[0][aa],"y")#平均し、切り上げしたのを整数値に変換
        for aa in range(buy_y_average.shape[0]):
            buy_y_average[aa]=self.UnidirectionalAverage(contours_classification[4][aa],"y")#平均し、切り上げしたのを整数値に変換
        #空の配列を補完
        sell_contours=[]#後でcontours_classificationの配列に差し替える
        buy_contours=[]#後でcontours_classificationの配列に差し替える
        sell_idx=0
        buy_idx=0
        is_sell=True
        is_buy=True
        for aa in range(len(price_y_average)):
            #売り
            if is_sell:
                if abs(sell_y_average[sell_idx]-price_y_average[aa])<coordinate_threshold:
                    sell_contours.append(contours_classification[0][sell_idx])
                    sell_idx+=1
                    if sell_idx==len(sell_y_average):
                        is_sell=False
                else:
                    sell_contours.append(np.empty((0,2),dtype=np.int32))
            else:
                sell_contours.append(np.empty((0,2),dtype=np.int32))
            #買い
            if is_buy:
                if abs(buy_y_average[buy_idx]-price_y_average[aa])<coordinate_threshold:
                    buy_contours.append(contours_classification[4][buy_idx])
                    buy_idx+=1
                    if buy_idx==len(buy_y_average):
                        is_buy=False
                else:
                    buy_contours.append(np.empty((0,2),dtype=np.int32))
            else:
                buy_contours.append(np.empty((0,2),dtype=np.int32))
        contours_classification[0]=sell_contours#最後差し替え
        contours_classification[4]=buy_contours#最後差し替え
        
        return contours_classification
    def add_margin(self, img):
        """
        「ステンシル」方式で、ノイズ背景に文字を転写する。
        """
        height, width = img.shape[:2]
        TARGET_WIDTH = 155
        TARGET_HEIGHT = 40

        # 元画像が目標サイズより大きい軸だけ中央を切り取る
        if width > TARGET_WIDTH:
            x_start = (width - TARGET_WIDTH) // 2
            img = img[:, x_start:x_start+TARGET_WIDTH]
        if height > TARGET_HEIGHT:
            y_start = (height - TARGET_HEIGHT) // 2
            img = img[y_start:y_start+TARGET_HEIGHT, :]
        height, width = img.shape[:2]

        # --- ステップ1: ノイズキャンバスの作成 ---

        # 1a. 四隅の「1ピクセル」からカラーパレットを作成
        color_palette = []
        # 四隅の単一ピクセルをサンプリング
        corner_pixels = [
            tuple(img[0, 0]),             # 左上
            tuple(img[0, width-1]),       # 右上
            tuple(img[height-1, 0]),     # 左下
            tuple(img[height-1, width-1])  # 右下
        ]
        # 重複する色を削除
        color_palette = list(set(corner_pixels))

        # パレットが空の場合は代替として左上の色を使用
        if not color_palette:
            color_palette = [tuple(img[0, 0])]

        # 1b. 目標サイズのキャンバスをランダムノイズで埋める
        noise_canvas = np.array(
            [random.choice(color_palette) for _ in range(TARGET_HEIGHT * TARGET_WIDTH)],
            dtype=np.uint8
        ).reshape(TARGET_HEIGHT, TARGET_WIDTH, 3)

        # --- ステップ2: 元画像から「型紙（マスク）」を作る ---

        # 2a. 元画像をグレースケールに変換して二値化
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2b. 閾値処理でマスクを作成 (背景が黒[0]、文字が白[255])
        # この閾値(240)は、背景の明るさに応じて調整が必要な場合があります
        _, mask = cv2.threshold(gray_img, 240, 255, cv2.THRESH_BINARY_INV)

        # --- ステップ3: 型紙を使って文字だけをノイズキャンバスに転写 ---

        # 3a. 貼り付け位置を計算 (中央揃え)
        x_offset = (TARGET_WIDTH - width) // 2
        y_offset = (TARGET_HEIGHT - height) // 2

        # 3b. 元画像の文字部分のピクセル座標を取得 (マスクが0でない場所)
        text_pixels_y, text_pixels_x = np.where(mask != 0)

        # 3c. 座標をずらして、ノイズキャンバス上の対応する位置を計算
        canvas_y_coords = text_pixels_y + y_offset
        canvas_x_coords = text_pixels_x + x_offset
        
        # 3d. 元画像の文字ピクセルの色を取得
        text_colors = img[text_pixels_y, text_pixels_x]

        # 3e. ノイズキャンバスの対応する位置に、文字の色を上書き（転写）
        noise_canvas[canvas_y_coords, canvas_x_coords] = text_colors

        return noise_canvas
    
    def threshold(self,bgr_img):
        """
        二値化
        """
        gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)# グレースケールに変換する。
        _,th_img=cv2.threshold(gray_img,160,255,cv2.THRESH_BINARY)#二値化160
        return th_img
    def convert_4d_to_2d(self,arr_4):
        """
        三次元配列を二次元配列に変換
        """
        arr_2=[]
        for a in range(len(arr_4)):
            for b in range(len(arr_4[a])):
                for c in range(len(arr_4[a][b])):
                    arr_2.append(arr_4[a][b][c])
        return arr_2
    def UnidirectionalAverage(self,arr,direction):
        """
        片方向のみの座標の平均値を取得
        direction:"x"もしくは"y"
        """
        if direction=="x":
            direction=0
        else:
            direction=1
        unidirectional=arr[:,direction]
        unidirectional=np.ceil(np.mean(unidirectional)).astype(int)
        return unidirectional
    def find_coordinates_beyond_distance_idx(self,coordinates,axis,coordinate_threshold):
        """
        既定の距離座標が離れた位置のインデックス番号を取得
        axis="x" or "y"
        """
        if axis=="x":
            axis=0
        elif axis=="y":
            axis=1
        coordinates=coordinates[np.argsort(coordinates[:,axis])]#指定の軸方向でソート
        diff_x=np.diff(coordinates[:,axis])#次の座標との差を計算
        diff_x=np.insert(diff_x,0,0)#最初は差が存在しないので先頭に0を追加
        over_threshold_idx=np.where(diff_x>coordinate_threshold)#座標が離れているところのインデックス番号を取得
        over_threshold_idx=np.insert(over_threshold_idx,0,0)#先頭の要素番号を追加
        over_threshold_idx=np.append(over_threshold_idx,len(diff_x))#末尾の要素番号を追加
        return coordinates,over_threshold_idx
    def format_number_string_with_commas_format(self,num_str: str) -> str:
        """
        数字の文字列を数値に変換し、カンマを3桁ごとに挿入して返します。
        数値に変換できない文字列が渡されるとエラーになります。
        Args:
            num_str (str): カンマを追加したい数字の文字列 (例: "1920", "1234567890")
        Returns:
            str: カンマが追加された文字列 (例: "1,920", "1,234,567,890")
        Raises:
            ValueError: 数字に変換できない文字列が渡された場合。
        """
        if '.' in num_str:
            # 小数点を含む場合はfloatに変換
            try:
                num = float(num_str)
            except ValueError:
                raise ValueError(f"'{num_str}' は浮動小数点数に変換できません。")
            return f"{num:,.1f}" # 小数点以下1桁まで表示
        else:
            # 整数に変換
            try:
                num = int(num_str)
            except ValueError:
                raise ValueError(f"'{num_str}' は整数に変換できません。")
            return f"{num:,}"
    def find_non_overlapping_matches(self,bgr_img: np.ndarray, reference_img: np.ndarray, threshold: float,min_overlap_ratio: float = 0.5) -> list[tuple[int, int, float]]:
        """
        テンプレートマッチングを実行し、重複する（近すぎる）検出結果を除外する。

        Args:
            bgr_img (np.ndarray): 検索対象となる元の画像（NumPy配列）。
            reference_img (np.ndarray): テンプレート画像（サーチしたい画像）。
            threshold (float): 検出と見なす類似度スコアの最小閾値（例: 0.70）。
            min_overlap_ratio (float): 重複と見なす最小距離の基準。
                                    テンプレートサイズのこの割合内にある検出は除外される。(0.0-1.0)
                                    デフォルトは0.5 (テンプレートサイズの半分の距離)。

        Returns:
            list[tuple[int, int, float]]: 重複が除外された検出結果のリスト。
                                        要素は (x座標, y座標, スコア) のタプルで、xとyはint型。
        """
        # テンプレートの幅と高さを取得
        template_height, template_width = reference_img.shape[:2]

        # テンプレートマッチングを実行
        # cv2.TM_CCOEFF_NORMED は類似度が高いほどスコアが1.0に近づく
        result = cv2.matchTemplate(bgr_img, reference_img, cv2.TM_CCOEFF_NORMED)
        
        # 閾値以上の座標を取得
        matching = np.where(result >= threshold)

        # 1. 座標とスコアを結合し、リストに格納 (score, x, y)
        detections = []
        for pt_y, pt_x in zip(*matching):
            # スコアをPython標準のfloat型に変換
            score = float(result[pt_y, pt_x])
            # x, yはint型
            detections.append((score, pt_x, pt_y)) 

        # スコア（類似度）が高い順にソート
        detections.sort(key=lambda x: x[0], reverse=True)

        # 2. 重複チェックのための最小距離を設定
        # テンプレートの幅/高さに min_overlap_ratio をかけた値を最小距離とする
        min_distance_x = int(template_width * min_overlap_ratio)
        min_distance_y = int(template_height * min_overlap_ratio)

        # 3. 座標のフィルタリング（近すぎる座標の削除）
        final_detections = []
        while detections:
            # 最もスコアが高い座標を取得
            best_score, best_x, best_y = detections.pop(0)

            # 最終リストに追加（x, y は int型）
            final_detections.append((int(best_x), int(best_y), best_score))

            # このベストな座標に近すぎる他の座標をリストから削除
            detections_filtered = []
            for score, x, y in detections:
                # 重複判定: XとY座標の両方が最小距離内にある場合に重複と見なす
                if (abs(best_x - x) < min_distance_x) and (abs(best_y - y) < min_distance_y):
                    # 破棄
                    pass
                else:
                    # 遠い場合は次のチェック対象として残す
                    detections_filtered.append((score, x, y))

            detections = detections_filtered

        return final_detections
    def unique_with_tolerance(self,data_list, tolerance=10):
        """
        数値リストから、指定した許容範囲内の近い値を重複と見なして除いた
        ユニークな要素のリストを生成します。

        Args:
            data_list (list): 処理したい数値のリスト。
            tolerance (int/float): 許容する数値の差。

        Returns:
            list: ユニーク化された要素のリスト。
        """
        if not data_list:
            return []

        # 1. リストをソートする（必須）
        sorted_list = sorted(data_list)
        
        # 2. 最初の要素を基準値として設定
        unique_results = [sorted_list[0]]
        
        # 3. 2番目の要素から順に比較
        for current_value in sorted_list[1:]:
            # 最後のユニークな値（基準値）との差を計算
            last_unique_value = unique_results[-1]
            
            # 差が許容範囲より大きい（離れている）場合のみ追加
            if current_value - last_unique_value > tolerance:
                unique_results.append(current_value)
                
        return unique_results
