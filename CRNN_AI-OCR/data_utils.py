# data_utils.py
import os
import glob
import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

# --- グローバル設定 ---
IMG_HEIGHT = 40
IMG_WIDTH = 155
# デフォルト値を設定しておき、configにキーがなくても動くようにする
AUG_PARAMS = {
    "brightness_delta": 0.2, "contrast_lower": 0.8, "contrast_upper": 1.2, "noise_stddev": 0.1,
    "perspective_prob": 0.0, "perspective_shear": 0.0,
    "blur_prob": 0.0,
    "cutout_prob": 0.0, "cutout_size_min": 0.1, "cutout_size_max": 0.3
}
SCALE_MAX = 1.0
SCALE_MIN = 1.0
ROTATION_RANGE = 0.0

# --- グローバルな辞書設定 ---
PAD_TOKEN = '<pad>'
characters = sorted(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',', '.'])
characters.append(PAD_TOKEN)

char_to_num = keras.layers.StringLookup(vocabulary=list(characters))
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    invert=True
)

def load_image_paths_and_labels(dir_path):
    image_paths = []
    labels = []
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"ディレクトリが見つかりません: {dir_path}")
    for label_dir in sorted(os.listdir(dir_path)):
        label_path = os.path.join(dir_path, label_dir)
        if os.path.isdir(label_path):
            files_in_dir = []
            for ext in ('*.jpg', '*.png', '*.jpeg'):
                files_in_dir.extend(glob.glob(os.path.join(label_path, ext)))
            if files_in_dir:
                image_paths.extend(sorted(files_in_dir))
                labels.extend([label_dir] * len(files_in_dir))
    return image_paths, labels

def crop_to_content(img_bgr):
    """画像から文字部分を切り抜く"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return img_bgr

    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = float('-inf'), float('-inf')
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    margin = 2
    h_img, w_img = img_bgr.shape[:2]
    x_min = int(max(0, x_min - margin))
    y_min = int(max(0, y_min - margin))
    x_max = int(min(w_img, x_max + margin))
    y_max = int(min(h_img, y_max + margin))

    if x_max <= x_min or y_max <= y_min: return img_bgr
    return img_bgr[y_min:y_max, x_min:x_max]

def create_canvas(height, width, img_bgr, mode='noise'):
    """背景キャンバスを作成"""
    if mode == 'white':
        return np.ones((height, width, 3), dtype=np.uint8) * 255
    
    elif mode == 'pastel':
        # --- Configからパラメータを取得 (デフォルト値を設定して安全に) ---
        s_min, s_max = AUG_PARAMS.get('bg_pastel_saturation_range', [10, 50])
        v_min, v_max = AUG_PARAMS.get('bg_pastel_value_range', [230, 255])
        
        # 色相(Hue)は全色ランダムでOK
        h = random.randint(0, 179)
        # 彩度(Sat)と明度(Val)をConfigの範囲内でランダム決定
        s = random.randint(s_min, s_max)
        v = random.randint(v_min, v_max)
        
        hsv_color = np.uint8([[[h, s, v]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        
        # 塗りつぶし
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = bgr_color
        return canvas

    else: # mode == 'noise'
        color_palette = []
        h_src, w_src = img_bgr.shape[:2]
        if h_src >= 1 and w_src >= 1:
            corner_pixels = [
                tuple(img_bgr[0, 0]), tuple(img_bgr[0, w_src-1]),
                tuple(img_bgr[h_src-1, 0]), tuple(img_bgr[h_src-1, w_src-1])
            ]
            color_palette = list(set(corner_pixels))
        if not color_palette:
            color_palette = [(255, 255, 255)]

        return np.array(
            [random.choice(color_palette) for _ in range(height * width)],
            dtype=np.uint8
        ).reshape(height, width, 3)

def opencv_process_common(img_bgr, mode):
    # 1. オートクロップ
    img_cropped = crop_to_content(img_bgr)
    h_content, w_content = img_cropped.shape[:2]
    if h_content == 0 or w_content == 0: return cv2.resize(img_bgr, (IMG_WIDTH, IMG_HEIGHT))

    # 2. キャンバス作成
    if mode == 'hard':
        # --- Configから確率を取得 ---
        pastel_prob = AUG_PARAMS.get('bg_pastel_prob', 0.5)
        
        # 確率に基づいてモード決定
        bg_mode = 'pastel' if random.random() < pastel_prob else 'noise'
    else:
        bg_mode = 'white'
        
    canvas = create_canvas(IMG_HEIGHT, IMG_WIDTH, img_bgr, mode=bg_mode)

    # 3. 変形パラメータ決定
    if mode == 'hard':
        scale = random.uniform(SCALE_MIN, SCALE_MAX)
        angle = random.uniform(-ROTATION_RANGE, ROTATION_RANGE) if ROTATION_RANGE > 0 else 0.0
    else:
        scale_h = IMG_HEIGHT / h_content
        scale_w = IMG_WIDTH / w_content
        scale = min(scale_h, scale_w) * 0.9
        angle = 0.0

    new_w = int(w_content * scale)
    new_h = int(h_content * scale)
    if new_w < 1: new_w = 1
    if new_h < 1: new_h = 1

    # 4. リサイズ
    img_resized = cv2.resize(img_cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    gray_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_resized, 160, 255, cv2.THRESH_BINARY_INV)
    
    # 5. 回転 & 透視変換 & ぼかし (ハードモードのみ)
    if mode == 'hard':
        # --- 回転 ---
        if angle != 0.0:
            center = (new_w//2, new_h//2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            cos = abs(M[0, 0])
            sin = abs(M[0, 1])
            rotated_w = int(np.ceil((new_h * sin) + (new_w * cos)))
            rotated_h = int(np.ceil((new_h * cos) + (new_w * sin)))
            M[0, 2] += (rotated_w / 2) - center[0]
            M[1, 2] += (rotated_h / 2) - center[1]
            img_resized = cv2.warpAffine(img_resized, M, (rotated_w, rotated_h), borderValue=(255, 255, 255))
            mask = cv2.warpAffine(mask, M, (rotated_w, rotated_h), borderValue=0)
            new_w, new_h = rotated_w, rotated_h

        # --- 透視変換 (Perspective Transform) ---
        # configから確率と強度を取得。キーがない場合は0(無効)
        if random.random() < AUG_PARAMS.get('perspective_prob', 0.0):
            shear = AUG_PARAMS.get('perspective_shear', 0.0)
            if shear > 0:
                src_pts = np.float32([[0, 0], [new_w, 0], [0, new_h], [new_w, new_h]])
                shear_w = new_w * shear
                shear_h = new_h * shear
                
                dst_pts = np.float32([
                    [random.uniform(0, shear_w), random.uniform(0, shear_h)],
                    [new_w - random.uniform(0, shear_w), random.uniform(0, shear_h)],
                    [random.uniform(0, shear_w), new_h - random.uniform(0, shear_h)],
                    [new_w - random.uniform(0, shear_w), new_h - random.uniform(0, shear_h)]
                ])
                M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
                img_resized = cv2.warpPerspective(img_resized, M_persp, (new_w, new_h), borderValue=(255, 255, 255))
                mask = cv2.warpPerspective(mask, M_persp, (new_w, new_h), borderValue=0)

        # --- ぼかし (Blur) ---
        if random.random() < AUG_PARAMS.get('blur_prob', 0.0):
            # カーネルサイズは3か5からランダムに選ぶ
            ksize = random.choice([3, 5])
            img_resized = cv2.GaussianBlur(img_resized, (ksize, ksize), 0)
            # マスクはぼかさない

        # 変形後の画像全体がキャンバス内に収まるよう、必要な場合だけ縮小する
        fit_scale = min(1.0, IMG_WIDTH / new_w, IMG_HEIGHT / new_h)
        if fit_scale < 1.0:
            new_w = max(1, int(new_w * fit_scale))
            new_h = max(1, int(new_h * fit_scale))
            img_resized = cv2.resize(img_resized, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 6. 配置決定
    if mode == 'hard':
        max_x = max(0, IMG_WIDTH - new_w)
        max_y = max(0, IMG_HEIGHT - new_h)
        start_x = random.randint(0, max_x)
        start_y = random.randint(0, max_y)
    else:
        start_x = (IMG_WIDTH - new_w) // 2
        start_y = (IMG_HEIGHT - new_h) // 2

    # 7. 転写 (ステンシル)
    end_x = min(start_x + new_w, IMG_WIDTH)
    end_y = min(start_y + new_h, IMG_HEIGHT)
    paste_w = end_x - start_x
    paste_h = end_y - start_y
    
    if paste_w > 0 and paste_h > 0:
        dest_region = canvas[start_y:end_y, start_x:end_x]
        src_region = img_resized[0:paste_h, 0:paste_w]
        mask_region = mask[0:paste_h, 0:paste_w]
        
        mask_3d = cv2.cvtColor(mask_region, cv2.COLOR_GRAY2BGR)
        paste_condition = (mask_3d > 0)
        dest_region[:] = np.where(paste_condition, src_region, dest_region)

    # --- 欠損 (Cutout) - ハードモードのみ ---
    if mode == 'hard':
        if random.random() < AUG_PARAMS.get('cutout_prob', 0.0):
            min_ratio = AUG_PARAMS.get('cutout_size_min', 0.1)
            max_ratio = AUG_PARAMS.get('cutout_size_max', 0.3)
            
            h_cut = int(IMG_HEIGHT * random.uniform(min_ratio, max_ratio))
            w_cut = int(IMG_WIDTH * random.uniform(min_ratio, max_ratio))
            
            if h_cut > 0 and w_cut > 0:
                y_cut = random.randint(0, max(0, IMG_HEIGHT - h_cut))
                x_cut = random.randint(0, max(0, IMG_WIDTH - w_cut))
                
                # 欠損部分を再度ノイズで埋める（自然に見せるため）
                noise_patch = create_canvas(h_cut, w_cut, img_bgr, mode='noise')
                canvas[y_cut:y_cut+h_cut, x_cut:x_cut+w_cut] = noise_patch

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

# --- ラッパー関数 ---
def opencv_process_hard(image_tensor):
    img_bgr = cv2.cvtColor((image_tensor.numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return opencv_process_common(img_bgr, mode='hard')

def opencv_process_easy(image_tensor):
    img_bgr = cv2.cvtColor((image_tensor.numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return opencv_process_common(img_bgr, mode='easy')

# --- 前処理関数 (モード別) ---
def preprocess_image_hard(image_path, label):
    """ハードモード用の前処理"""
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image_float = tf.image.convert_image_dtype(image, tf.float32)
    
    image_np_rgb = tf.py_function(func=opencv_process_hard, inp=[image_float], Tout=tf.uint8)
    image_np_rgb.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
    image = tf.image.convert_image_dtype(image_np_rgb, tf.float32)
    
    # TensorFlow Pixel Augmentation
    image = tf.image.random_brightness(image, max_delta=AUG_PARAMS['brightness_delta'])
    image = tf.image.random_contrast(image, lower=AUG_PARAMS['contrast_lower'], upper=AUG_PARAMS['contrast_upper'])
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=AUG_PARAMS['noise_stddev'], dtype=tf.float32)
    image = tf.add(image, noise)
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    image = tf.image.rgb_to_grayscale(image)
    image = tf.transpose(image, perm=[1, 0, 2])
    
    label = char_to_num(tf.strings.unicode_split(label, "UTF-8"))
    return {"image": image, "label": label}

def preprocess_image_easy(image_path, label):
    """イージーモード用の前処理"""
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image_float = tf.image.convert_image_dtype(image, tf.float32)
    
    image_np_rgb = tf.py_function(func=opencv_process_easy, inp=[image_float], Tout=tf.uint8)
    image_np_rgb.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
    image = tf.image.convert_image_dtype(image_np_rgb, tf.float32)
    
    image = tf.image.rgb_to_grayscale(image)
    image = tf.transpose(image, perm=[1, 0, 2])
    
    label = char_to_num(tf.strings.unicode_split(label, "UTF-8"))
    return {"image": image, "label": label}

def create_dataset(paths, labels, batch_size, is_training=False):
    """データセットを作成。学習時はHard:Easy=1:1"""
    base_ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    padding_value = char_to_num(PAD_TOKEN)
    
    if is_training:
        ds_hard = base_ds.map(preprocess_image_hard, num_parallel_calls=tf.data.AUTOTUNE)
        ds_easy = base_ds.map(preprocess_image_easy, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = ds_hard.concatenate(ds_easy)
        dataset = dataset.shuffle(buffer_size=len(paths)*2, reshuffle_each_iteration=True)
    else:
        dataset = base_ds.map(preprocess_image_easy, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes={"image": [IMG_WIDTH, IMG_HEIGHT, 1], "label": [None]},
        padding_values={"image": 0.0, "label": padding_value}
    )
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
