import OCRST
import numpy as np
import datetime
import os
ocr=OCRST.OCR()
dt_now=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')#日時のフォルダ名作成
path_name=os.path.dirname(os.path.abspath(__file__))
folder=path_name+"/"+dt_now#保存先
os.mkdir(folder)#フォルダ生成
#以下各情報の座標登録
#歩み値

img=ocr.MainWindowImage("1")
textPrint=[
    "歩み値時間",
    "歩み値約定値",
    "歩み値出来高"
]
windowname="stepvalue"
coordinate=ocr.GetCoordinate(img,textPrint,windowname,35)
np.save(folder+"/"+windowname+".npy",coordinate)


#全板
img=ocr.MainWindowImage("1")
textPrint=[
    "全板値段",
    "全板買い",
    "全板買い件数",
    "全板売り",
    "全板売り件数"
]
windowname="fullboard"
coordinate=ocr.GetCoordinate(img,textPrint,windowname,35)
np.save(folder+"/"+windowname+".npy",coordinate)


#ランキング銘柄コード(銘柄コード欄に文字があるかないかでCSVを入手できるか判定している)
img=ocr.MainWindowImage("2")
textPrint=[
    "銘柄コード"
]
windowname="ranking_stockcode"
coordinate=ocr.GetCoordinate(img,textPrint,windowname,1)
np.save(folder+"/"+windowname+".npy",coordinate)


#全板銘柄コード
img=ocr.MainWindowImage("1")
textPrint=[
    "全板銘柄コード"
]
windowname="fullboard_stockcode"
coordinate=ocr.GetCoordinate(img,textPrint,windowname,1)
np.save(folder+"/"+windowname+".npy",coordinate)


#スピード注文
img=ocr.MainWindowImage("2")
textPrint=[
    "買い",
    "売り",
]
windowname="speedorder"
coordinate=ocr.GetCoordinate(img,textPrint,windowname,2)
np.save(folder+"/"+windowname+".npy",coordinate)


#スピード注文銘柄コード
img=ocr.MainWindowImage("2")
textPrint=[
    "銘柄コード"
]
windowname="speedorder_stockcode"
coordinate=ocr.GetCoordinate(img,textPrint,windowname,1)
np.save(folder+"/"+windowname+".npy",coordinate)
