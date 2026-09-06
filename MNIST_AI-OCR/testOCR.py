import os
from pathlib import Path
import numpy as np
import OCRST
import re
from time import time
import cv2
from multiprocessing import Process
def natural_sort_key(s):return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
def OCRaction(episode_dir):
    #初期設定
    ocr=OCRST.OCR()
    image_cut=[0,960,1920]#トリミング位置
    #各種フォルダ名設定
    path_name=os.path.dirname(os.path.abspath(__file__))
    output_dir=path_name+"/OCR_Result_Numpy/"
    #エピソードループ
    for a in range(len(episode_dir)):
        #座標情報の読み出し
        sever_name=episode_dir[a].name[:episode_dir[a].name.find('_')]#エピソードディレクトリ名からサーバー名を取得
        if sever_name=="P":sever_name="p"#大文字で設定してしまったので小文字に変更
        stepvalue_coordinate=np.load(path_name+"/coordinate_OCR_Result/"+sever_name+"/stepvalue.npy")
        tradingstatus_coordinate=np.load(path_name+"/coordinate_OCR_Result/"+sever_name+"/tradingstatus.npy")
        time_coordinate=stepvalue_coordinate[0]
        value_coordinate=stepvalue_coordinate[1]
        volume_coordinate=stepvalue_coordinate[2]
        price_coordinate=tradingstatus_coordinate[0]
        buy_coordinate=tradingstatus_coordinate[1]
        buyNumber_coordinate=tradingstatus_coordinate[2]
        sell_coordinate=tradingstatus_coordinate[3]
        sellNumber_coordinate=tradingstatus_coordinate[4]
        #画像ファイルパス取得
        episode_list=list(Path(episode_dir[a]).glob('**/*.jpg'))
        episode_list= [str(p) for p in episode_list]#文字列に変換
        #自然順序でソート
        episode_list.sort(key=natural_sort_key)
        #mainサーバーのみ左側しかOCRしないからここで調整
        if sever_name=="p":
            count=1
        else:
            count=len(image_cut)-1
        #OCR処理！
        for b in range(count):#画像トリミング位置
            flag=True
            for c in range(len(episode_list)):
                starttime=time()
                mstimg=cv2.imread(episode_list[c])#画像取得
                color_img=mstimg[0:1080,image_cut[b]:image_cut[b+1]].copy()
                draw_img=ocr.threshold(color_img.copy())
                time_OCR_result=ocr.output_ocr(draw_img,time_coordinate,True,False)
                value_OCR_result=ocr.output_ocr(draw_img,value_coordinate,False,False)
                volume_OCR_result=ocr.output_ocr(draw_img,volume_coordinate,False,False)
                price_OCR_result=ocr.output_ocr(draw_img,price_coordinate,False,False)
                nowPriceIndex_OCR_result=ocr.PositionInvestigation(color_img,price_coordinate)
                buy_OCR_result=ocr.output_ocr(draw_img,buy_coordinate,False,True)
                buyNumber_OCR_result=ocr.output_ocr(draw_img,buyNumber_coordinate,False,True)
                sell_OCR_result=ocr.output_ocr(draw_img,sell_coordinate,False,True)
                sellNumber_OCR_result=ocr.output_ocr(draw_img,sellNumber_coordinate,False,True)
                if time_OCR_result is None:continue
                if value_OCR_result is None:continue
                if volume_OCR_result is None:continue
                if price_OCR_result is None:continue
                if nowPriceIndex_OCR_result is None:continue
                if buy_OCR_result is None:continue
                if buyNumber_OCR_result is None:continue
                if sell_OCR_result is None:continue
                if sellNumber_OCR_result is None:continue
                if flag:
                    flag=False
                    now_time=np.array([],dtype=int)
                    time_OCR=np.empty((0,len(time_OCR_result)))
                    value_OCR=np.empty((0,len(value_OCR_result)))
                    volume_OCR=np.empty((0,len(volume_OCR_result)))
                    price_OCR=np.empty((0,len(price_OCR_result)))
                    nowPriceIndex_OCR=np.array([],dtype=int)
                    buy_OCR=np.empty((0,len(buy_OCR_result)))
                    buyNumber_OCR=np.empty((0,len(buyNumber_OCR_result)))
                    sell_OCR=np.empty((0,len(sell_OCR_result)))
                    sellNumber_OCR=np.empty((0,len(sellNumber_OCR_result)))
                last_1=episode_list[c].rfind('/')#/はubuntu \\はwindows
                last_2=episode_list[c].rfind('.')
                now_time=np.append(now_time,int(episode_list[c][last_1 + 1:last_2]))#これは取得したときの時間
                time_OCR=np.append(time_OCR,time_OCR_result.reshape(1,-1),axis=0)
                value_OCR=np.append(value_OCR,value_OCR_result.reshape(1,-1),axis=0)
                volume_OCR=np.append(volume_OCR,volume_OCR_result.reshape(1,-1),axis=0)
                price_OCR=np.append(price_OCR,price_OCR_result.reshape(1,-1),axis=0)
                nowPriceIndex_OCR=np.append(nowPriceIndex_OCR,nowPriceIndex_OCR_result)
                buy_OCR=np.append(buy_OCR,buy_OCR_result.reshape(1,-1),axis=0)
                buyNumber_OCR=np.append(buyNumber_OCR,buyNumber_OCR_result.reshape(1,-1),axis=0)
                sell_OCR=np.append(sell_OCR,sell_OCR_result.reshape(1,-1),axis=0)
                sellNumber_OCR=np.append(sellNumber_OCR,sellNumber_OCR_result.reshape(1,-1),axis=0)
                endtime=time()
                print(episode_dir[a].name,":",b,"-",c,":",end="")
                print(",処理時間：",endtime-starttime)
            name=output_dir+episode_dir[a].name+"_"+str(b)
            os.mkdir(name)#初回は要注意
            np.save(name+"/now_time.npy",now_time)
            np.save(name+"/time.npy",time_OCR)
            np.save(name+"/value.npy",value_OCR)
            np.save(name+"/volume.npy",volume_OCR)
            np.save(name+"/price.npy",price_OCR)
            np.save(name+"/nowPriceIndex.npy",nowPriceIndex_OCR)
            np.save(name+"/buy.npy",buy_OCR)
            np.save(name+"/buyNumber.npy",buyNumber_OCR)
            np.save(name+"/sell.npy",sell_OCR)
            np.save(name+"/sellNumber.npy",sellNumber_OCR)

path_name=os.path.dirname(os.path.abspath(__file__))
episode_dir=path_name+"/OCR_result/"
episode_dir = list(Path(episode_dir).glob('*'))#各フォルダ名取得
print("並列処理の為、分けます")

#サーバーのコア数に応じて手動で変更
vCPU=[
    [],
    [],
    [],
    []
]
maxIdx=len(vCPU)
#vCPUの数分エピソードを分割する
for idx in range(0,len(episode_dir),maxIdx):
    try:
        vCPU[0].append(episode_dir[idx])
        vCPU[1].append(episode_dir[idx+1])
        vCPU[2].append(episode_dir[idx+2])
        vCPU[3].append(episode_dir[idx+3])
    except:
        break
processes = []
for cpu in vCPU:
    process = Process(target=OCRaction, args=(cpu,))
    processes.append(process)
    process.start()

for process in processes:
    process.join()
