import datetime
import os
from tkinter import messagebox
import numpy as np
from PIL import ImageGrab
import cv2
import tensorflow as tf


class OCR:
    """OCR処理(画像分類)"""
    #StepValue.npy
    #StepValue[変数名,行,列]の3次元配列
    def __init__(self) -> None:
        #各機能の使用可否を最後に出力する
        outputtext=""
        outputtext+="OCRクラスを読み込みます\n"
        path_name=os.path.dirname(os.path.abspath(__file__))
        self.point_x=0
        self.point_y=0
        self.nowprice_OCR_result=0
        outputtext+=" OCR機能:"
        try:
            self.OCRmodel = tf.keras.models.load_model(path_name+"/MNIST/modelSGD.h5")
            outputtext+="使用可能!\n"
        except:
            outputtext+="使用不可\n"
        print(outputtext)
    #メインウィンドウの画像を取得
    #"1"：左側ウィンドウ,"2"：右側ウィンドウ,"3"：全画面ウィンドウ
    def MainWindowImage(self,item):
        #どこのウィンドウを取得するか設定する
        if item=="1":
            x1=0
            x2=960
        elif item=="2":
            x1=960
            x2=1920
        elif item=="3":
            x1=0
            x2=1920
        else:
            print("メインウィンドウ取得で設定のない画面を取得しようとしています")
            return None
        img=ImageGrab.grab(bbox=(x1,0,x2,1080))
        img=np.array(img)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        return img
    #二値化処理（グレースケール込み）
    def threshold(self,img):
        img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        _,img=cv2.threshold(img,140,255,cv2.THRESH_BINARY)
        return img
    #recordは全画面録画のみ
    def Record(self,savefolder):
        img=self.MainWindowImage("3")
        fname=datetime.datetime.now().strftime('%H%M%S')
        cv2.imwrite(savefolder+"/"+fname+".jpg",img)
    #座標情報をnpyファイルから取得
    def CoordinatesAccomodate(self,folder,img,flag,windowname):
        #npyファイルから座標情報取得
        coordinate=np.load(folder)
        #画像上に四角を表示の為の処理
        if flag:
            img=np.array(img)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            for a in range(coordinate.shape[0]):
                for b in range(coordinate.shape[1]):
                    cv2.rectangle(img,(coordinate[a][b][0],coordinate[a][b][1]),(coordinate[a][b][2],coordinate[a][b][3]),(255, 0, 0))
            cv2.imshow(windowname,img)
            cv2.waitKey(0)
        return coordinate
    def output_ocr(self,mstimg,coordinate,isTime_OCR:bool,abnormal_contour_flag:bool):
        margin=10#余白
        draw_img=mstimg.copy()#マスタ画像をコピー
        if coordinate.ndim==1:coordinate=coordinate.reshape((1, 4))#OCR数が1の場合、一次元配列で入力されるので二次元配列に変換
        ocr_result=np.zeros((coordinate.shape[0]))#OCR結果格納
        for a in range(coordinate.shape[0]):
            #座標取得
            top=coordinate[a][1]
            bottom=coordinate[a][3]
            left=coordinate[a][0]
            right=coordinate[a][2]
            Cutimg=draw_img[top+1:bottom-2,left+1:right-2].copy()#+1とか-2とかは余白線対策
            Cutimg=cv2.resize(Cutimg,(0, 0), fx=5, fy=5)#画像拡大
            contours, _=cv2.findContours(Cutimg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)#輪郭検出
            if not contours:#輪郭がない場合は処理を中止する
                if abnormal_contour_flag:
                    ocr_result[a]=0#全板買い売りなどは輪郭が無い場合は強制的に0にする
                    continue
                else:
                    return None
            contours=[cv2.boundingRect(contour) for contour in contours]
            contours=sorted(contours, key=lambda x: x[0])
            cropped_data = []#トリミングして余白を追加した画像をリストの初期化
            #一文字ずつに切り出し、余白を追加し配列に画像を格納
            for contour in contours:
                x=contour[0]
                y=contour[1]
                w=contour[2]
                h=contour[3]
                img=Cutimg[y:y+h, x:x+w].copy()
                img=cv2.resize(img, (0, 0), fx=1/5, fy=1/5)
                height, width= img.shape
                width =  width // 2#余白は上下左右から入れるので割る2が必要
                height = height // 2
                #輪郭抽出で幅が大きい場合、複数文字の可能性があるため分割して処理する
                if width<5 :
                    img=cv2.copyMakeBorder(img, margin - height, margin - height, margin - width, margin - width,cv2.BORDER_CONSTANT,(0))
                    img=cv2.resize(img,(20,21))
                    cropped_data.append(img)#各輪郭のトリミング画像をリストに格納
                else:
                    roop=width//4#1文字の幅は基本的に4になるので4で割り小数点以下切り捨て
                    errorH,errorW=img.shape
                    for i in range(roop):
                        errorimg=img[0:errorH, (errorW//roop)*(i) : (errorW//roop)*(i+1) ].copy()#要素番号を調べているわけでないからi+1でも問題なし
                        NewerrorH,NewerrorW=errorimg.shape
                        NewerrorH=NewerrorH//2
                        NewerrorW=NewerrorW//2
                        errorimg=cv2.copyMakeBorder(errorimg, margin - NewerrorH, margin - NewerrorH, margin - NewerrorW, margin - NewerrorW,cv2.BORDER_CONSTANT,(0))
                        errorimg=cv2.resize(errorimg,(20,21))
                        cropped_data.append(errorimg)#各輪郭のトリミング画像をリストに格納
            cropped_data =np.array(cropped_data)#画像をnumpy配列に変換
            cropped_data=cropped_data/255
            predictions=self.OCRmodel(cropped_data)
            max_indices = np.argmax(predictions, axis=1)
            max_labels=np.array(["0","1","2","3","4","5","6","7","8","9","10"])[max_indices]
            #if np.count_nonzero(max_labels=="10")>2:return None#小数点が複数存在
            if isTime_OCR:
                max_labels[max_labels=="10"]=""#クラス10相当を除外
            else:
                max_labels[max_labels=="10"]="."#10を全て.に変換
            result = ""
            for label in max_labels:result += label#文字列結合
            try:
                ocr_result[a]=float(result)
            except:
                return None#OCRミス
        return ocr_result

    #現在値位置特定
    def PositionInvestigation(self,mstimg,coordinate):
        top=coordinate[0][1]
        bottom=coordinate[coordinate.shape[0]-1][3]
        left=coordinate[0][0]
        right=coordinate[coordinate.shape[0]-1][2]
        hsv_img=mstimg[top+1:bottom-2,left+1:right-2]
        hsv_img=cv2.cvtColor(hsv_img,cv2.COLOR_RGB2HSV)
        lower_yellow = np.array([50, 100, 100])#20, 100, 100
        upper_yellow = np.array([100, 255, 255])#30, 255, 255
        mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
        cnts,_ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts)==0:
            print(",現在値位置取得NG",end="")
            return None
        # 面積が最大の領域の中心座標を取得する
        max_area = 0
        max_index = -1
        for i in range(len(cnts)):
            area = cv2.contourArea(cnts[i])
            if area > max_area:
                max_area = area
                max_index =i
        cnt = cnts[max_index]
        M = cv2.moments(cnt)
        cy=0
        if M["m00"] != 0:
            cy = int(M["m01"] / M["m00"])
        else:
            return None
        cy+=coordinate[0][1]
        #現在値のセル位置を特定
        for a in range(coordinate.shape[0]):
            result1=cy > coordinate[a][1]
            result2=cy < coordinate[a][3]
            if result1 and result2:
                nowindex=a
        return nowindex
    #座標取得のマウスイベントのコールバック関数
    def mouse_callback(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            userdata.point_x=x
            userdata.point_y=y

    #座標取得
    def GetCoordinate(self,mstimg,textPrint,windowname,element_count):
        X_coord=np.array([],dtype=int)
        Y_coord=np.array([],dtype=int)
        draw_img=mstimg.copy()
        copy_img=mstimg.copy()
        cv2.imshow(windowname,mstimg)
        cv2.setMouseCallback(windowname,OCR.mouse_callback,self)
        a=0
        while a<len(textPrint)*2:
            if a%2==0:
                os.system('cls')
                print("コマンド：[a]線確定[b]一つ前に戻る[r]最初からやり直す")
                print(windowname)
                print(textPrint[int(a/2)],"のセル左上選択")
            else:
                print(textPrint[int(a/2)],"のセル右上選択")
            while True:
                #登録済みのX座標を描画
                for lineOutput in X_coord:
                    cv2.line(draw_img,(lineOutput,100),(lineOutput,200),(255, 0, 0), 1, cv2.LINE_AA)
                if not self.point_x==0 and not self.point_y==0:
                    cv2.line(draw_img,(self.point_x,self.point_y),(self.point_x,self.point_y+100),(0, 0, 255), 1, cv2.LINE_AA)#登録前の線を描画
                cv2.imshow(windowname,draw_img)
                key = cv2.waitKey(1)
                if key == 97:#[a]
                    X_coord=np.append(X_coord,self.point_x)
                    self.point_x=0
                    self.point_y=0
                    a+=1
                    break
                elif key==98:#[b]
                    #配列に要素があるなら最後の値を削除
                    if len(X_coord)>0:
                        X_coord=np.delete(X_coord,-1)
                        self.point_x=0#確定したか分からなくなるから縦棒を0,0に移す
                        self.point_y=0
                        a-=1
                        break
                elif key==114:#[r]
                    #配列の要素を全て削除（初期化）
                    X_coord=np.array([],dtype=int)
                    self.point_x=0
                    self.point_y=0
                    a=0
                    break
                draw_img=copy_img.copy()
            draw_img=copy_img.copy()
        print("各セルの左上・下の順で選択")
        print("コマンド：[a]線確定[b]一つ前に戻る[r]最初からやり直す[e]終了")
        #Y座標
        while True:
            #登録済みのY座標を描画
            for lineOutput in Y_coord:
                cv2.line(draw_img,(X_coord[0],lineOutput),(X_coord[0]+100,lineOutput),(255, 0, 0), 1, cv2.LINE_AA)
            if not self.point_x==0 and not self.point_y==0:
                cv2.line(draw_img,(self.point_x,self.point_y),(self.point_x+100,self.point_y),(0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow(windowname,draw_img)
            key = cv2.waitKey(1)
            if key == 97:#[a]
                if len(Y_coord)+1>element_count+1:#Y座標は要素数+1必要
                    messagebox.showerror('エラー', f'要素数が多いです。最大要素数は{element_count}です。')
                else:
                    Y_coord=np.append(Y_coord,self.point_y)
                    self.point_x=0#確定したか分からなくなるから縦棒を0,0に移す
                    self.point_y=0
            elif key==98:#[b]
                #配列に要素があるなら最後の値を削除
                if len(Y_coord)>0:
                    Y_coord=np.delete(Y_coord,-1)
                    self.point_x=0#確定したか分からなくなるから縦棒を0,0に移す
                    self.point_y=0
            elif key==114:#[r]
                #配列の要素を全て削除（初期化）
                Y_coord=np.array([],dtype=int)
            elif key==101:#[e]
                if len(Y_coord)==element_count+1:
                    self.point_x=0
                    self.point_y=0
                    break
                else:
                    messagebox.showerror('エラー', "要素数が "+str(35-len(Y_coord))+" つ足りていません")
            draw_img=copy_img.copy()
        cv2.destroyWindow(windowname)
        #座標格納
        coordinate=np.zeros((int(len(textPrint)),len(Y_coord)-1,4),dtype=int)
        ndim1=0
        for x in range(0,len(X_coord)-1,2):
            ndim2=0
            for y in range(len(Y_coord)-1):
                coordinate[ndim1][ndim2][0]=X_coord[x]#x1
                coordinate[ndim1][ndim2][1]=Y_coord[y]#y1
                coordinate[ndim1][ndim2][2]=X_coord[x+1]#x2
                coordinate[ndim1][ndim2][3]=Y_coord[y+1]#y2
                ndim2+=1
            ndim1+=1
        img=mstimg.copy()
        #最後の確認のための描画
        for x in range(coordinate.shape[0]):
            for y in range(coordinate.shape[1]):
                x1,y1=coordinate[x][y][0],coordinate[x][y][1]
                x2,y2=coordinate[x][y][2],coordinate[x][y][3]
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),1,cv2.LINE_4,0)
        cv2.imshow("output",img)
        cv2.waitKey(0)
        return coordinate
