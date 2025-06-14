# 機器學習期末專題 — 人臉辨識系統

## 專案介紹
本專題為人臉辨識系統，主要使用以下技術：  
- 人臉檢測：MTCNN  
- 特徵提取：InceptionResnetV1  
- 分類模型：支援向量機（SVM）

系統可透過手動輸入圖片或攝影機輸入進行人物照片新增。

## 專案架構
- `end.py`：主程式，直接執行此檔案即可啟動辨識流程  
- `photo_person/`：存放之前手動加入訓練的人臉資料  
- `face_data/`：儲存訓練好的 SVM 模型與特徵

## 環境建立與執行說明

請先安裝 Python 3.9 版本，並依照以下步驟建立虛擬環境並執行專案：

1. 建立虛擬環境：
```bash
py -3.9 -m venv face-venv
```

2. 啟動虛擬環境（Windows）：
```
face-venv\Scripts\activate
```
3. 更新 pip:
```
python.exe -m pip install --upgrade pip
```
4. 安裝專案所需套件：
```
pip install -r requirements.txt
```
5. 若已離開虛擬環境，請重新啟動：
```
face-venv\Scripts\activate
```
6. 切換磁碟機並進入專案資料夾：
```
cd 你的程式所在資料夾
```
7. 執行主程式：
```
python end.py
```

#可以從這裡直接複製
```
py -3.9 -m venv face-venv
face-venv\Scripts\activate
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
face-venv\Scripts\activate
D:
cd 程式所在資料夾
python end.py
```

## 功能說明

本專案提供以下主要功能：

1. **輸入圖片**  
   如果要加入新人物的圖片，將圖片檔須為.jpg或.png，以數字依序命名放入 `photo` 資料夾，再執行 功能1: 從 `photo` 資料夾中讀取圖片，將資料加入模型訓練。

2. **開始辨識**  
   對輸入的人臉進行辨識，並顯示歐式距離來判斷辨識結果的相似度。

3. **退出程式**  
   結束辨識系統的執行。

4. **訓練模型**  
   使用現有資料進行 SVM 模型的訓練，並顯示模型準確率。

5. **錄影訓練**  
   利用相機錄影，擷取人臉照片以增強訓練資料，進行模型訓練。

6. **顯示人物資料數量**  
   顯示目前資料庫中已收集的人物總數，以每個人的照片數量。
