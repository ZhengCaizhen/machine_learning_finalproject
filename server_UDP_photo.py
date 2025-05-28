import cv2
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import socket
import io
from PIL import Image

# 設定 UDP 服務器地址和端口26.203.117.144
UDP_IP = "26.203.117.144"
UDP_PORT = 5005
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化 MTCNN 和 ResNet 模型
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device, min_face_size=60, thresholds=[0.5, 0.7, 0.8], factor=0.8)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 加載模型
face_model = joblib.load('svm_model.pkl')
face_chat = joblib.load('scaler.pkl')

passing_line = 0.7

while True:
    # 創建 UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    print(f"正在等待來自 {UDP_IP} 的圖片...")
    
    # 設置 response 預設值
    response = "處理錯誤"
    count = 1
    # 接收數據
    data, addr = sock.recvfrom(65507)  # 最大接收大小是 65507 字節

    # 將接收到的數據轉換為圖片
    frame = Image.open(io.BytesIO(data))  # 讀取圖片數據
    frame = np.array(frame)  # 轉換為 numpy 陣列
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 轉換為 OpenCV 需要的格式
    pil_img = Image.fromarray(frame)  # 轉回 PIL 物件供 MTCNN 使用

    # 偵測臉部
    face_1, _ = mtcnn.detect(frame)

    if face_1 is None:
        print("未偵測到人臉")
        response = "未偵測到人臉"
        sock.sendto(response.encode(), addr)
        continue
        
    else:
        for (x1, y1, x2, y2) in face_1:
            # 截取人臉區域
            face = frame[int(y1):int(y2), int(x1):int(x2)]
            face = mtcnn(pil_img)  # 使用 MTCNN 對人臉進行對齊
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            

            embedding2 = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()  # 提取特徵向量

            new_features_scaled = face_chat.transform(embedding2)  # 標準化特徵
            predictions = face_model.predict(new_features_scaled)  # 預測類別
            while True:
            # 檢查檔案是否存在
                embedding_path = f"face_data/{predictions}_{count}_embedding.npy"
                if not os.path.exists(embedding_path):
                    print("找不到檔案！")
                    response = "找不到！"
                    break

                embedding1 = np.load(embedding_path)

                def euclidean_distance(embedding1, embedding2):  # 計算歐式距離
                    return np.linalg.norm(embedding1 - embedding2)

                distance = euclidean_distance(embedding1, embedding2)
                print(f"歐式距離: {distance}")

                if distance > passing_line:  # 比對結果
                    print(f"核對中 {count}")
                    print(f'預測: {predictions}\n')
                    count += 1
                    
                else:
                    print(f'辨識成功: {predictions}\n')
                    response = predictions[0]  # 轉成字串
                    count = 1
                    break

    # 顯示圖片
    
    cv2.destroyAllWindows()
    # 送出 response 給客戶端
    sock.sendto(response.encode(), addr)
    
    
    

# 關閉 socket
    sock.close()