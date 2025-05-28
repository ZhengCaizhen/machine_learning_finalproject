# server_tcp_photo.py
import socket
import io
from PIL import Image
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import cv2
import time

face_model = joblib.load('svm_model.pkl')# 加載模型
face_chat = joblib.load('scaler.pkl')# 加載模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# 設定設備
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device, min_face_size=60, thresholds=[0.7, 0.7, 0.8], factor=0.8)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)# 初始化 MTCNN 和 ResNet 模型
output_dir = "login-history-photo"# 設定輸出目錄
os.makedirs(output_dir, exist_ok=True)

HOST = '26.203.117.144'# 伺服器地址
PORT = 5005
passing_line = 0.7# 設定閾值
def runserver():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(2)
    print(f"📡 TCP 伺服器啟動：{HOST}:{PORT} 等待連線...")

    conn, addr = server.accept()


    try:
        print(f"✅ 連線來自：{addr}")
        while True:
            # 第一步：接收 4 bytes 的圖片長度
            length_bytes = conn.recv(4)
            if not length_bytes:
                break
            length = int.from_bytes(length_bytes, byteorder='big')
            print(f"📦 預期接收 {length} bytes")

            # 第二步：接收完整圖片 bytes
            data = b''
            while len(data) < length:
                packet = conn.recv(length - len(data))
                if not packet:
                    break
                data += packet

            print(f"✅ 已接收完整圖片，共 {len(data)} bytes")

            # 第三步：用 PIL 開圖
            image = Image.open(io.BytesIO(data))
            #image.show()
            times=time.strftime("%Y-%m%d-%H-%M-%S")
            image.save(os.path.join(output_dir,f"received-{times}.jpg"))

            print("🎉 圖片成功轉成.jpg！")
        
            response = "處理錯誤"
            count = 1
            image = cv2.imread(os.path.join(output_dir,f"received-{times}.jpg"))
            
            image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image_RGB)
            face_1, _ = mtcnn.detect(image_RGB)
            if face_1 is None:
                    print("未偵測到人臉")
                    response = "未偵測到人臉"
                    conn.sendall((response + '\n').encode('utf-8'))
                    continue
                    
            else:
                    for (x1, y1, x2, y2) in face_1:
                        # 截取人臉區域
                        face = image_RGB[int(y1):int(y2), int(x1):int(x2)]
                        face = mtcnn(pil_img)  # 使用 MTCNN 對人臉進行對齊
                        cv2.rectangle(image_RGB, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        

                        embedding2 = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()  # 提取特徵向量

                        new_features_scaled = face_chat.transform(embedding2)  # 標準化特徵
                        predictions = face_model.predict(new_features_scaled)  # 預測類別
                        while True:
                        # 檢查檔案是否存在
                            embedding_path = f"face_data/{predictions}_{count}_embedding.npy"
                            if not os.path.exists(embedding_path):
                                print("找不到檔案！")
                                response = "找不到人！"
                                print("回應已發送：", response)
                                conn.sendall((response + '\n').encode('utf-8'))
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
                                print("回應已發送：", response)
                                conn.sendall((response + '\n').encode('utf-8'))  # 發送回應給客戶端
                                break

                    
    

        

    except Exception as e:
        print("❌ 發生錯誤：", e)
    finally:
        conn.close()
        server.close()
        print("🔌 伺服器關閉")
        runserver()
        
runserver()        


