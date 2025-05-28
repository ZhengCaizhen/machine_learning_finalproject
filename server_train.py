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
UDP_PORT = 5665
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化 MTCNN 和 ResNet 模型
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device, min_face_size=60, thresholds=[0.5, 0.7, 0.8], factor=0.8)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
output_dir = "face_data"# 創建保存資料夾
os.makedirs(output_dir, exist_ok=True)



while True:
    # 創建 UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    
    
    # 設置 response 預設值
    

    # 接收數據
    data, addr = sock.recvfrom(65507)# 最大接收大小是 65507 字節
    if data.startswith ("name" ):                         
        name = data
        label = name
    elif data.startswith("END"):
                embeddings = []# 加載特徵向量與標籤
                labels = []
                Y=[]
                for file in os.listdir(output_dir):
                    if file.endswith("_embedding.npy"):
                        embedding = np.load(os.path.join(output_dir, file))
                        label = file.split("label.npy")[0]
                        embeddings.append(embedding)
                        labels.append(label)
                    
                    if file.endswith("_label.npy"):
                        Y_=np.load(os.path.join(output_dir,file))
                        Y.append(Y_)
                # 將列表轉為 numpy 數組
                X = np.vstack(embeddings)  # 特徵向量
                y = np.array(labels)       # 對應的標籤
                print(f"訓練數據集大小: {X.shape}, 標籤數量: {len(y)}")

                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(X)
                print("標準化後數據的形狀:", features_scaled.shape)

                assert X.shape[0] == y.shape[0], "樣本數與標籤數不一致！"
                        #print(Y)
                        # 訓練模型
                model = SVC(kernel='linear')
                model.fit(features_scaled, Y)
                print("SVM 模型訓練完成！")
                joblib.dump(model, 'svm_model.pkl')
                joblib.dump(scaler, 'scaler.pkl')
                break    
    elif data.startswith("quit"):
         break

    else :
        frame = Image.open(io.BytesIO(data))  # 讀取圖片數據
        frame = np.array(frame)  # 轉換為 numpy 陣列
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 轉換為 OpenCV 需要的格式
        pil_img = Image.fromarray(frame)  # 轉回 PIL 物件供 MTCNN 使用

        face = mtcnn(pil_img)
        if face is not None:
            a=a+1
                            
                            #提取特徵向量
            face_embedding =resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()
            np.save(os.path.join(output_dir, f"['{name}']_{a}_embedding.npy"), face_embedding)
                                # 保存特徵向量
            np.save(os.path.join(output_dir, f"{name}_{a}_label.npy"), label)
                            # 保存圖片
            face_pil = Image.fromarray((face.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            face_pil.save(os.path.join(output_dir, f"{name}-{a}.jpg"))

            print(f"已保存圖片與特徵向量，標籤為: {label}\n" + f"{a}step")



sock.close()