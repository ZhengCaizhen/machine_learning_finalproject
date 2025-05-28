import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
from PIL import Image ,ImageDraw, ImageFont
import torch
while True:
    print("要輸入圖片請輸入1，要開始輸入2，要退出輸入3，要訓練模型請輸入4，要錄影訓練請輸入5")
    a=input()
    if a=="1":
        a=0
# 初始化 MTCNN 和 ResNet 模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用裝置: {device}")
        mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

        while True:
            output_dir = "face_data"# 創建保存資料夾
            os.makedirs(output_dir, exist_ok=True)
            name =input("輸入姓名: ")
            label = name
            output_dir_photo ="photo"#圖片庫
            for file in os.listdir(output_dir_photo):
                if file.endswith('.jpg'):
                    img =os.path.join(output_dir_photo,file)
                    image = cv2.imread(f"{img}")


                    image_RBG =cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(image_RBG)

        #檢測人臉
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
                    else:
                        print("未發現")

                elif file.endswith('.png'):
                    img =os.path.join(output_dir_photo,file)
                    image = cv2.imread(f"{img}")


                    image_RBG =cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(image_RBG)

                    #檢測人臉
                    face = mtcnn(pil_img)
                    if face is not None:
                        a=a+1
                        

                        #提取特徵向量
                        face_embedding =resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()
                        np.save(os.path.join(output_dir, f"['{name}']_{a}_embedding.npy"), face_embedding)
                            # 保存特徵向量
                        np.save(os.path.join(output_dir, f"{name}{a}_label.npy"), label)
                        # 保存圖片
                        face_pil = Image.fromarray((face.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                        face_pil.save(os.path.join(output_dir, f"{name}-{a}.jpg"))

                        print(f"已保存圖片與特徵向量，標籤為: {label}\n" + f"{a}step")
                    else:
                        print("未發現")

            print("是否要繼續輸入圖片？否請按q是請按y")
            user_input = input()
            if user_input=="q":
                output_dir_1= "face_data"
                # 加載特徵向量與標籤
                embeddings = []
                labels = []
                Y=[]
                for file in os.listdir(output_dir_1):
                    if file.endswith("_embedding.npy"):
                        embedding = np.load(os.path.join(output_dir_1, file))
                        label = file.split("label.npy")[0]
                        embeddings.append(embedding)
                        labels.append(label)
                    
                    if file.endswith("_label.npy"):
                        Y_=np.load(os.path.join(output_dir_1,file))
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
            elif user_input=="y":
                continue
                #################################################################################################################################################
        # 資料夾路徑
#output_dir = "face_data_lin_Ting_Hsuan"
    if a=="4":    
        output_dir_1= "face_data"
# 加載特徵向量與標籤
        embeddings = []
        labels = []
        Y=[]
        for file in os.listdir(output_dir_1):
            if file.endswith("_embedding.npy"):
                embedding = np.load(os.path.join(output_dir_1, file))
                label = file.split("label.npy")[0]
                embeddings.append(embedding)
                labels.append(label)
        
            if file.endswith("_label.npy"):
                Y_=np.load(os.path.join(output_dir_1,file))
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
#################################################################################################################################################
    elif a=="2":
        font_path = "msjh.ttc"
        font = ImageFont.truetype(font_path, 30)#設定字體大小
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device,min_face_size=60,thresholds=[0.5, 0.7,0.8], factor=0.8)
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        face_model =joblib.load('svm_model.pkl')
        face_chat =joblib.load('scaler.pkl')
        # 打開攝影機
        cap = cv2.VideoCapture(0)
        
        
        print("按 'esc' 退出")
        frame_count = 0
        output_dir_1= "face_data"
        passing_line =0.7
        count = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                print("無法讀取攝影機畫面")
                break

            # 轉換為 RGB 格式
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            face_1,_ = mtcnn.detect(img)
            
            if face_1 is not None:
                
                for (x1, y1, x2, y2) in face_1:
                    # 截取人臉區域
                    face = frame[int(y1):int(y2), int(x1):int(x2)]
                    face = mtcnn(pil_img)# 使用 MTCNN 對人臉進行對齊
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    
                    if frame_count%5 ==0:
                        embedding2 = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy() # 提取特徵向量
                        
                        new_features_scaled = face_chat.transform(embedding2) # 標準化embedding2 
                        predictions = face_model.predict(new_features_scaled) # 預測類別
                        img_chinses =Image.fromarray(frame)
                        draw = ImageDraw.Draw(img_chinses)
                        #print(f'{predictions}\n')
                        
                        
                        
                        
                        embedding1 =np.load(f"face_data/{predictions}_{count}_embedding.npy")    
                        def euclidean_distance(embedding1, embedding2):#計算歐式距離
                            return np.linalg.norm(embedding1 - embedding2)
                        distance = euclidean_distance(embedding1 , embedding2)
                        print(distance)
                        if distance > passing_line:
                            print(f"核對中{count}")
                            count += 1               
                        else:
                            text = f"{predictions}"
                            print(f'{predictions}\n')
                            position = (int(x1), int(y1) -10)  # 文字位置
                            text_color = (0, 255, 0)  # 藍色 (BGR)
                            draw.text(position, text, font=font, fill=text_color)
                                
                            
                        
                        
                    
                    frame_count += 1
                           
                frame = np.array(img_chinses)
                cv2.imshow("Camera", frame)

            key = cv2.waitKey(10) & 0xFF
            if  frame_count == 100:
                break
        cap.release()
        cv2.destroyAllWindows()
    elif a=="3":
        break

    if a=="5":
        a=0
        # 初始化 MTCNN 和 ResNet 模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用裝置: {device}")
        mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

        # 創建保存資料夾
        output_dir = "face_data"# 創建保存資料夾
        os.makedirs(output_dir, exist_ok=True)
        name =input("輸入姓名: ")
        label = name
        # 打開攝影機
        cap = cv2.VideoCapture(0)
        print("按 's' 抓取圖片並保存，按 'q' 退出程序")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("無法讀取攝影機畫面")
                break

            # 轉換為 RGB 格式
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            
            # 檢測人臉
            face = mtcnn(pil_img)
            
            # 顯示畫面
            cv2.imshow("Camera", frame)
            
            # 保存圖片與特徵向量
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and face is not None :
                a=a+1
                face_embedding =resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()
                np.save(os.path.join(output_dir, f"['{name}']_{a}_embedding.npy"), face_embedding)
                    # 保存特徵向量
                np.save(os.path.join(output_dir, f"{name}_{a}_label.npy"), label)
                # 保存圖片
                face_pil = Image.fromarray((face.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                face_pil.save(os.path.join(output_dir, f"{name}-{a}.jpg"))

                print(f"已保存圖片與特徵向量，標籤為: {label}\n" + f"{a}step")
          
            elif key == 27 or key == ord('q'):
                output_dir_1= "face_data"
                # 加載特徵向量與標籤
                embeddings = []
                labels = []
                Y=[]
                for file in os.listdir(output_dir_1):
                    if file.endswith("_embedding.npy"):
                        embedding = np.load(os.path.join(output_dir_1, file))
                        label = file.split("label.npy")[0]
                        embeddings.append(embedding)
                        labels.append(label)
                    
                    if file.endswith("_label.npy"):
                        Y_=np.load(os.path.join(output_dir_1,file))
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


        cap.release()
        cv2.destroyAllWindows() 

    if a == "6":
        output_dir = "face_data"
        if not os.path.exists(output_dir):
            print("face_data 資料夾不存在。")
            continue

        embedding_files = [f for f in os.listdir(output_dir) if f.endswith("_embedding.npy")]
        names = set()

        for file in embedding_files:
            if file.startswith("['") and "']_" in file:
                name = file.split("']_")[0][2:]  # 去掉 [' 和 ']_
            else:
                name = file.split("_")[0]  # 處理沒有 [] 的情況
            names.add(name)

        if names:
            print(f"目前共有 {len(names)} 位不同人物的資料：")
            for name in sorted(names):
                print(f" - {name}")
        else:
            print("目前尚無人臉資料。")