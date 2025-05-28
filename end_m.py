import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch

while True:
    print("要輸入圖片請輸入1，要開始輸入2，要退出輸入3，要訓練模型請輸入4，要錄影訓練請輸入5，要查看人物清單請輸入6")
    a = input()

    if a == "1":
        a = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用裝置: {device}")
        mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

        while True:
            output_dir = "face_data"
            os.makedirs(output_dir, exist_ok=True)
            name = input("輸入姓名: ")
            label = name
            output_dir_photo = "photo"

            for file in os.listdir(output_dir_photo):
                if file.endswith(('.jpg', '.png')):
                    img = os.path.join(output_dir_photo, file)
                    image = cv2.imread(img)
                    image_RBG = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(image_RBG)

                    face = mtcnn(pil_img)
                    if face is not None:
                        a += 1
                        face_embedding = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()
                        np.save(os.path.join(output_dir, f"['{name}']_{a}_embedding.npy"), face_embedding)
                        np.save(os.path.join(output_dir, f"{name}_{a}_label.npy"), label)
                        face_pil = Image.fromarray((face.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                        face_pil.save(os.path.join(output_dir, f"{name}-{a}.jpg"))
                        print(f"已保存圖片與特徵向量，標籤為: {label}\n{a} step")
                    else:
                        print("未發現人臉")

            print("是否要繼續輸入圖片？否請按q，是請按y")
            user_input = input()
            if user_input == "q":
                break
            elif user_input == "y":
                continue

    elif a == "4":
        output_dir_1 = "face_data"
        embeddings, labels, Y = [], [], []

        for file in os.listdir(output_dir_1):
            if file.endswith("_embedding.npy"):
                embedding = np.load(os.path.join(output_dir_1, file))
                label = file.split("_embedding.npy")[0]
                embeddings.append(embedding)
                labels.append(label)
            if file.endswith("_label.npy"):
                Y_ = np.load(os.path.join(output_dir_1, file))
                Y.append(Y_)

        X = np.vstack(embeddings)
        y = np.array(labels)
        print(f"訓練數據集大小: {X.shape}, 標籤數量: {len(y)}")
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(X)
        model = SVC(kernel='linear')
        model.fit(features_scaled, Y)
        print("SVM 模型訓練完成！")
        joblib.dump(model, 'svm_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')

    elif a == "2":
        font = ImageFont.truetype("msjh.ttc", 30)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device, min_face_size=60,
                      thresholds=[0.5, 0.7, 0.8], factor=0.8)
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        model = joblib.load('svm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        cap = cv2.VideoCapture(0)
        print("按 'esc' 退出")

        frame_count, count = 0, 1
        passing_line = 0.7

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            boxes, _ = mtcnn.detect(img)

            if boxes is not None:
                for (x1, y1, x2, y2) in boxes:
                    face = mtcnn(pil_img)
                    if face is not None:
                        embedding2 = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()
                        new_features_scaled = scaler.transform(embedding2)
                        prediction = model.predict(new_features_scaled)[0]
                        embedding1_path = f"face_data/['{prediction}']_{count}_embedding.npy"

                        if os.path.exists(embedding1_path):
                            embedding1 = np.load(embedding1_path)
                            distance = np.linalg.norm(embedding1 - embedding2)
                            if distance <= passing_line:
                                img_draw = ImageDraw.Draw(pil_img)
                                img_draw.text((int(x1), int(y1) - 10), prediction, font=font, fill=(0, 255, 0))
                    frame_count += 1

                frame = np.array(pil_img)
                cv2.imshow("Camera", frame)

            if cv2.waitKey(10) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    elif a == "3":
        break

    elif a == "5":
        a = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用裝置: {device}")
        mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        output_dir = "face_data"
        os.makedirs(output_dir, exist_ok=True)
        name = input("輸入姓名: ")
        label = name
        cap = cv2.VideoCapture(0)
        print("按 's' 抓取圖片，'q' 退出")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            face = mtcnn(pil_img)
            cv2.imshow("Camera", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and face is not None:
                a += 1
                embedding = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()
                np.save(os.path.join(output_dir, f"['{name}']_{a}_embedding.npy"), embedding)
                np.save(os.path.join(output_dir, f"{name}_{a}_label.npy"), label)
                face_pil = Image.fromarray((face.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                face_pil.save(os.path.join(output_dir, f"{name}-{a}.jpg"))
                print(f"已保存圖片與特徵向量，標籤為: {label}\n{a} step")

            elif key == 27 or key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    elif a == "6":
        output_dir = "face_data"
        if not os.path.exists(output_dir):
            print("face_data 資料夾不存在。")
            continue

        embedding_files = [f for f in os.listdir(output_dir) if f.endswith("_embedding.npy")]
        names = set()

        for file in embedding_files:
            if file.startswith("['") and "]_" in file:
                name = file.split("]_")[0][2:]
            else:
                name = file.split("_")[0]
            names.add(name)

        if names:
            print(f"目前共有 {len(names)} 位不同人物的資料：")
            for name in sorted(names):
                print(f" - {name}")
        else:
            print("目前尚無人臉資料。")
