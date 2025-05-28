import socket
from PIL import Image
import io
import cv2
# 設定 UDP 目標地址和端口
UDP_IP = "26.203.117.144"
UDP_PORT = 5005
i=0
count_1 = 1
photo = []
# 創建 socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

name = "name"+input("輸入你的名字")
sock.sendto(name, (UDP_IP, UDP_PORT))

cap = cv2.VideoCapture(0)

if not cap.isOpened():
        print("無法訪問攝影機")
        exit()

while True:
        # 捕獲每一幀
        ret, frame = cap.read()
        
        if not ret:
            print("無法獲取視頻幀")
            break
        
        # 顯示攝影機畫面
        cv2.imshow('Camera', frame)

        # 按 'c' 鍵拍照並保存圖片
        if cv2.waitKey(1) & 0xFF == ord('c'):
            
            photo[i] = "IMG"+"captured_image.jpg"
            i += 1
            
            
        
            

        # 按 'ESC' 鍵退出
        key=cv2.waitKey(1) & 0xFF
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()
photo_quantity = len(photo)
for a in range(0,photo_quantity-1):
    image_path = photo[a]
    image = Image.open(image_path)
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_data = image_bytes.getvalue()
    sock.sendto(image_data, (UDP_IP, UDP_PORT))
    print(f"圖片已發送到 {UDP_IP}:{UDP_PORT}")
    # 發送圖片數據



    # 關閉 socket    
sock.close()