import os

import cv2
import torch
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()


user = os.getenv("USER")
pwd = os.getenv("PWD")
ip = os.getenv("IP")

# IPCAM URL with RTSP
ipcam_url = f"rtsp://{user}:{pwd}@{ip}/live/profile.0"

# 連接IPCAM
# 連接IPCAM
cap = cv2.VideoCapture(ipcam_url)

if not cap.isOpened():
    print("Cannot open IPCAM")
    exit()

# 使用YOLOv5模型
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

while True:
    # 擷取畫面
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # 降低畫質
    frame = cv2.resize(frame, (640, 360))  # 調整到 640x360 的分辨率

    # 使用YOLOv5進行物件偵測
    results = model(frame)

    # 取得偵測結果
    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.5:  # 設定信心閾值
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    # 顯示結果
    cv2.imshow("IPCAM Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
