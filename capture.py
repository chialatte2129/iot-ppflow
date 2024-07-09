import os
from datetime import datetime

import cv2
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

user = os.getenv("USER")
pwd = os.getenv("PWD")
ip = os.getenv("IP")
ipcam_url = f"rtsp://{user}:{pwd}@{ip}/live/profile.0"

# 連接IPCAM
cap = cv2.VideoCapture(ipcam_url)

if not cap.isOpened():
    print("Cannot open IPCAM")
    exit()

# 擷取畫面
ret, frame = cap.read()

if ret:
    # 生成當前時間的檔名
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
    # 儲存畫面為png
    cv2.imwrite(filename, frame)
    print(f"Image saved as {filename}")
else:
    print("Failed to capture image")

# 釋放資源
cap.release()
