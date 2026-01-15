"""
手勢手寫數字辨識 (Gesture Digit Recognizer)

功能：
1. 使用 MediaPipe 偵測手勢與食指位置。
2. 在虛擬畫布上書寫數字。
3. 使用 PyTorch MNIST 模型辨識手寫數字。

操作說明：
- 食指向上 (Pointing_Up): 繪圖模式 (Draw)
- 握拳 (Closed_Fist): 暫停繪圖 (Hover)
- 手掌張開 (Open_Palm): 清除畫布 (Clear)
- 拇指向上 (Thumb_Up): 執行辨識 (Recognize)
- 按鍵 'c': 清除畫布
- 按鍵 'q' / 'ESC': 離開
"""

import cv2
import mediapipe as mp
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# === 設定參數 ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model', 'gesture_recognizer.task')
MNIST_MODEL_PATH = os.path.join(SCRIPT_DIR, 'model', 'mnist_cnn.pth')
CONFIDENCE_THRESHOLD = 0.5

# 繪圖參數
BRUSH_COLOR = (255, 255, 255)  # 白色
BRUSH_THICKNESS = 15
CANVAS_SIZE = (480, 640) # H, W (與攝影機一致)

# === 1. 定義 MNIST 模型架構 (需與訓練時一致) ===
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# === 2. 初始化全域變數 ===
recognition_result = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mnist_model = None
prev_x, prev_y = 0, 0

# === 3. 輔助函式 ===
def load_mnist_model():
    global mnist_model
    try:
        model = ConvNet().to(device)
        # 載入權重
        if os.path.exists(MNIST_MODEL_PATH):
            model.load_state_dict(torch.load(MNIST_MODEL_PATH, map_location=device))
            model.eval()
            print(f"MNIST 模型載入成功: {MNIST_MODEL_PATH}")
            return model
        else:
            print(f"錯誤: 找不到 MNIST 模型檔案: {MNIST_MODEL_PATH}")
            return None
    except Exception as e:
        print(f"載入模型時發生錯誤: {e}")
        return None

def save_result(result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global recognition_result
    recognition_result = result

def get_finger_pos(hand_landmarks, width, height):
    # 食指指尖 (Index Finger Tip) 是索引 8
    x = int(hand_landmarks[8].x * width)
    y = int(hand_landmarks[8].y * height)
    return x, y

def predict_digit(canvas_img, model):
    if model is None:
        return "Model Error", 0.0

    # 1. 轉為 PIL Image
    # canvas_img 是黑底白線 (numpy array), shape (480, 640, 3) 或 (480, 640)
    if len(canvas_img.shape) == 3:
        img = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
    else:
        img = canvas_img
    
    pil_img = Image.fromarray(img)

    # 2. 預處理 (Resize -> ToTensor -> Normalize)
    # 這裡我們需要找到數字的 bounding box 並裁切，以獲得更好的辨識效果
    # 否則 28x28 會包含大量空白
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # 增加一些 padding
        pad = 20
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(img.shape[1] - x, w + 2 * pad)
        h = min(img.shape[0] - y, h + 2 * pad)
        
        # 確保是正方形 (以長邊為準)
        if w > h:
            y_center = y + h // 2
            h = w
            y = max(0, y_center - h // 2)
        else:
            x_center = x + w // 2
            w = h
            x = max(0, x_center - w // 2)

        crop_img = pil_img.crop((x, y, x+w, y+h))
    else:
        crop_img = pil_img # 全黑

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    img_tensor = transform(crop_img).unsqueeze(0).to(device)

    # 3. 推論
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        pred_prob, pred_label = torch.max(probabilities, 1)
        
    return str(pred_label.item()), pred_prob.item()

# === 主程式 ===
def main():
    global recognition_result, prev_x, prev_y, mnist_model

    # 載入 MNIST 模型
    mnist_model = load_mnist_model()
    
    # 初始化 MediaPipe Gesture Recognizer
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=CONFIDENCE_THRESHOLD,
        min_hand_presence_confidence=CONFIDENCE_THRESHOLD,
        min_tracking_confidence=CONFIDENCE_THRESHOLD,
        result_callback=save_result
    )

    # 建立畫布 (全黑)
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 預測結果顯示變數
    last_prediction = ""
    last_confidence = 0.0
    status_text = "Init..."
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("=== 手勢手寫數字辨識系統啟動 ===")
    print("操作：食指向上(畫), 握拳(停), 手掌張開(清空), 拇指向上(辨識)")

    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # 翻轉鏡頭 (Selfie mode)
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # 轉換給 MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # 非同步辨識
            timestamp = time.time_ns() // 1_000_000
            recognizer.recognize_async(mp_image, timestamp)

            # 處理辨識結果
            current_gesture = "None"
            
            if recognition_result and recognition_result.gestures:
                # 取得手勢名稱
                current_gesture = recognition_result.gestures[0][0].category_name
                
                # 取得食指位置
                if recognition_result.hand_landmarks:
                    hand_landmarks = recognition_result.hand_landmarks[0]
                    cx, cy = get_finger_pos(hand_landmarks, w, h)
                    
                    # 繪製食指游標
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 255), -1)

                    # 根據手勢執行動作
                    if current_gesture == "Pointing_Up":
                        status_text = "Drawing Mode"
                        if prev_x == 0 and prev_y == 0:
                            prev_x, prev_y = cx, cy
                        
                        # 在畫布上繪畫
                        cv2.line(canvas, (prev_x, prev_y), (cx, cy), BRUSH_COLOR, BRUSH_THICKNESS)
                        prev_x, prev_y = cx, cy
                        
                    elif current_gesture == "Closed_Fist":
                        status_text = "Hover Mode"
                        prev_x, prev_y = 0, 0
                        
                    elif current_gesture == "Open_Palm":
                        status_text = "Clear Canvas"
                        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                        prev_x, prev_y = 0, 0
                        last_prediction = ""
                        
                    elif current_gesture == "Thumb_Up":
                        status_text = "Recognizing..."
                        # 避免連續快速觸發，可加一個簡單的冷卻機制 (這裡簡化處理)
                        digit, conf = predict_digit(canvas, mnist_model)
                        last_prediction = digit
                        last_confidence = conf
                        prev_x, prev_y = 0, 0
                    
                    else:
                        status_text = f"Gesture: {current_gesture}"
                        prev_x, prev_y = 0, 0
            else:
                status_text = "No Hand Detected"
                prev_x, prev_y = 0, 0

            # 融合畫面：將畫布疊加到攝影機畫面上
            # 建立遮罩
            gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            
            # 前景(畫線)
            fg = cv2.bitwise_and(canvas, canvas, mask=mask)
            # 背景(攝影機畫面挖空)
            bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            
            # 合併
            final_frame = cv2.add(bg, fg)

            # 顯示資訊 UI
            # 1. 左上角：當前狀態
            cv2.rectangle(final_frame, (0, 0), (250, 100), (0, 0, 0), -1) # 背景黑框
            cv2.putText(final_frame, f"Mode: {status_text}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(final_frame, f"Gesture: {current_gesture}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # 2. 右側：辨識結果
            if last_prediction:
                cv2.putText(final_frame, f"Digit: {last_prediction}", (10, 140), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 100, 255), 3)
                cv2.putText(final_frame, f"Conf: {last_confidence:.2%}", (10, 180), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)

            # 顯示小地圖 (Canvas Preview) - 可選
            # canvas_small = cv2.resize(canvas, (160, 120))
            # final_frame[360:480, 480:640] = canvas_small
            # cv2.rectangle(final_frame, (480, 360), (640, 480), (255, 255, 255), 2)

            cv2.imshow('Gesture Digit Recognizer', final_frame)

            # 鍵盤控制
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'): # ESC or q
                break
            elif key == ord('c'):
                canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                last_prediction = ""

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
