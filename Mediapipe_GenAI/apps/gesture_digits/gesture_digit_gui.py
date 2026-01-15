import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageTk, ImageOps
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# === 設定參數 ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 往上兩層找到 models 資料夾
MODEL_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'models', 'gesture_recognizer.task'))
MNIST_MODEL_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'models', 'mnist_cnn.pth'))
CONFIDENCE_THRESHOLD = 0.5
BRUSH_COLOR = (0, 255, 255)  # 黃色畫筆
BRUSH_THICKNESS = 15

# === 1. 定義 MNIST 模型架構 ===
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

# === 2. GUI 應用程式 ===
class GestureGUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI 手勢手寫辨識系統")
        self.root.geometry("1000x700") # 加大一點以容納預覽
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # 初始化變數
        self.prev_x, self.prev_y = 0, 0
        self.is_drawing = False # 繪圖狀態
        self.canvas_mask = np.zeros((480, 640, 3), dtype=np.uint8) # 繪圖層
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 載入模型
        self.init_models()

        # 建立 UI
        self.create_widgets()

        # 啟動攝影機
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.update_frame()

    def init_models(self):
        # 1. MNIST
        self.mnist_model = ConvNet().to(self.device)
        if os.path.exists(MNIST_MODEL_PATH):
            self.mnist_model.load_state_dict(torch.load(MNIST_MODEL_PATH, map_location=self.device))
            self.mnist_model.eval()
            print("MNIST 模型載入成功")
        else:
            print(f"警告: 找不到 MNIST 模型 {MNIST_MODEL_PATH}")

        # 2. MediaPipe
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=CONFIDENCE_THRESHOLD,
            min_hand_presence_confidence=CONFIDENCE_THRESHOLD,
            min_tracking_confidence=CONFIDENCE_THRESHOLD,
            result_callback=self.mp_callback
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        self.latest_result = None

    def mp_callback(self, result, output_image, timestamp_ms):
        self.latest_result = result

    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左側：視訊顯示區
        self.video_label = ttk.Label(main_frame)
        self.video_label.grid(row=0, column=0, rowspan=5, padx=10, pady=10)

        # 右側：控制區
        control_frame = ttk.LabelFrame(main_frame, text="控制面板")
        control_frame.grid(row=0, column=1, sticky="ns", padx=10, pady=10)

        # 狀態顯示
        self.lbl_status = ttk.Label(control_frame, text="狀態: ✋ 暫停中 (只移動游標)", font=("微軟正黑體", 12))
        self.lbl_status.pack(pady=10, fill=tk.X)

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

        # 繪圖控制按鈕 (Toggle)
        self.btn_toggle_draw = tk.Button(control_frame, text="開始繪圖 (Start)", bg="lightgreen", font=("Arial", 12, "bold"), command=self.toggle_drawing)
        self.btn_toggle_draw.pack(fill=tk.X, pady=10, ipady=10)

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

        # 辨識結果
        self.lbl_result = ttk.Label(control_frame, text="預測數字: ?", font=("Arial", 36, "bold"), foreground="blue")
        self.lbl_result.pack(pady=5)
        
        self.lbl_conf = ttk.Label(control_frame, text="信心度: 0.0%", font=("Arial", 12))
        self.lbl_conf.pack(pady=5)

        # --- 新增：AI 視野預覽 ---
        ttk.Label(control_frame, text="AI 視野 (28x28):").pack(pady=(10, 0))
        self.lbl_debug_img = ttk.Label(control_frame, relief="solid")
        self.lbl_debug_img.pack(pady=5)
        # 初始化一個空的黑色圖片
        empty_img = Image.new('L', (100, 100), 0)
        self.tk_debug_img = ImageTk.PhotoImage(empty_img)
        self.lbl_debug_img.configure(image=self.tk_debug_img)
        # -----------------------

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

        # 功能按鈕
        btn_recognize = ttk.Button(control_frame, text="✨ 辨識 (Recognize)", command=self.recognize_digit)
        btn_recognize.pack(fill=tk.X, pady=5, ipady=10)

        btn_clear = ttk.Button(control_frame, text="🗑️ 清除畫布 (Clear)", command=self.clear_canvas)
        btn_clear.pack(fill=tk.X, pady=5, ipady=10)
        
        ttk.Label(control_frame, text="操作說明:\n1. 點擊「開始繪圖」按鈕\n2. 移動食指指尖即可寫字\n3. 點擊「暫停」可移動不寫字\n4. 點擊「辨識」查看結果", 
                  wraplength=200, foreground="gray").pack(side=tk.BOTTOM, pady=10)

    def toggle_drawing(self):
        self.is_drawing = not self.is_drawing
        if self.is_drawing:
            self.btn_toggle_draw.config(text="暫停繪圖 (Pause)", bg="#ffcccb") # 淺紅色
            self.lbl_status.config(text="狀態: ✍️ 繪圖中", foreground="green")
            # 重置起點，避免一點下去就連線到上次的位置
            self.prev_x, self.prev_y = 0, 0
        else:
            self.btn_toggle_draw.config(text="開始繪圖 (Start)", bg="lightgreen")
            self.lbl_status.config(text="狀態: ✋ 暫停中 (只移動游標)", foreground="black")
            self.prev_x, self.prev_y = 0, 0

    def update_frame(self):
        success, frame = self.cap.read()
        if success:
            # 1. 影像前處理
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 2. MediaPipe 推論
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp = time.time_ns() // 1_000_000
            self.recognizer.recognize_async(mp_image, timestamp)

            # 3. 處理辨識結果
            if self.latest_result and self.latest_result.hand_landmarks:
                hand_landmarks = self.latest_result.hand_landmarks[0]
                
                # 取得食指指尖座標 (Index Tip: 8)
                cx = int(hand_landmarks[8].x * w)
                cy = int(hand_landmarks[8].y * h)
                
                # 判斷是否繪圖 (完全由按鈕狀態決定)
                if self.is_drawing:
                    if self.prev_x == 0 and self.prev_y == 0:
                        self.prev_x, self.prev_y = cx, cy
                    
                    # 畫在 mask 上
                    cv2.line(self.canvas_mask, (self.prev_x, self.prev_y), (cx, cy), BRUSH_COLOR, BRUSH_THICKNESS)
                    self.prev_x, self.prev_y = cx, cy
                    
                    # 游標顏色: 綠色 (繪圖中)
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
                else:
                    # 暫停模式，重置筆畫起點
                    self.prev_x, self.prev_y = 0, 0
                    # 游標顏色: 灰色 (暫停中)
                    cv2.circle(frame, (cx, cy), 10, (100, 100, 100), -1)
                
            else:
                # 未偵測到手部
                self.prev_x, self.prev_y = 0, 0

            # 4. 影像疊加 (將畫布疊加到鏡頭畫面)
            # 建立遮罩
            gray_mask = cv2.cvtColor(self.canvas_mask, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_mask, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            
            # 背景 (挖空)
            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            # 前景 (線條)
            frame_fg = cv2.bitwise_and(self.canvas_mask, self.canvas_mask, mask=mask)
            # 合併
            final_frame = cv2.add(frame_bg, frame_fg)

            # 5. 轉換為 Tkinter 格式並顯示
            img = Image.fromarray(cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        # 遞迴呼叫
        self.root.after(10, self.update_frame)

    def clear_canvas(self):
        self.canvas_mask = np.zeros((480, 640, 3), dtype=np.uint8)
        self.lbl_result.config(text="預測數字: ?")
        self.lbl_conf.config(text="信心度: 0.0%")
        # 清除時自動暫停，避免馬上又畫上去 (可選)
        if self.is_drawing:
            self.toggle_drawing()

    def recognize_digit(self):
        # 1. 取得畫布內容 (轉灰階)
        gray_canvas = cv2.cvtColor(self.canvas_mask, cv2.COLOR_BGR2GRAY)
        
        # 2. 找到數字的最小矩形 (Bounding Box)
        coords = cv2.findNonZero(gray_canvas)
        if coords is None:
            return # 畫布是空的

        x, y, w, h = cv2.boundingRect(coords)
        
        # 裁切出數字部分
        digit_crop = gray_canvas[y:y+h, x:x+w]
        
        if digit_crop.size == 0:
            return

        # 3. 轉換為 PIL Image 進行縮放處理
        pil_img = Image.fromarray(digit_crop)
        
        # 計算縮放比例，讓長邊變成 20 pixel (MNIST 標準是數字在 20x20 內)
        # 這樣可以留 4 pixel 的邊框 ( (28-20)/2 = 4 )
        max_side = max(w, h)
        scale = 20.0 / max_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        if new_w <= 0 or new_h <= 0:
            return
            
        # 縮放 (使用 High quality resizing)
        pil_img_resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 4. 建立 28x28 黑色背景
        final_img = Image.new('L', (28, 28), 0)
        
        # 5. 將縮小的數字貼在正中間 (Center of Mass 的簡化版：幾何中心)
        paste_x = (28 - new_w) // 2
        paste_y = (28 - new_h) // 2
        final_img.paste(pil_img_resized, (paste_x, paste_y))

        # --- 更新 Debug 預覽視窗 ---
        # 放大顯示，讓使用者看清楚
        debug_view = final_img.resize((100, 100), Image.Resampling.NEAREST)
        self.tk_debug_img = ImageTk.PhotoImage(debug_view)
        self.lbl_debug_img.configure(image=self.tk_debug_img)
        # --------------------------

        # 6. 轉為 Tensor 準備推論
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        img_tensor = transform(final_img).unsqueeze(0).to(self.device)

        # 7. 推論
        with torch.no_grad():
            output = self.mnist_model(img_tensor)
            probabilities = F.softmax(output, dim=1)
            pred_prob, pred_label = torch.max(probabilities, 1)

        digit = str(pred_label.item())
        prob = pred_prob.item() * 100

        self.lbl_result.config(text=f"預測數字: {digit}")
        self.lbl_conf.config(text=f"信心度: {prob:.2f}%")

    def on_close(self):
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = GestureGUIApp(root)
    root.mainloop()
