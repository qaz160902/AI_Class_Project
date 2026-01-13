"""
食指水果忍者 (Fruit Ninja Style)
- 水果會從底部拋出
- 用食指指尖「切」過水果即可引爆得分
- 加入粒子爆炸特效
"""

import cv2
import mediapipe as mp
import time
import os
import math
import random
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# === 設定參數 ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model', 'gesture_recognizer.task')
CONFIDENCE_THRESHOLD = 0.5

# 遊戲參數
GRAVITY = 0.6          # 重力
FRUIT_SPAWN_RATE = 40  # 每幾幀產生一個水果 (越小越快)
FRUIT_RADIUS = 25

# 顏色庫 (BGR 格式)
COLORS = [
    (0, 0, 255),    # 紅 (蘋果)
    (0, 255, 255),  # 黃 (檸檬)
    (0, 255, 0),    # 綠 (西瓜)
    (128, 0, 128),  # 紫 (葡萄)
    (0, 165, 255)   # 橘 (橘子)
]

# === 類別定義 ===

class Fruit:
    def __init__(self, screen_w, screen_h):
        self.x = random.randint(50, screen_w - 50)
        self.y = screen_h + FRUIT_RADIUS  # 從底部生成
        # 隨機拋射速度
        self.vx = random.uniform(-4, 4)       # 水平隨機
        self.vy = random.uniform(-18, -25)    # 垂直向上衝力
        self.color = random.choice(COLORS)
        self.radius = FRUIT_RADIUS
        self.active = True

    def update(self):
        self.vy += GRAVITY  # 重力作用
        self.x += self.vx
        self.y += self.vy

    def draw(self, frame):
        if self.active:
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius, self.color, -1)
            # 畫一點高光質感
            cv2.circle(frame, (int(self.x - 8), int(self.y - 8)), 6, (255, 255, 255), -1)

class Particle:
    """爆炸碎片"""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-5, 5)
        self.vy = random.uniform(-5, 5)
        self.life = 1.0  # 生命值 1.0 -> 0.0
        self.color = color
        self.size = random.randint(3, 8)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += GRAVITY * 0.5 # 碎片也要受一點重力
        self.life -= 0.05  # 衰減

    def draw(self, frame):
        if self.life > 0:
            # 隨生命值變小
            radius = int(self.size * self.life)
            if radius > 0:
                cv2.circle(frame, (int(self.x), int(self.y)), radius, self.color, -1)

# === 全域變數 ===
recognition_result = None
fruits = []
particles = []
score = 0
prev_finger_x, prev_finger_y = 0, 0
frame_count = 0

# 手部連線定義
INDEX_FINGER_TIP = 8
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

def save_result(result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global recognition_result
    recognition_result = result

def get_index_finger_position(hand_landmarks, w, h):
    index_tip = hand_landmarks[INDEX_FINGER_TIP]
    return int(index_tip.x * w), int(index_tip.y * h)

def draw_hand(frame, hand_landmarks):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    
    # 畫連線
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (200, 200, 200), 2)
    
    # 畫關鍵點
    for p in points:
        cv2.circle(frame, p, 3, (0, 255, 0), -1)

def main():
    global recognition_result, fruits, particles, score, frame_count
    global prev_finger_x, prev_finger_y

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

    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("=" * 50)
        print("食指水果忍者 (Finger Fruit Ninja)")
        print("=" * 50)
        print("用食指切開水果得分!")
        print("按 ESC 離開")
        print("=" * 50)

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            # 1. 鏡像翻轉
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # 2. MediaPipe 辨識
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp = time.time_ns() // 1_000_000
            recognizer.recognize_async(mp_image, timestamp)

            # 3. 遊戲邏輯：生成水果
            frame_count += 1
            if frame_count % FRUIT_SPAWN_RATE == 0:
                fruits.append(Fruit(w, h))

            # 4. 手部互動
            finger_detected = False
            curr_finger_x, curr_finger_y = 0, 0

            if recognition_result and recognition_result.hand_landmarks:
                hand_landmarks = recognition_result.hand_landmarks[0]
                draw_hand(frame, hand_landmarks)
                
                curr_finger_x, curr_finger_y = get_index_finger_position(hand_landmarks, w, h)
                finger_detected = True

                # 畫出食指刀刃軌跡
                if prev_finger_x != 0:
                    cv2.line(frame, (prev_finger_x, prev_finger_y), (curr_finger_x, curr_finger_y), (255, 255, 255), 5)
                
                cv2.circle(frame, (curr_finger_x, curr_finger_y), 10, (0, 0, 255), -1)

            # 5. 更新水果與碰撞檢測
            for fruit in fruits[:]: # 使用 slice copy 進行迭代以便移除
                fruit.update()
                
                # 檢查是否掉出畫面
                if fruit.y > h + 50:
                    fruits.remove(fruit)
                    continue

                # 碰撞檢測 (只在有偵測到手指時)
                if finger_detected and fruit.active:
                    dist = math.hypot(curr_finger_x - fruit.x, curr_finger_y - fruit.y)
                    if dist < fruit.radius + 15: # 15是手指半徑緩衝
                        # 引爆！
                        fruit.active = False
                        score += 1
                        fruits.remove(fruit)
                        
                        # 產生爆炸粒子
                        for _ in range(10):
                            particles.append(Particle(fruit.x, fruit.y, fruit.color))

            # 6. 更新並繪製粒子
            for p in particles[:]:
                p.update()
                p.draw(frame)
                if p.life <= 0:
                    particles.remove(p)

            # 7. 繪製水果
            for fruit in fruits:
                fruit.draw(frame)

            # 8. 介面顯示
            # 背景黑框讓分數清楚一點
            cv2.rectangle(frame, (10, 10), (200, 60), (0, 0, 0), -1)
            cv2.putText(frame, f"Score: {score}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            
            if not finger_detected:
                cv2.putText(frame, "Show Finger!", (w//2 - 100, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 更新手指前一幀位置
            if finger_detected:
                prev_finger_x, prev_finger_y = curr_finger_x, curr_finger_y
            else:
                prev_finger_x, prev_finger_y = 0, 0 # 重置

            cv2.imshow('Fruit Ninja Finger', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
