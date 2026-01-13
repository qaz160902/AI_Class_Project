import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# 模型路徑
MODEL_PATH = 'model/gesture_recognizer.task'

# 手部關鍵點連線定義 (參考 MediaPipe Hands)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),       # 食指
    (5, 9), (9, 10), (10, 11), (11, 12),  # 中指
    (9, 13), (13, 14), (14, 15), (15, 16),# 無名指
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # 小指
]

def draw_landmarks_on_image(rgb_image, hand_landmarks):
    """使用 OpenCV 繪製手部關鍵點與連線"""
    image_height, image_width, _ = rgb_image.shape
    annotated_image = rgb_image.copy()

    # 繪製連線
    for connection in HAND_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        
        start_point = hand_landmarks[start_idx]
        end_point = hand_landmarks[end_idx]
        
        start_x = int(start_point.x * image_width)
        start_y = int(start_point.y * image_height)
        end_x = int(end_point.x * image_width)
        end_y = int(end_point.y * image_height)
        
        cv2.line(annotated_image, (start_x, start_y), (end_x, end_y), (224, 224, 224), 2)

    # 繪製關鍵點
    for landmark in hand_landmarks:
        x = int(landmark.x * image_width)
        y = int(landmark.y * image_height)
        cv2.circle(annotated_image, (x, y), 4, (0, 255, 0), -1)

    return annotated_image

def main():
    # 1. 設定 MediaPipe GestureRecognizer 選項
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,  # 最多偵測兩隻手
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 2. 建立 GestureRecognizer 實例
    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        # 開啟 Webcam (索引通常為 0)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("無法開啟 Webcam")
            return

        print("按下 'q' 或 'ESC' 離開程式")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("無法讀取影像")
                break

            # 3. 準備影像資料
            # MediaPipe 需要 RGB 格式，OpenCV 預設為 BGR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # 計算時間戳記 (微秒)
            # 使用 frame_count 或系統時間皆可，這裡簡單用系統時間
            timestamp_ms = int(time.time() * 1000)

            # 4. 進行辨識 (Video Mode)
            recognition_result = recognizer.recognize_for_video(mp_image, timestamp_ms)

            # 5. 繪製結果
            # 為了方便繪圖，我們將 frame (BGR) 複製一份來畫圖
            display_frame = frame.copy()
            
            if recognition_result.gestures:
                for i, gestures in enumerate(recognition_result.gestures):
                    # 取得第一個最可能的手勢
                    top_gesture = gestures[0]
                    hand_landmarks = recognition_result.hand_landmarks[i]
                    
                    # 顯示手勢名稱與信心分數
                    title = f"{top_gesture.category_name} ({top_gesture.score:.2f})"
                    
                    # 簡單計算手腕位置來放置文字
                    # hand_landmarks[0] 通常是手腕
                    wrist = hand_landmarks[0]
                    h, w, _ = display_frame.shape
                    x, y = int(wrist.x * w), int(wrist.y * h)

                    # 手動繪製骨架
                    display_frame = draw_landmarks_on_image(display_frame, hand_landmarks)

                    cv2.putText(display_frame, title, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # 顯示影像
            cv2.imshow('MediaPipe Gesture Recognition', display_frame)

            # 按鍵偵測
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
