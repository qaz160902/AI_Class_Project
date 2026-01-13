# System Instructions
請一律使用繁體中文 (Traditional Chinese) 回覆。
目前作業系統為 Windows 11。

# AI Class Project - Gemini Context

本目錄 `D:\AWORKSPACE\Github\AI_Class_Project` 是一個 AI 課程的實作集合，包含三個主要子專案，涵蓋電腦視覺、影像辨識模型應用以及全端網頁開發。

## 📂 專案結構概覽

### 1. Project_Mediapipe_20260113 (MediaPipe 手勢應用)
基於 Google MediaPipe 與 OpenCV 的手勢辨識與互動應用。

*   **核心技術**: Python, MediaPipe, OpenCV
*   **主要檔案**:
    *   `finger_ball.py`: 食指踢球遊戲，包含物理碰撞模擬。
    *   `fruit_ninja.py`: 水果忍者手勢遊戲 (推測)。
    *   `gesture_alt_tab.py`: 手勢控制視窗切換 (推測)。
    *   `recognize_hand.py`, `TestHand.py`: 手部辨識測試腳本。
*   **模型**: `model/gesture_recognizer.task`
*   **如何執行**:
    ```bash
    cd Project_Mediapipe_20260113
    python finger_ball.py
    ```

### 2. Project_Teachable_20260108 (Teachable Machine 影像辨識)
整合 Google Teachable Machine 匯出模型與 CustomTkinter GUI 的即時影像辨識系統。

*   **核心技術**: Python, TensorFlow/Keras, OpenCV, CustomTkinter
*   **主要檔案**:
    *   `gemini_gui.py`: 現代化深色模式 GUI 主程式 (建議執行此檔)。
    *   `gui_app.py`: 基礎版 GUI。
    *   `tm.py`, `opencv_tm.py`: 模型推論邏輯。
*   **模型**: `model/keras_model.h5`, `model/labels.txt`
*   **如何執行**:
    ```bash
    cd Project_Teachable_20260108
    # 安裝依賴 (若尚未安裝)
    # pip install opencv-python numpy pillow customtkinter tensorflow tf-keras
    python gemini_gui.py
    ```

### 3. Project_ToDoList_20260107 (全端待辦事項與日曆)
前後端分離的個人生產力工具。

*   **核心技術**:
    *   **Frontend**: Vue 3, Vite, FullCalendar/V-Calendar (推測)
    *   **Backend**: Python Flask, SQLite, SQLAlchemy
*   **目錄結構**:
    *   `backend/`: Flask API Server (`app.py`, `models.py`)
    *   `frontend/`: Vue 3 Client (`src/components/`, `src/views/`)
*   **如何執行**:
    *   **後端**:
        ```bash
        cd Project_ToDoList_20260107/todo-project/backend
        # pip install -r requirements.txt
        python app.py
        ```
    *   **前端**:
        ```bash
        cd Project_ToDoList_20260107/todo-project/frontend
        # npm install
        npm run dev
        ```

## 🛠️ 開發慣例與注意事項

1.  **環境管理**: 建議為每個 Python 子專案建立獨立的虛擬環境 (Virtual Environment)，避免套件衝突。
    *   例如 Teachable Machine 專案需要 TensorFlow，而 Mediapipe 專案需要 MediaPipe。
2.  **編碼風格**:
    *   Python: 遵循 PEP 8 (Snake case `function_name`)。
    *   Vue/JS: Component 命名使用 PascalCase (`CalendarView.vue`)。
3.  **路徑處理**: 專案中多處使用 `os.path` 處理模型路徑，確保跨平台相容性。

## 📝 常用指令備忘

*   **列出當前依賴**: `pip list` or `npm list`
*   **啟動 Vue 開發伺服器**: `npm run dev`
*   **Git 操作**:
    *   `git status`: 檢查檔案狀態
    *   `git add .`: 加入所有更動
    *   `git commit -m "訊息"`: 提交更動
