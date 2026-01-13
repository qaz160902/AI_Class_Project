# System Instructions
請一律使用繁體中文 (Traditional Chinese) 回覆。
目前作業系統為 Windows 11。

# AI Class Project - Gemini Context

本目錄 `D:\AWORKSPACE\Github\AI_Class_Project` 是一個 AI 課程的實作集合，包含三個主要子專案，涵蓋電腦視覺、影像辨識模型應用、生成式 AI 助理以及全端網頁開發。

## 📂 專案結構概覽

### 1. Project_Mediapipe_20260113 (MediaPipe & GenAI Agent)
基於 Google MediaPipe 的手勢應用，以及整合 Gemini 2.0 的智慧助理。

*   **核心技術**: Python, MediaPipe, OpenCV, **LangChain, Gemini 2.0 Flash**
*   **主要檔案**:
    *   `calendar_agent.py`: **[NEW]** 基於 LangChain 與 Gemini 的 Google 日曆 AI 助理。支援自然語言查詢、新增行程。
    *   `finger_ball.py`: 食指踢球遊戲，包含物理碰撞模擬。
    *   `fruit_ninja.py`: 水果忍者手勢遊戲。
    *   `test_gemini_new.py`: Gemini API 連線測試腳本。
*   **模型**:
    *   手勢: `model/gesture_recognizer.task`
    *   LLM: `gemini-2.0-flash-exp` (用於 Agent)
*   **環境設定 (.env)**:
    *   請在 `Project_Mediapipe_20260113/Gemini/` 下建立 `.env` 檔案，並設定 `GEMINI_API_KEY`。
    *   Google Calendar 憑證 `credentials.json` 需放置於同目錄 (需設定為 Desktop App 類型)。
*   **如何執行**:
    *   **AI 日曆助理**:
        ```bash
        cd Project_Mediapipe_20260113
        python calendar_agent.py
        ```
    *   **手勢遊戲**:
        ```bash
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
    *   Teachable Machine: 需要 TensorFlow。
    *   Mediapipe & Agent: 需要 MediaPipe, LangChain, Google GenAI。
2.  **編碼風格**:
    *   Python: 遵循 PEP 8。
    *   Vue/JS: Component 命名使用 PascalCase。
3.  **安全性**:
    *   **絕對不要**將 `.env`, `credentials.json`, `token.json` 上傳至 GitHub。
    *   已設定 `.gitignore` 自動排除這些敏感檔案。

## 📝 常用指令備忘

*   **列出當前依賴**: `pip list` or `npm list`
*   **啟動 Vue 開發伺服器**: `npm run dev`
*   **Git 操作**:
    *   `git status`: 檢查檔案狀態
    *   `git add .`: 加入所有更動
    *   `git commit -m "訊息"`: 提交更動