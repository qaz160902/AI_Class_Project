# System Instructions
請一律使用繁體中文 (Traditional Chinese) 回覆。
目前作業系統為 Windows 11。

# AI Class Project - Gemini Context

本目錄 `D:\AWORKSPACE\Github\AI_Class_Project` 是一個 AI 課程的實作集合，包含三個主要子專案，涵蓋電腦視覺、影像辨識模型應用、生成式 AI 助理以及全端網頁開發。

## 📂 專案結構概覽

### 1. Mediapipe_GenAI (MediaPipe & GenAI Agent)
基於 Google MediaPipe 的手勢應用，以及整合 Gemini 2.0 的智慧助理。

*   **核心技術**: Python, MediaPipe, OpenCV, **LangChain, Gemini 2.0 Flash**, **PyTorch (CUDA enabled), Tkinter**
*   **主要檔案 (apps/)**:
    *   `apps/agent/calendar_agent.py`: 基於 LangChain 與 Gemini 的 Google 日曆 AI 助理。
    *   `apps/gesture_digits/gesture_digit_gui.py`: AI 手勢手寫數字辨識系統。
    *   `apps/games/finger_ball.py`: 食指踢球遊戲。
    *   `apps/games/fruit_ninja.py`: 水果忍者手勢遊戲。
*   **工具 (tools/)**:
    *   `tools/mnist_train/mnist_gui.py`: 靜態圖片手寫數字辨識工具。
*   **模型 (models/)**:
    *   手勢: `models/gesture_recognizer.task`
    *   影像辨識: `models/mnist_cnn.pth` (PyTorch CNN 手寫數字模型)
*   **環境設定 (config/)**:
    *   請在 `Mediapipe_GenAI/config/` 下建立 `.env` 檔案，並設定 `GEMINI_API_KEY`。
    *   Google Calendar 憑證 `credentials.json` 需放置於同目錄。
*   **如何執行**:
    *   **AI 日曆助理**:
        ```bash
        cd Mediapipe_GenAI
        python apps/agent/calendar_agent.py
        ```
    *   **手勢手寫辨識**:
        ```bash
        cd Mediapipe_GenAI
        python apps/gesture_digits/gesture_digit_gui.py
        ```
    *   **手勢遊戲**:
        ```bash
        cd Mediapipe_GenAI
        python apps/games/finger_ball.py
        ```

### 2. TeachableMachine_Vision (Teachable Machine 影像辨識)
整合 Google Teachable Machine 匯出模型與 CustomTkinter GUI 的即時影像辨識系統。

*   **核心技術**: Python, TensorFlow/Keras, OpenCV, CustomTkinter
*   **主要檔案**:
    *   `gemini_gui.py`: 現代化深色模式 GUI 主程式 (建議執行此檔)。
    *   `gui_app.py`: 基礎版 GUI。
    *   `tm.py`, `opencv_tm.py`: 模型推論邏輯。
*   **模型**: `model/keras_model.h5`, `model/labels.txt`
*   **如何執行**:
    ```bash
    cd TeachableMachine_Vision
    python gemini_gui.py
    ```

### 3. FullStack_ToDoList (全端待辦事項與日曆)
前後端分離的個人生產力工具。

*   **核心技術**:
    *   **Frontend**: Vue 3, Vite, FullCalendar/V-Calendar
    *   **Backend**: Python Flask, SQLite, SQLAlchemy
*   **目錄結構**:
    *   `backend/`: Flask API Server (`app.py`, `models.py`)
    *   `frontend/`: Vue 3 Client (`src/components/`, `src/views/`)
*   **如何執行**:
    *   **後端**:
        ```bash
        cd FullStack_ToDoList/todo-project/backend
        python app.py
        ```
    *   **前端**:
        ```bash
        cd FullStack_ToDoList/todo-project/frontend
        npm run dev
        ```

## 🛠️ 開發慣例與注意事項

1.  **環境管理**: 建議為每個 Python 子專案建立獨立的虛擬環境 (Virtual Environment)，避免套件衝突。
    *   Teachable Machine: 需要 TensorFlow。
    *   Mediapipe & Agent: 需要 MediaPipe, LangChain, Google GenAI, PyTorch (建議安裝 CUDA 版本)。
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