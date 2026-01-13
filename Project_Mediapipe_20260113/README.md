# MediaPipe 手勢應用與 AI 助理 (2026-01-13)

本專案結合了 Google MediaPipe 的電腦視覺技術與 Gemini 2.0 的生成式 AI 能力，展示了多模態互動的可能性。

## 🌟 主要功能

### 1. 🤖 Google Calendar AI Agent (`calendar_agent.py`)
一個基於自然語言的智慧日曆助理，讓您能用「說」的來管理行程。
*   **核心技術**: LangChain, Gemini 2.0 Flash, Google Calendar API
*   **功能**:
    *   查詢行程 ("明天下午有什麼會議？")
    *   新增活動 ("後天早上10點提醒我看牙醫")
    *   具備對話記憶，能理解上下文。

### 2. 👆 食指踢球遊戲 (`finger_ball.py`)
透過攝影機捕捉食指動作，與虛擬足球進行物理互動的趣味遊戲。
*   **核心技術**: MediaPipe Hands, OpenCV, 物理碰撞模擬
*   **玩法**: 移動食指去「踢」畫面中的橘色球，球會根據撞擊力道反彈。

### 3. 🍎 水果忍者手勢版 (`fruit_ninja.py`)
使用手勢模擬刀刃切水果的經典遊戲。

![Fruit Ninja Demo](demo/fruit_ninja_demo.gif)

## 🚀 快速開始

### 環境準備
1.  建立並啟動虛擬環境 (建議)。
2.  安裝依賴套件:
    ```bash
    pip install -r requirements.txt
    # 若無 requirements.txt，請手動安裝:
    # pip install opencv-python mediapipe langchain-google-genai langchain-google-community
    ```
3.  **設定 AI 助理 (若要執行 calendar_agent)**:
    *   在 `Gemini/` 目錄下建立 `.env` 檔，填入 `GEMINI_API_KEY=您的金鑰`。
    *   將 Google Cloud Console 下載的 OAuth 憑證 (`credentials.json`) 放入 `Gemini/` 目錄。

### 執行程式

**啟動 AI 日曆助理:**
```bash
python calendar_agent.py
```

**啟動食指踢球遊戲:**
```bash
python finger_ball.py
```

## 📂 目錄結構
*   `Gemini/`: 存放 API 金鑰與憑證 (已忽略，不需上傳)。
*   `model/`: 存放 MediaPipe 手勢辨識模型。
*   `*.py`: 各項功能的主程式。
