# MediaPipe 手勢應用與 AI 助理 (Mediapipe_GenAI)

本專案結合了 Google MediaPipe 的電腦視覺技術、PyTorch 深度學習以及 Gemini 2.0 的生成式 AI 能力，展示了多模態互動的可能性。

## 🌟 主要功能

### 1. ✍️ AI 手勢手寫數字辨識 (`apps/gesture_digits/gesture_digit_gui.py`)
透過食指指尖在虛擬空間中書寫數字，並由 AI 即時辨識。
*   **核心技術**: MediaPipe Hands, PyTorch (MNIST CNN), Tkinter
*   **特色功能**:
    *   **手控繪圖**: 使用「開始/暫停」按鈕控制，食指指尖即為筆刷。
    *   **AI 視野 (AI View)**: 即時顯示縮放、置中處理後的 28x28 圖像，展示 AI 真正看到的畫面。
    *   **標準化預處理**: 自動裁切與縮放，確保手寫數字符合 MNIST 標準，大幅提升辨識率。

### 2. 🤖 Google Calendar AI Agent (`apps/agent/calendar_agent.py`)
一個基於自然語言的智慧日曆助理，讓您能用「說」的來管理行程。
*   **核心技術**: LangChain, Gemini 2.0 Flash, Google Calendar API
*   **功能**:
    *   查詢行程 ("明天下午有什麼會議？")
    *   新增活動 ("後天早上10點提醒我看牙醫")
    *   具備對話記憶，能理解上下文。

### 3. 🖼️ 手寫數字辨識工具 (`tools/mnist_train/mnist_gui.py`)
針對靜態圖片進行辨識的桌面應用程式。
*   **功能**: 支援匯入 PNG/JPG 圖片，自動處理白底黑字/黑底白字轉換並進行辨識。

### 4. 👆 食指踢球遊戲 (`apps/games/finger_ball.py`)
透過攝影機捕捉食指動作，與虛擬足球進行物理互動的趣味遊戲。
*   **核心技術**: MediaPipe Hands, OpenCV, 物理碰撞模擬
*   **玩法**: 移動食指去「踢」畫面中的橘色球，球會根據撞擊力道反彈。

### 5. 🍎 水果忍者手勢版 (`apps/games/fruit_ninja.py`)
使用手勢模擬刀刃切水果的經典遊戲。

![Fruit Ninja Demo](assets/demo/fruit_ninja_demo.gif)

## 🚀 快速開始

### 環境準備
1.  建立並啟動虛擬環境 (建議)。
2.  安裝基礎套件:
    ```bash
    pip install -r requirements.txt
    ```
3.  **安裝 PyTorch (建議 CUDA 版本以加速辨識)**:
    ```bash
    # 針對 CUDA 12.4
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    ```
4.  **設定 AI 助理 (若要執行 calendar_agent)**:
    *   在 `config/` 目錄下建立 `.env` 檔，填入 `GEMINI_API_KEY=您的金鑰`。
    *   將 Google Cloud Console 下載的 OAuth 憑證 (`credentials.json`) 放入 `config/` 目錄。

### 執行程式

**啟動 AI 手勢手寫辨識系統 (推薦體驗):**
```bash
python apps/gesture_digits/gesture_digit_gui.py
```

**啟動靜態圖數字辨識工具:**
```bash
python tools/mnist_train/mnist_gui.py
```

**啟動 AI 日曆助理:**
```bash
python apps/agent/calendar_agent.py
```

**啟動食指踢球遊戲:**
```bash
python apps/games/finger_ball.py
```

## 📂 目錄結構
*   `apps/`: 所有應用程式入口 (Agent, Games, Gesture Digits)。
*   `models/`: 存放所有模型檔案 (`.task`, `.pth`)。
*   `tools/`: 訓練腳本與工具。
*   `config/`: 存放設定檔與憑證 (已忽略，不需上傳)。
*   `assets/`: 靜態資源 (Demo 圖片等)。