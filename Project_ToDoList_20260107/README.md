# 全端待辦事項與日曆系統 (2026-01-07)

這是一個前後端分離的個人生產力工具，整合了待辦事項清單 (To-Do List) 與日曆檢視功能。

## 🛠️ 技術架構

*   **前端 (Frontend)**:
    *   Vue 3 (Composition API)
    *   Vite (建置工具)
    *   FullCalendar (日曆套件)
*   **後端 (Backend)**:
    *   Python Flask (RESTful API)
    *   SQLite (資料庫)
    *   SQLAlchemy (ORM)

## 📂 目錄結構

*   `todo-project/backend/`: 後端 API 伺服器程式碼。
*   `todo-project/frontend/`: 前端 Vue 應用程式碼。

## 🚀 快速開始

本專案需要分別啟動後端與前端伺服器。

### 1. 啟動後端 (Backend)
```bash
cd todo-project/backend
# 安裝依賴
pip install -r requirements.txt
# 啟動 Flask 伺服器 (預設 port: 5000)
python app.py
```

### 2. 啟動前端 (Frontend)
開啟一個新的終端機視窗：
```bash
cd todo-project/frontend
# 安裝依賴
npm install
# 啟動開發伺服器
npm run dev
```

### 3. 使用
打開瀏覽器訪問前端顯示的網址 (通常是 `http://localhost:5173`) 即可開始使用。
