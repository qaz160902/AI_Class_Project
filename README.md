# AI_Class_Project - AI 新秀計劃課程實作

歡迎來到我的 AI 實作專案集！這裡記錄了我在「AI 新秀計劃」課程中的學習成果與程式碼實作。

本專案使用 **Gemini CLI** 協助開發，透過自然語言指令加速程式碼編寫、除錯與文檔生成。

## 🚀 實作專案列表

以下專案按開發時間倒序排列：

### 1. [MediaPipe 手勢應用與 AI 助理](Project_Mediapipe_20260113/README.md) (2026-01-13)
結合電腦視覺與生成式 AI 的進階應用。
*   **🤖 AI 日曆助理**: 使用 Gemini 2.0 Flash 與 Google Calendar API，打造能「聽懂」人話的行程管理助手。
*   **👆 食指踢球遊戲**: 透過 MediaPipe 捕捉手勢，與畫面中的虛擬足球互動。
*   **👉 [點此查看詳細說明與執行方式](Project_Mediapipe_20260113/README.md)**

### 2. [Gemini Vision Pro - 影像辨識系統](Project_Teachable_20260108/README.md) (2026-01-08)
整合 Google Teachable Machine 模型與現代化 GUI 的即時辨識系統。
*   **👁️ 即時辨識**: 支援攝影機即時分類與單張快照模式。
*   **🎨 現代化介面**: 使用 CustomTkinter 打造深色模式 UI。
*   **👉 [點此查看詳細說明與執行方式](Project_Teachable_20260108/README.md)**

### 3. [全端待辦事項與日曆系統](Project_ToDoList_20260107/README.md) (2026-01-07)
前後端分離的個人生產力工具。
*   **Frontend**: Vue 3 + Vite + FullCalendar
*   **Backend**: Python Flask + SQLite
*   **👉 [點此查看詳細說明與執行方式](Project_ToDoList_20260107/README.md)**

---

## 🛠️ 開發環境與工具

*   **Gemini CLI**: 在終端機中直接與 Gemini AI 協作，輔助程式碼生成與 Git 操作。
*   **LangChain**: 用於建構 AI Agent 的框架。
*   **MediaPipe / TensorFlow**: 電腦視覺與機器學習模型支援。
*   **Vue 3 / Flask**: 全端網頁開發框架。

---

## 🤖 Gemini CLI 斜線指令 (Slash Commands)

本專案在開發過程中使用 Gemini CLI 進行輔助，以下是完整的可用指令清單：

| 指令 (Command) | 說明 (Description) |
| :--- | :--- |
| `/about` | Show version info (顯示版本資訊) |
| `/auth` | Manage authentication (管理登入驗證) |
| `/bug` | Submit a bug report (回報錯誤) |
| `/chat` | Manage conversation history (管理對話歷史) |
| `/clear` | Clear the screen and conversation history (清除畫面與對話) |
| `/compress` | Compresses the context by replacing it with a summary (壓縮上下文) |
| `/copy` | Copy the last result or code snippet to clipboard (複製結果) |
| `/docs` | Open full Gemini CLI documentation in your browser (開啟線上文件) |
| `/directory` | Manage workspace directories (管理工作目錄) |
| `/editor` | Set external editor preference (設定外部編輯器) |
| `/extensions` | Manage extensions (管理擴充功能) |
| `/help` | For help on gemini-cli (顯示幫助) |
| `/ide` | Manage IDE integration (管理 IDE 整合) |
| `/init` | Analyzes the project and creates a tailored GEMINI.md file (初始化專案分析) |
| `/mcp` | Manage configured Model Context Protocol (MCP) servers (管理 MCP 伺服器) |
| `/memory` | Commands for interacting with memory (記憶功能操作) |
| `/model` | Opens a dialog to configure the model (設定 AI 模型) |
| `/privacy` | Display the privacy notice (隱私權聲明) |
| `/policies` | Manage policies (管理策略) |
| `/quit` | Exit the cli (離開程式) |
| `/resume` | Browse and resume auto-saved conversations (恢復先前的對話) |
| `/stats` | Check session stats (查看統計資訊). Usage: `/stats [session\|model\|tools]` |

---
*AI 新秀計劃課程導師：蘇弘舉*