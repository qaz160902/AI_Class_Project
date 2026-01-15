from google import genai
import os
from dotenv import load_dotenv

# 取得目前腳本所在的目錄
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 指定 .env 檔案的路徑 (在 Gemini 子目錄下)
ENV_PATH = os.path.join(SCRIPT_DIR, "Gemini", ".env")

# 載入指定路徑的 .env 檔案
load_dotenv(ENV_PATH)

# 從環境變數讀取 API Key
API_KEY = os.getenv("GEMINI_API_KEY")

def main():
    if not API_KEY or API_KEY == "YOUR_REAL_API_KEY_HERE":
        print("錯誤: 請先在 .env 檔案中設定正確的 GEMINI_API_KEY")
        return

    # 初始化 Client
    client = genai.Client(api_key=API_KEY)

    print("正在發送請求給 Gemini...")
    
    try:
        # 使用最新的 gemini-2.0-flash 模型進行測試
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents='哈囉 Gemini！請用繁體中文簡單自我介紹一下。'
        )
        
        print("\n--- Gemini 的回應 ---")
        print(response.text)
        print("---------------------")
        
    except Exception as e:
        print(f"發生錯誤: {e}")
        print("\n提示: 請確認您的 API Key 是否正確。")

if __name__ == "__main__":
    main()


