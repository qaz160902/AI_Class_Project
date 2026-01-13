from google import genai
import os
from dotenv import load_dotenv

# 載入 .env 檔案中的環境變數
load_dotenv()

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


