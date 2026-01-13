import os
from dotenv import load_dotenv
from langchain_google_community import GoogleCalendarToolkit
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# === 設定路徑 ===
# 取得目前腳本所在的目錄
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 設定 Gemini 子目錄路徑
GEMINI_DIR = os.path.join(SCRIPT_DIR, "Gemini")
# 設定各個檔案的絕對路徑
ENV_PATH = os.path.join(GEMINI_DIR, ".env")
CREDENTIALS_PATH = os.path.join(GEMINI_DIR, "credentials.json")
TOKEN_PATH = os.path.join(GEMINI_DIR, "token.json")

# === 1. 環境設定 ===
# 載入 .env 檔案
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
    # 將 GEMINI_API_KEY 轉為 GOOGLE_API_KEY (LangChain 預設使用 GOOGLE_API_KEY)
    if "GEMINI_API_KEY" in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
else:
    print(f"警告: 找不到 .env 檔案: {ENV_PATH}")

# === 2. 初始化日曆工具 ===
print("初始化 Google Calendar 工具...")
print(f"憑證路徑: {CREDENTIALS_PATH}")

# 確保 credentials.json 存在
if not os.path.exists(CREDENTIALS_PATH):
    print("錯誤: 找不到 credentials.json，請確認檔案是否在 Gemini 資料夾中。")
    exit(1)

# 初始化 Toolkit
# 注意: 這會嘗試讀取 token_file，如果過期或不存在，會使用 credentials_file 進行 OAuth 登入流程
toolkit = GoogleCalendarToolkit(
    credentials_file=CREDENTIALS_PATH,
    token_file=TOKEN_PATH,
    read_only=False  # 設為 False 才能新增行程
)
tools = toolkit.get_tools()

# === 3. 設定 Gemini 模型 ===
# 使用 gemini-2.0-flash (依您的需求)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# === 4. 建立 Agent ===
print("建立 AI Agent...")
# 使用 LangChain Hub 的標準React Prompt
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# === 5. 互動迴圈 ===
if __name__ == "__main__":
    print("\n=== Google Calendar AI Agent ===")
    print("輸入 'exit' 或 'quit' 離開")
    print("首次執行時，請留意瀏覽器跳出的 Google 授權視窗。")
    print("================================\n")

    # 首次測試
    initial_query = "幫我查看我明天下午有什麼會議？"
    print(f"正在執行測試指令: {initial_query}")
    try:
        response = agent_executor.invoke({"input": initial_query})
        print(f"\nAI 回覆: {response['output']}")
    except Exception as e:
        print(f"發生錯誤: {e}")

    while True:
        user_input = input("\n請輸入指令 (例如: 新增一個後天早上的會議): ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        try:
            response = agent_executor.invoke({"input": user_input})
            print(f"\nAI 回覆: {response['output']}")
        except Exception as e:
            print(f"發生錯誤: {e}")