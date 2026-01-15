import os
import json
from dotenv import load_dotenv
from langchain_google_community import CalendarToolkit
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

# === 設定路徑 ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 往上兩層找到 config 資料夾 (apps/agent -> apps -> Mediapipe_GenAI -> config)
GEMINI_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'config'))
ENV_PATH = os.path.join(GEMINI_DIR, ".env")
CREDENTIALS_PATH = os.path.join(GEMINI_DIR, "credentials.json")
TOKEN_PATH = os.path.join(GEMINI_DIR, "token.json")

# === 1. 環境設定 ===
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
    if "GEMINI_API_KEY" in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
else:
    print(f"警告: 找不到 .env 檔案: {ENV_PATH}")

# === 2. 初始化日曆工具 ===
print("初始化 Google Calendar 工具...")
from langchain_google_community.calendar.utils import build_calendar_service
from langchain_google_community._utils import get_google_credentials

try:
    creds = get_google_credentials(
        scopes=["https://www.googleapis.com/auth/calendar"],
        token_file=TOKEN_PATH,
        client_secrets_file=CREDENTIALS_PATH
    )
    calendar_service = build_calendar_service(credentials=creds)
    toolkit = CalendarToolkit(api_resource=calendar_service)
    tools = toolkit.get_tools()
    tools_dict = {tool.name: tool for tool in tools}
    print("Google Calendar 工具初始化成功。")
except Exception as e:
    print(f"初始化工具時發生錯誤: {e}")
    exit(1)

# === 3. 設定 Gemini 模型並綁定工具 ===
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm_with_tools = llm.bind_tools(tools)

# === 4. 執行 Agent 邏輯 (支援記憶) ===
def run_calendar_agent(user_query: str, chat_history: list):
    print(f"\n[處理中]: {user_query}")
    
    # 將新的使用者訊息加入歷史紀錄
    chat_history.append(HumanMessage(content=user_query))
    
    # 建立本次執行的暫時訊息列表 (避免汙染主歷史紀錄，主要用於工具呼叫的來回)
    # 注意: 這裡我們直接使用 chat_history，因為我們希望工具呼叫的過程也被模型記住
    # 但為了避免 token 爆炸，實際應用中通常會做修剪。這裡為了簡單演示記憶，直接使用。
    
    # 限制最多 5 次工具互動防止死迴圈
    for i in range(5):
        response = llm_with_tools.invoke(chat_history)
        chat_history.append(response) # 將模型的初步回應 (可能包含工具呼叫) 加入歷史
        
        # 如果模型沒有要呼叫工具，則回傳內容
        if not response.tool_calls:
            return response.content
        
        # 模型想要呼叫工具
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"  -> 呼叫工具: {tool_name}")
            
            if tool_name in tools_dict:
                try:
                    observation = tools_dict[tool_name].invoke(tool_args)
                    chat_history.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
                except Exception as e:
                    chat_history.append(ToolMessage(content=f"工具執行錯誤: {e}", tool_call_id=tool_call["id"]))
            else:
                chat_history.append(ToolMessage(content=f"錯誤: 找不到工具 {tool_name}", tool_call_id=tool_call["id"]))

    return "Agent 達到最大互動次數，無法完成請求。"

# === 5. 互動迴圈 ===
if __name__ == "__main__":
    print("\n=== Google Calendar AI Agent (具備對話記憶) ===")
    print("輸入 'exit' 或 'quit' 離開")
    print("輸入 'clear' 清除記憶")
    print("================================\n")

    # 初始化對話歷史
    chat_history = []

    while True:
        user_input = input("\n您的指令: ")
        
        if user_input.lower() in ["exit", "quit"]:
            break
        elif user_input.lower() == "clear":
            chat_history = []
            print("記憶已清除。")
            continue
        
        try:
            result = run_calendar_agent(user_input, chat_history)
            
            # 優化輸出顯示: 如果是 list (通常包含 metadata)，只顯示 text 部分
            if isinstance(result, list) and len(result) > 0 and 'text' in result[0]:
                 print(f"\nAI 回覆: {result[0]['text']}")
            elif isinstance(result, str):
                 print(f"\nAI 回覆: {result}")
            else:
                 print(f"\nAI 回覆: {result}")

        except Exception as e:
            print(f"發生錯誤: {e}")