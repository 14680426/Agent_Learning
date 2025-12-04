from qwen_agent.agents import Assistant
import sys
import os
from dotenv import load_dotenv
load_dotenv() 
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..")
sys.path.insert(0, src_dir)

# LLM 配置
llm_cfg = {
    "model": 'DeepSeek-V3.1-Terminus',
    "model_server": os.environ["BINGXING_BASE_URL"],
    "api_key": os.environ["BINGXING_API_KEY"],
}

# 系统消息
system = "你是会查询时间和转换时间的助手"

# 工具列表
tools = [
    {
        "mcpServers": {
            "time": {
                "type": "sse",
                "url": "https://open.bigmodel.cn/api/mcp-broker/proxy/time/sse",
                "headers": {
                    "Authorization": f"Bearer {os.environ['ZHIPUAI_API_KEY']}"
                }
            },
            "amap-maps": {
                "type": "sse",
                "url": "https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/sse",
                "headers": {
                    "Authorization": f"Bearer {os.environ['DASHSCOPE_API_KEY']}"
                }
            },
        }
    }
]

# 创建助手实例
bot = Assistant(
    llm=llm_cfg,
    name="助手",
    description="查询时间，转换时间，查询路线",
    system_message=system,
    function_list=tools,
)

messages = []

while True:
    query = input("\nuser question: ")
    if not query.strip():
        print("user question cannot be empty！")
        continue
    messages.append({"role": "user", "content": query})
    bot_response = ""
    is_tool_call = False
    tool_call_info = {}
    for response_chunk in bot.run(messages):
        new_response = response_chunk[-1]
        if "function_call" in new_response:
            is_tool_call = True
            tool_call_info = new_response["function_call"]
        elif "function_call" not in new_response and is_tool_call:
            is_tool_call = False
            print("\n" + "=" * 20)
            print("工具调用信息：", tool_call_info)
            print("工具调用结果：", new_response)
            print("=" * 20)
        elif new_response.get("role") == "assistant" and "content" in new_response:
            incremental_content = new_response["content"][len(bot_response):]
            print(incremental_content, end="", flush=True)
            bot_response += incremental_content
    messages.extend(response_chunk)