# tools/mcp_tool_wrapper.py
from langchain.tools import tool
from qwen_agent.agents import Assistant
from dotenv import load_dotenv
load_dotenv() 
import os
import logging

# 设置日志
logger = logging.getLogger(__name__)

@tool
def mcp_assistant(query: str):
    """
    使用MCP助手执行复杂任务，包括获取实时时间、地理位置查询、路线规划等。
    当需要获取当前时间、查询地理位置信息或进行路线规划时使用此工具。
    
    Args:
        query (str): 用户想要查询的问题
    """
    try:
        # 创建助手实例
        assistant = _create_assistant()
        
        # 创建消息历史
        messages = [{"role": "user", "content": query}]
        
        # 执行助手并收集响应，只保留最终结果
        response_text = ""
        for response_chunk in assistant.run(messages):
            new_response = response_chunk[-1]
            if new_response.get("role") == "assistant" and "content" in new_response:
                response_text = new_response["content"] 
                
        return response_text
    except Exception as e:
        logger.error(f"MCP工具执行出错: {str(e)}")
        return f"MCP工具执行出错: {str(e)}"

def _create_assistant():
    llm_cfg = {
        "model": 'DeepSeek-V3.1-Terminus',
        "model_server": os.environ.get("BINGXING_BASE_URL", ""),
        "api_key": os.environ.get("BINGXING_API_KEY", ""),
    }
    
    tools = [
        {
            "type": "mcp",
            "mcpServers": {
                "time": {
                    "type": "sse",
                    "url": "https://open.bigmodel.cn/api/mcp-broker/proxy/time/sse",
                    "headers": {
                        "Authorization": f"Bearer {os.environ.get('ZHIPUAI_API_KEY', '')}"
                    }
                },
                "amap-maps": {
                    "type": "sse",
                    "url": "https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/sse",
                    "headers": {
                        "Authorization": f"Bearer {os.environ.get('DASHSCOPE_API_KEY', '')}"
                    }
                },
            }
        }
    ]
    
    return Assistant(
        llm=llm_cfg,
        name="MCP助手",
        description="可以执行实时时间查询、地理位置查询规划等任务的助手",
        system_message="你是会查询实时时间、转换时间以及能够执行路线导航规划的助手",
        function_list=tools,
    )