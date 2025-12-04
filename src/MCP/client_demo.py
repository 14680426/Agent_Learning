import asyncio
from typing import Optional
import subprocess
import sys
import os
import time
import signal
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


from dotenv import load_dotenv
load_dotenv()  # load environment variables from .env

from openai import OpenAI
import json

api_key = os.environ["BINGXING_API_KEY"]
base_url = os.environ["BINGXING_BASE_URL"]

model_type='DeepSeek-V3.1-Terminus'

# 存储服务端进程引用
server_process = None


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = OpenAI(api_key=api_key, base_url=base_url)

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        print(f"🚀 启动MCP客户端: {server_script_path}")
        
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])


    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        print("\n正在处理Query:", repr(query))
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()

        available_tools = []

        for tool in response.tools:
            tool_schema = getattr(
                tool,
                "inputSchema",
                {"type": "object", "properties": {}, "required": []},
            )

            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool_schema,
                },
            }
            available_tools.append(openai_tool)

        # Initial Claude API call
        model_response = self.anthropic.chat.completions.create(
            model=model_type,
            max_tokens=1000,
            messages=messages,
            tools=available_tools,
        )

        # Process response and handle tool calls
        tool_results = []

        # 添加完整的模型响应到消息历史中
        messages.append(model_response.choices[0].message.model_dump())
        print("模型首次响应:", messages[-1])
        
        if (model_response.choices[0].message.tool_calls and 
            len(model_response.choices[0].message.tool_calls) > 0):
            
            for tool_call in model_response.choices[0].message.tool_calls:
                tool_args = json.loads(tool_call.function.arguments)

                tool_name = tool_call.function.name
                result = await self.session.call_tool(tool_name, tool_args)
                print("工具调用结果:", tool_name, tool_args, result)
                tool_results.append({"call": tool_name, "result": result})

                messages.append(
                    {
                        "role": "tool",
                        "content": str(result),
                        "tool_call_id": tool_call.id,
                    }
                )

            # Get next response from Claude
            response = self.anthropic.chat.completions.create(
                model=model_type,
                max_tokens=1000,
                messages=messages,
            )

            messages.append(response.choices[0].message.model_dump())
            print("模型最终响应:", messages[-1])

        return messages[-1]["content"]



    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        print("\n🤖 MCP 客户端已启动！输入 'quit' 退出")

        while True:
            query = input("Query: ")
            
            query = query.strip()

            if query.lower() == 'quit':
                break

            if not query:  # 如果输入为空，则重新提示
                print("请输入有效的查询内容")
                continue

            response = await self.process_query(query)
            print("\n" + response)

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
        
        # 关闭服务端进程
        global server_process
        if server_process is not None:
            try:
                if server_process.poll() is None:  # 进程仍在运行
                    print("🛑 关闭MCP服务端...")
                    server_process.terminate()
                    try:
                        server_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        server_process.kill()
                # 如果进程已退出，无需操作
            except (OSError, ValueError) as e:
                print(f"⚠️ 清理服务端进程时发生异常: {e}")
            finally:
                server_process = None  # 避免重复操作


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    print("Connecting to server...")
    print(sys.argv)
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())