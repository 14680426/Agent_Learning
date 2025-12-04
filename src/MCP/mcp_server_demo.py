"""
FastMCP快速入门示例。
"""

from mcp.server.fastmcp import FastMCP

# 创建一个MCP服务器
mcp = FastMCP("Demo", json_response=True)


# 添加一个加法工具
@mcp.tool()
def add(a: int, b: int) -> int:
    """将两个数字相加"""
    return a * b

# 使用可流式HTTP传输运行
if __name__ == "__main__":
    # mcp.run(transport="streamable-http")
    mcp.run(transport="stdio")
    # mcp.run(transport="sse")