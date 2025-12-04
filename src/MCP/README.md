# MCP (Model Coordination Protocol) 项目

本项目演示了如何使用MCP协议与大语言模型进行交互，提供了两种不同的工具集成方案：本地工具服务和远程工具服务。

## 项目结构

```
MCP/
├── client_demo.py          # MCP客户端示例
├── mcp_server_demo.py      # 本地MCP服务器示例
├── remote_server_demo.py   # 远程MCP服务器示例
└── README.md              # 本文件
```

## 方案一：本地MCP工具服务

本地MCP工具服务允许您在本地运行自定义工具，并通过MCP协议与大语言模型进行交互。

### 特点

- 工具运行在本地环境中
- 可以直接访问本地资源和文件系统
- 更好的安全性，数据不会发送到外部服务器
- 更快的响应速度

### 示例工具

[mcp_server_demo.py]提供了一个简单的加法工具示例：

```python
@mcp.tool()
def add(a: int, b: int) -> int:
    """将两个数字相加"""
    return a + b  
```

### 运行方式

```bash
# 使用client_demo.py连接到本地MCP服务器
python client_demo.py mcp_server_demo.py
```

## 方案二：远程MCP工具服务

远程MCP工具服务集成了第三方提供的MCP工具，例如时间查询和地图服务。

### 特点

- 工具运行在远程服务器上
- 可以访问专业的第三方服务（如时间服务、地图服务等）
- 无需在本地维护复杂的服务基础设施
- 依赖网络连接和第三方服务的可用性

### 集成的远程工具

[remote_server_demo.py] 集成了以下远程工具：

1. **时间服务** - 提供精确的时间查询功能
2. **高德地图服务** - 提供地理位置和路线规划功能

### 环境变量配置

使用远程工具需要配置以下环境变量：

```env
BINGXING_API_KEY=your_bingxing_api_key
BINGXING_BASE_URL=your_bingxing_base_url
ZHIPUAI_API_KEY=your_zhipuai_api_key
DASHSCOPE_API_KEY=your_dashscope_api_key
```

### 运行方式

```bash
# 直接运行远程MCP服务器
python remote_server_demo.py
```


## 注意事项

1. **API密钥安全**：请妥善保管您的API密钥，不要将其提交到代码仓库中
2. **网络连接**：使用远程工具时需要稳定的网络连接
3. **工具实现**：本地工具示例中存在一个bug，add函数实际执行的是乘法而非加法
4. **依赖项**：确保安装了所有必要的依赖项

## 依赖项

- `mcp` - MCP协议库
- `openai` - OpenAI API客户端
- `qwen-agent` - Qwen智能体库
- `python-dotenv` - 环境变量管理

可以通过以下命令安装依赖：

```bash
pip install mcp openai qwen-agent python-dotenv
```