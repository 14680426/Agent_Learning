import sys
import os

# 添加项目src目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..")
sys.path.insert(0, src_dir)

from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate
from Models import ModelManager
from Agent.tools import MyWebSearchTool, RAGTool

# 初始化模型管理器并获取模型实例
model_manager = ModelManager()
llm = model_manager.get_qwen_model()
# llm = model_manager.get_deepseek_model()

# 创建工具实例
web_search_tool = MyWebSearchTool()
rag_tool = RAGTool()

# 定义系统提示词 
system_prompt = """你是一个智能助手。请尽你所能回答以下问题。你可以使用以下工具：

工具列表：
1. web_search_tool - 网络搜索引擎工具：
   - 适用于需要从互联网获取最新信息的问题
   - 可以搜索各种公开网页内容
   - 适用于通用知识、新闻、实时信息等

2. rag_tool - 本地知识库检索工具：
   - 专门用于检索与"抓取检测"和"滑动检测"相关的本地文档
   - 包含技术文档、操作手册、详细说明等
   - 适用于需要准确、专业、详细技术信息的查询
   - 此工具将返回相关文档内容，你需要基于这些内容回答用户问题

请使用以下格式：

问题：你需要回答的输入问题
思考：你应该始终思考要做什么
行动：要采取的行动，应该是 [web_search_tool, rag_tool] 中的一个
行动输入：行动的输入内容
观察：行动的结果
...（这个思考/行动/行动输入/观察可以重复N次）
思考：我现在知道最终答案了
最终答案：对原始输入问题的最终答案

重要注意事项：
1. 对于时效性内容，你必须使用多个工具进行综合分析，不能依赖单一工具
2. RAG知识库中的知识不具备时效性，对于时效性问题，必须结合网络搜索工具获取最新信息。你应该先执行RAG知识库检索，本地知识库并不完全正确，请你再继续调用web_search_tool搜索问题答案，然后结合给出最终答案。

开始！"""


# 创建 React Agent
agent_executor = create_agent(
    model=llm,
    tools=[web_search_tool, rag_tool],
    system_prompt=system_prompt
)



