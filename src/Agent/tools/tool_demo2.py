from pydantic import BaseModel, Field
from typing import Type
from langchain_core.tools import BaseTool


class SearchArgs(BaseModel):
    query: str = Field(description="需要进行网络搜索的信息，适用于通用知识、最新信息或RAG工具无法解答的问题。")

# 网络搜索的工具
class MyWebSearchTool(BaseTool):
    # 下面是定义工具的四个要素
    # 工具名字
    name: str = "web_search_tool"
    
    description: str = "搜索互联网上公开内容的工具，适用于获取通用知识、最新信息或补充RAG工具无法提供的信息"
    
    return_direct: bool = False
    
    args_schema: Type[BaseModel] = SearchArgs

    def _run(self, query) -> str:
        try:
            print("执行我的Web搜索工具，输入的参数为:", query)
            from Models import ModelManager
            model_manager = ModelManager()
            zhipuai_client = model_manager.get_zhipuai_model()
            response = zhipuai_client.web_search.web_search(
                search_engine="search_std",
                search_query=query
            )
            print(response)
            if response.search_result:
                return "\n".join([d.content for d in response.search_result])
            return '没有搜索到任何内容！'
        except Exception as e:
            print(e)
            return '哎呀！搜索失败。'