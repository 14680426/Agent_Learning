from pydantic import BaseModel, Field
from Models import ModelManager
from langchain_core.tools import tool
@tool('web_search', parse_docstring=True)#parse_docstring=True 的作用是让大模型能够读取并理解函数文档字符串中的参数说明和功能描述，从而更准确地调用工具。
def web_search(query: str) -> str: #必须要加上谷歌格式的注释，这样大模型会根据注释理解这个工具的作用
    """使用谷歌搜索，返回搜索结果。
    
    Args:
        query (str): 搜索内容。
    
    Returns:
        str: 搜索结果。
    """
    try:
        # 初始化模型管理器并获取模型实例
        model_manager = ModelManager()
        zhipuai_client = model_manager.get_zhipuai_model()
        resp = zhipuai_client.web_search.web_search(
            search_engine='search_std',  #搜索引擎 可选
            search_query=query,
        )
        if resp.search_result:
            return "\n\n".join([d.content for d in resp.search_result])
        else:
            return "没有搜索到结果"
    except Exception as e:
        return str(e)
    

class WebSearchInput(BaseModel):
    query: str = Field(..., description="搜索内容")


if __name__ == "__main__":
    print(web_search.name)
    print(web_search.description)
    print(web_search.args_schema.model_json_schema())
    print(web_search.args)

    result = web_search.invoke({'query': '如何使用langchain?'})
    print(result)