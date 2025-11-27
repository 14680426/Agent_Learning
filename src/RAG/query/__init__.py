from .base import BaseRAG
from .multi_query import MultiQueryRAG
from .single_query import SingleQueryRAG
from .abstract_query import AbstractQueryRAG


__all__ = [
    "BaseRAG",
    "MultiQueryRAG",
    "SingleQueryRAG",
    "AbstractQueryRAG"
]


def get_rag_instance(rag_type: str, vectorstore, llm=None):
    """
    获取RAG实例的工厂方法
    
    Args:
        rag_type: RAG类型 ('single', 'multi_query')
        vectorstore: 向量数据库
        llm: 语言模型（某些方法需要）
        
    Returns:
        BaseRAG: RAG实例
    """
    if rag_type == "single":
        return SingleQueryRAG(vectorstore, llm)
    elif rag_type == "multi_query":
        return MultiQueryRAG(vectorstore, llm)
    elif rag_type == "abstract":
        return AbstractQueryRAG(vectorstore, llm)
    else:
        raise ValueError(f"Unsupported RAG type: {rag_type}")