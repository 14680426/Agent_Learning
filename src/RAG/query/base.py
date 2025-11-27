from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document


class BaseRAG(ABC):
    """RAG检索抽象基类"""
    
    def __init__(self, vectorstore, llm=None):
        """
        初始化RAG检索基类
        
        Args:
            vectorstore: 向量数据库
            llm: 语言模型（某些检索方法可能需要）
        """
        self.vectorstore = vectorstore
        self.llm = llm
    
    @abstractmethod
    def retrieve(self, question: str, **kwargs) -> List[Document]:
        """
        执行检索的抽象方法
        
        Args:
            question: 用户问题
            **kwargs: 其他参数
            
        Returns:
            List[Document]: 检索到的文档列表
        """
        pass