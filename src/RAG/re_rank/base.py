from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document


class BaseReranker(ABC):
    """重排序器的抽象基类"""
    
    @abstractmethod
    def rerank(self, query: str, docs: List[Document], k: int = 5) -> List[Document]:
        """
        对文档进行重排序
        
        Args:
            query: 查询语句
            docs: 待排序的文档列表
            k: 返回的文档数量
            
        Returns:
            重排序后的文档列表
        """
        pass
