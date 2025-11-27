"""
文本分块基类定义
"""

from abc import ABC, abstractmethod
from typing import List, Union


class TextSplitter(ABC):
    """
    文本分块器抽象基类
    定义统一的文本分块接口
    """

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        将文本分割成多个块
        
        Args:
            text: 待分割的文本
            
        Returns:
            List[str]: 分割后的文本块列表
        """
        pass

    @abstractmethod
    def split_documents(self, documents) -> List:
        """
        将文档列表分割成多个文档块
        
        Args:
            documents: 待分割的文档列表
            
        Returns:
            List: 分割后的文档块列表
        """
        pass