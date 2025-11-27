"""
递归字符文本分块器
"""

from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from RAG.tools.splits.base import TextSplitter


class RecursiveTextSplitter(TextSplitter):
    """
    递归字符文本分块器
    继承自 TextSplitter 基类
    """

    def __init__(self, 
                 chunk_size: int = 250,
                 chunk_overlap: int = 30,
                 separators: Optional[List[str]] = None):
        """
        初始化递归字符文本分块器
        
        Args:
            chunk_size: 文本块大小
            chunk_overlap: 文本重叠大小
            separators: 分隔符列表
        """
        if separators is None:
            separators = ["\n\n", "\n", ". ", "。", "! ", "！ ", "? ", "？ ", " ", ""]
        
        self.splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def split_text(self, text: str) -> List[str]:
        """
        将文本分割成多个块
        
        Args:
            text: 待分割的文本
            
        Returns:
            List[str]: 分割后的文本块列表
        """
        return self.splitter.split_text(text)

    def split_documents(self, documents) -> List:
        """
        将文档列表分割成多个文档块
        
        Args:
            documents: 待分割的文档列表
            
        Returns:
            List: 分割后的文档块列表
        """
        all_chunks = []
        for doc in documents:
            # 确保doc是一个文档对象而不是字符串
            if isinstance(doc, str):
                # 如果是字符串，创建一个简单的文档对象
                text = doc
                source_metadata = {}
            else:
                # 如果是文档对象，提取文本内容
                text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                source_metadata = getattr(doc, 'metadata', {})
            
            chunks = self.split_text(text)
            
            # 为每个块创建文档对象
            for i, chunk in enumerate(chunks):
                # 创建简单的文档对象结构
                chunk_doc = type('Document', (), {
                    'page_content': chunk,
                    'metadata': source_metadata.copy()
                })()
                # 添加块索引信息
                chunk_doc.metadata["chunk_index"] = i
                chunk_doc.metadata["total_chunks"] = len(chunks)
                # 确保包含源文件信息
                if "source" not in chunk_doc.metadata and "source" in source_metadata:
                    chunk_doc.metadata["source"] = source_metadata["source"]
                all_chunks.append(chunk_doc)
                
        return all_chunks


if __name__ == "__main__":
    # 创建示例文本
    sample_text = """
    人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
    人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大。
    可以设想，未来人工智能带来的科技产品，将会是人类智慧的"容器"。
    人工智能可以对人的意识、思维的信息过程的模拟。
    人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。
    人工智能是一门极富挑战性的科学。
    从事这项工作的人必须懂得计算机知识，心理学和哲学。
    人工智能是包括十分广泛的科学，它由不同的领域组成，
    如机器学习，计算机视觉等等，总的说来，人工智能研究的一个主要目标是使机器能够胜任一些通常需要人类智能才能完成的复杂工作。
    """

    # 实例化递归字符文本分块器
    splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=20)
    
    # 调用分割方法
    chunks = splitter.split_text(sample_text)
    
    # 打印结果
    print(f"原始文本长度: {len(sample_text)} 字符")
    print(f"分割成 {len(chunks)} 个文本块:")
    for i, chunk in enumerate(chunks):
        print(f"\n块 {i+1}: {repr(chunk)}")