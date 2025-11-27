"""
语义文本分块器
"""

from typing import List, Optional
import numpy as np
import re
import os
from modelscope.hub.snapshot_download import snapshot_download
from langchain_community.embeddings import HuggingFaceEmbeddings
from RAG.tools.splits.base import TextSplitter


class SemanticTextSplitter(TextSplitter):
    """
    语义文本分块器
    继承自 TextSplitter 基类
    """

    def __init__(self, 
                 embeddings=None,
                 similarity_threshold: float = 0.6):
        """
        初始化语义文本分块器
        
        Args:
            embeddings: 嵌入模型，如果为None则加载默认的本地模型
            similarity_threshold: 相似度阈值，低于此值的句子将被分到不同块中
        """
        if embeddings is None:
            # 创建本地模型目录，从src目录开始
            local_models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Models")
            local_models_dir = os.path.abspath(local_models_dir)
            os.makedirs(local_models_dir, exist_ok=True)

            # 检查本地是否存在模型，如果存在则直接使用，否则从 ModelScope 下载
            model_id = "maidalun/bce-embedding-base_v1"

            # 构建本地模型路径
            local_model_path = os.path.join(local_models_dir, "maidalun", "bce-embedding-base_v1")

            # 如果本地模型不存在或者目录为空，则从 ModelScope 下载
            if not os.path.exists(local_model_path) or not os.listdir(local_model_path):
                print(f"📥 本地未找到模型 {model_id}，正在从 ModelScope 下载...")
                # 确保模型目录存在
                os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
                local_model_path = snapshot_download(model_id, local_dir=local_model_path)
                print(f"✅ 模型下载完成: {local_model_path}")
            else:
                print(f"✅ 使用本地模型: {local_model_path}")

            # 初始化嵌入模型（启用 GPU）
            self.embeddings = HuggingFaceEmbeddings(
                model_name=local_model_path,
                model_kwargs={"device": "cuda"},               # 使用 GPU 加速
                encode_kwargs={"normalize_embeddings": True}   # 归一化便于计算余弦相似度
            )
        else:
            self.embeddings = embeddings
            
        self.similarity_threshold = similarity_threshold
        print(f"语义文本分块器已初始化，相似度阈值为 {similarity_threshold}")
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            float: 余弦相似度值
        """
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _split_sentences(self, text: str) -> List[str]:
        """
        分割句子（中文优化）
        
        Args:
            text: 待分割的文本
            
        Returns:
            List[str]: 句子列表
        """
        sentence_endings = r'[。！？；\n]'
        sentences = re.split(sentence_endings, text)
        # 更严格的过滤条件
        sentences = [s.strip() for s in sentences if s and s.strip()]
        # 确保句子以标点结尾
        sentences = [s + '。' for s in sentences if not s.endswith(('。', '!', '?', '！', '？', '；'))]
        return sentences

    def _get_sentence_embeddings(self, sentences: List[str]) -> List[np.ndarray]:
        """
        获取句子嵌入
        
        Args:
            sentences: 句子列表
            
        Returns:
            List[np.ndarray]: 句子嵌入向量列表
        """
        embeddings = []
        for sent in sentences:
            # 增加文本验证
            if not sent or not isinstance(sent, str):
                print(f"警告: 跳过无效句子: {repr(sent)}")
                # 使用零向量作为占位符
                embeddings.append(np.zeros(768))  # 假设768维向量，可根据实际情况调整
                continue
                
            # 清理文本，移除可能导致问题的特殊字符
            cleaned_sent = self._clean_text(sent)
            if not cleaned_sent:
                print(f"警告: 清理后句子为空: {repr(sent)}")
                embeddings.append(np.zeros(768))
                continue
                
            try:
                emb = self.embeddings.embed_query(cleaned_sent)
                embeddings.append(np.array(emb))
            except Exception as e:
                print(f"警告: 处理句子 '{sent[:50]}...' 时出错: {e}")
                # 出错时使用零向量
                embeddings.append(np.zeros(768))
        return embeddings

    def _clean_text(self, text: str) -> str:
        """
        清理文本，移除可能导致tokenizer错误的字符
        
        Args:
            text: 原始文本
            
        Returns:
            str: 清理后的文本
        """
        if not text:
            return ""
        
        # 移除控制字符和其他可能导致问题的字符
        cleaned = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        # 移除多余的空白字符
        cleaned = ' '.join(cleaned.split())
        return cleaned

    def split_text(self, text: str) -> List[str]:
        """
        基于语义相似度分块
        
        Args:
            text: 待分割的文本
            
        Returns:
            List[str]: 分割后的文本块列表
        """
        if not text or not isinstance(text, str):
            return [text] if text else [""]
        
        # 1. 按句子分割
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [text]
        
        # 2. 计算每个句子的嵌入
        embeddings = self._get_sentence_embeddings(sentences)
        
        # 3. 计算相邻句子的相似度
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self.cosine_similarity(embeddings[i], embeddings[i+1])
            similarities.append(sim)
        
        # 4. 在相似度低的地方切分
        chunks = []
        current_chunk = [sentences[0]]
        
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                chunks.append(''.join(current_chunk))
                current_chunk = [sentences[i+1]]
            else:
                current_chunk.append(sentences[i+1])
        
        # 添加最后一块
        if current_chunk:
            chunks.append(''.join(current_chunk))
        
        return chunks

    def split_documents(self, documents) -> List:
        """
        将文档列表分割成多个文档块
        
        Args:
            documents: 待分割的文档列表
            
        Returns:
            List: 分割后的文档块列表
        """
        # 这里为了保持接口一致性，但实际使用中可能需要更复杂的实现
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

    # 实例化语义文本分块器
    splitter = SemanticTextSplitter(similarity_threshold=0.5)
    
    # 调用分割方法
    chunks = splitter.split_text(sample_text)
    
    # 打印结果
    print(f"语义分块结果: {len(chunks)} 块")
    for i, chunk in enumerate(chunks):
        print(f"\n块 {i+1}:\n{chunk}")