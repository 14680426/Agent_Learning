from sentence_transformers import CrossEncoder
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.config import Settings
from RAG.re_rank.base import BaseReranker
import os


class LocalCrossEncoderReranker(BaseReranker):
    """使用本地交叉编码器模型重排序"""
    
    def __init__(self, embeddings, model_path: str = None):
        self.embeddings = embeddings
        
        if model_path is None:
            raise ValueError("必须提供交叉编码器模型路径，无法使用默认路径。请指定model_path参数。")
        
        print(f"🔍 使用交叉编码器模型路径: {model_path}")
        
        # 直接加载交叉编码器模型
        print(f"📥 加载交叉编码器模型: {model_path}")
        self.cross_encoder = CrossEncoder(model_path)
        
        # 设置持久化目录
        src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.persist_directory: str = os.path.join(src_dir, "RAG/tools/chroma_db")

        client = chromadb.PersistentClient(path=self.persist_directory)  
        print("🔍 直接连接到现有的 Chroma 集合 'local_rerank'")
        
        # 创建新的向量存储
        self.vectorstore = Chroma(
            collection_name="local_rerank",
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        print("✅ 交叉编码器模型加载完成")
    
    def rerank(self, query: str, docs: List[Document], k: int = 5) -> List[Document]:
        """
        使用交叉编码器对文档进行重排序
        
        Args:
            query: 查询语句
            docs: 待排序的文档列表
            k: 返回的文档数量
            
        Returns:
            重排序后的文档列表
        """
        # 准备查询-文档对
        query_doc_pairs = [
            [query, doc.page_content] for doc in docs
        ]
        
        # 使用交叉编码器计算相关性分数
        print(f"🎯 使用交叉编码器重新评分...")
        scores = self.cross_encoder.predict(query_doc_pairs)
        
        # 根据分数排序
        scored_docs = [
            {'document': doc, 'score': score}
            for doc, score in zip(docs, scores)
        ]
        
        # 按分数降序排序
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        
        # 返回top-k
        reranked_docs = scored_docs[:k]
        
        print("\n重排序结果:")
        for i, item in enumerate(reranked_docs):
            print(f"{i+1}. [得分: {item['score']:.4f}] {item['document'].page_content[:100]}...")
        
        return [item['document'] for item in reranked_docs]

    def retrieve_and_rerank(
        self,
        query: str,
        initial_k: int = 20,
        final_k: int = 5
    ):
        """检索并重排序"""
        # 1. 初始检索
        initial_docs = self.vectorstore.similarity_search(query, k=initial_k)
        
        # 2. 调用rerank方法进行重排序
        final_docs = self.rerank(query, initial_docs, final_k)
        
        return final_docs
    
    def load_vectorstore(self, embedding_model_path: str = None):
        """加载向量存储"""
        print("📂 加载向量存储...")
        
        # 如果没有提供嵌入模型路径，则提示用户必须提供路径
        if embedding_model_path is None:
            raise ValueError("必须提供嵌入模型路径，无法使用默认路径。请指定embedding_model_path参数。")
        
        print(f"✅ 使用嵌入模型: {embedding_model_path}")
            
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)
        # 获取src目录路径
        src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.vectorstore = Chroma(
            persist_directory=os.path.join(src_dir, "RAG/tools/chroma_db"),
            embedding_function=embeddings,
            collection_name="local_pdf_chunks"
        )
        print("✅ 向量存储加载完成")

def main():
    """测试本地交叉编码器重排序功能"""
    # 由于依赖特定的嵌入模型和环境配置，这里只展示测试结构
    # 实际测试需要根据具体环境配置进行
    
    print("=" * 50)
    print("本地交叉编码器重排序测试")
    print("=" * 50)
    
    
    from langchain_core.documents import Document
    from langchain_community.embeddings import HuggingFaceEmbeddings
    
    # 获取src目录路径
    src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(src_dir, "Models", "maidalun", "bce-embedding-base_v1")
    
    # 初始化嵌入模型
    embeddings = HuggingFaceEmbeddings(model_name=model_path)
    
    # 创建重排序器实例，需要提供模型路径
    reranker_model_path = os.path.join(src_dir, "Models", "ms-marco-MiniLM-L-6-v2")
    reranker = LocalCrossEncoderReranker(embeddings, reranker_model_path)
    
    # 创建测试文档
    test_docs = [
        Document(page_content="人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。", metadata={"source": "ai_intro"}),
        Document(page_content="机器学习是人工智能的一个子集，它使计算机能够从数据中学习并做出决策或预测。", metadata={"source": "ml_intro"}),
        Document(page_content="深度学习是机器学习的一个分支，它模仿人脑的工作方式处理数据和创建模式，用于决策制定。", metadata={"source": "dl_intro"}),
        Document(page_content="自然语言处理是人工智能领域中的一个重要方向，涉及计算机与人类语言之间的交互。", metadata={"source": "nlp_intro"}),
        Document(page_content="计算机视觉是人工智能的一个领域，专注于教会计算机如何解释和理解视觉世界。", metadata={"source": "cv_intro"})
    ]
    
    # 添加文档到存储中
    reranker.add_documents(test_docs)
    
    # 测试查询
    query = "什么是人工智能？"
    
    print("=" * 50)
    print("测试本地交叉编码器重排序")
    print("=" * 50)
    print(f"查询: {query}")
    print("-" * 30)
    
    # 执行检索和重排序
    results = reranker.retrieve_and_rerank(query, initial_k=10, final_k=3)
    
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)


if __name__ == "__main__":
    main()