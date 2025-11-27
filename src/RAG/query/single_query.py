from typing import List
from langchain_core.documents import Document
from RAG.query.base import BaseRAG


class SingleQueryRAG(BaseRAG):
    """Single-Query RAG检索实现"""
    
    def __init__(self, vectorstore, llm=None):
        """
        初始化Single-Query RAG检索
        
        Args:
            vectorstore: 向量数据库
            llm: 语言模型
        """
        super().__init__(vectorstore, llm)
        print("✅ Single-Query RAG检索初始化完成")
    
    def retrieve(self, question: str, k: int = 3) -> List[Document]:
        """执行Single-Query RAG检索"""
        print(f"🎯 问题: {question}")
        print("-" * 50)
        
        # 直接检索文档
        try:
            docs = self.vectorstore.similarity_search(question, k=k)
            print(f"📚 检索到 {len(docs)} 个文档")
        except Exception as e:
            print(f"❌ 检索失败: {e}")
            docs = []
        
        return docs


if __name__ == "__main__":
    # 测试代码 - 展示如何使用SingleQueryRAG
    print("SingleQueryRAG检索测试")
    print("=" * 50)
    
    import os
    # 创建本地模型目录
    local_models_dir = "Models"
    os.makedirs(local_models_dir, exist_ok=True)

    # 检查本地是否存在模型，如果存在则直接使用，否则从 ModelScope 下载
    from modelscope.hub.snapshot_download import snapshot_download
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
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=local_model_path,
        model_kwargs={"device": "cuda"},               # 使用 GPU 加速
        encode_kwargs={"normalize_embeddings": True}   # 归一化便于计算余弦相似度
    )

    # 加载Chroma向量数据库，指定集合名称
    from langchain_community.vectorstores import Chroma
    vectorstore = Chroma(
        persist_directory="RAG/tools/chroma_db",
        embedding_function=embeddings,
        collection_name="local_pdf_chunks"  # 指定默认集合名称
    )
    
    # 创建SingleQueryRAG对象
    rag = SingleQueryRAG(vectorstore)
    
    # 执行查询
    question = "什么是滑动检测？"
    print(f"\n🔍 查询问题: {question}")
    docs = rag.retrieve(question)
    print(f"📚 检索到 {len(docs)} 个文档")
    
    # 显示检索到的文档片段
    for i, doc in enumerate(docs):
        print(f"\n[{i+1}] {doc.page_content[:200]}...")

    # 实际使用示例:
    # rag = SingleQueryRAG(vectorstore)
    # docs = rag.retrieve("你的问题")
    # for i, doc in enumerate(docs):
    #     print(f"文档 {i+1}: {doc.page_content[:100]}...")
    #     print(f"元数据: {doc.metadata}")
    #     print("-" * 50)