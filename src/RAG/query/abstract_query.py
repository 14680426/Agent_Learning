from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from RAG.query.base import BaseRAG


class AbstractQueryRAG(BaseRAG):
    """抽象化查询RAG实现（Step Back RAG）"""
    
    def __init__(self, vectorstore, llm):
        """
        初始化抽象化查询RAG
        
        Args:
            vectorstore: 向量数据库
            llm: 语言模型（用于生成抽象问题和最终答案）
        """
        super().__init__(vectorstore, llm)
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        print("✅ 抽象化查询RAG初始化完成")
    
    def retrieve(self, question: str, k: int = 3) -> List[Document]:
        """执行抽象化查询RAG检索"""
        print(f"❓ 原始问题: {question}")
        print("-" * 50)
        
        # Step 1: 生成抽象化问题
        abstract_question = self._generate_abstract_question(question)
        print(f"📚 抽象化问题: {abstract_question}")
        
        # Step 2: 检索背景知识（抽象问题）
        background_docs = self._retrieve_documents(abstract_question, k)
        print(f"🔍 检索到 {len(background_docs)} 个背景知识文档")
        
        # Step 3: 检索具体信息（原始问题）
        specific_docs = self._retrieve_documents(question, k)
        print(f"📄 检索到 {len(specific_docs)} 个具体信息文档")
        
        # Step 4: 合并并去重文档
        all_docs = background_docs + specific_docs
        unique_docs = self._get_unique_documents([all_docs])
        print(f"✨ 合并去重后剩余 {len(unique_docs)} 个文档")
        
        return unique_docs
    
    def _generate_abstract_question(self, question: str) -> str:
        """生成抽象化问题"""
        # 定义提示模板
        prompt_template = PromptTemplate.from_template(
            "你是一个AI助手，擅长将具体问题转化为更抽象、更概括的问题。\n\n"
            "给定一个具体问题，请生成一个能够提供背景知识的更抽象问题。这个抽象的问题要与原来的问题有一定的关联性，不能够虚构。\n\n"
            "示例：\n"
            "具体问题: \"什么是量子计算？\"\n"
            "Step Back问题: \"量子计算的基本原理并且有什么应用？\"\n\n"
            "具体问题: \"请检索量子计算的相关信息。\"\n"
            "Step Back问题: \"请用学术语言描述量子计算的基本原理。\"\n\n"
            "现在请为以下具体问题生成Step Back问题：\n\n"
            "具体问题: {question}\n"
            "Step Back问题:"
        )
        
        # 格式化提示
        prompt = prompt_template.format(question=question)
        
        # 使用LLM生成抽象问题
        try:
            response = self.llm.invoke(prompt)
            abstract_question = response.content.strip()
        except Exception as e:
            print(f"⚠️  LLM调用出错: {e}")
            # 返回原始问题作为后备方案
            return question
        
        return abstract_question
    
    def _retrieve_documents(self, query: str, k: int = 3) -> List[Document]:
        """检索相关文档"""
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return docs[:k]
        except Exception as e:
            print(f"❌ 检索失败 '{query}': {e}")
            return []
    
    def _get_unique_documents(self, documents: List[List[Document]]) -> List[Document]:
        """去重文档"""
        unique_docs = {}
        for doc_list in documents:
            for doc in doc_list:
                content = doc.page_content
                if content not in unique_docs:
                    unique_docs[content] = doc
        return list(unique_docs.values())


if __name__ == "__main__":
    # 测试代码 - 展示如何使用AbstractQueryRAG
    print("AbstractQueryRAG检索测试")
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
    from langchain_chroma import Chroma
    vectorstore = Chroma(
        persist_directory="RAG/tools/chroma_db",
        embedding_function=embeddings,
        collection_name="local_pdf_chunks"  # 指定默认集合名称
    )
    
    # 创建AbstractQueryRAG对象
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    rag = AbstractQueryRAG(vectorstore, llm)
    
    # 执行查询
    question = "什么是滑动检测？"
    print(f"\n🔍 查询问题: {question}")
    docs = rag.retrieve(question)
    print(f"📚 检索到 {len(docs)} 个文档")
    
    # 显示检索到的文档片段
    for i, doc in enumerate(docs):
        print(f"\n[{i+1}] {doc.page_content[:200]}...")