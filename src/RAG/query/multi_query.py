from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from RAG.query.base import BaseRAG


class MultiQueryRAG(BaseRAG):
    """Multi-Query RAG检索实现"""
    
    def __init__(self, vectorstore, llm):
        """
        初始化Multi-Query RAG检索
        
        Args:
            vectorstore: 向量数据库
            llm: 语言模型（用于生成查询变体）
        """
        super().__init__(vectorstore, llm)
        self.llm = llm
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        print("✅ Multi-Query RAG检索初始化完成")
    
    def retrieve(self, question: str, k: int = 3) -> List[Document]:
        """执行Multi-Query RAG检索"""
        print(f"🎯 原始问题: {question}")
        print("-" * 50)
        
        # Step 1: 生成查询变体
        queries = self._generate_query_variants(question)
        print(f"📝 生成了 {len(queries)} 个查询:")
        for i, q in enumerate(queries, 1):
            print(f"   {i}. {q}")
        
        # Step 2: 检索文档
        all_docs = self._retrieve_documents_multi(queries, k)
        total_docs_before_dedup = sum(len(docs) for docs in all_docs)
        print(f"📚 检索到 {total_docs_before_dedup} 个文档（含重复）")
        
        # Step 3: 去重
        unique_docs = self._get_unique_documents(all_docs)
        print(f"✨ 去重后剩余 {len(unique_docs)} 个文档")
        
        return unique_docs
    
    def _generate_query_variants(self, question: str) -> List[str]:
        """生成查询变体"""
        # 定义提示模板
        query_prompt_template = PromptTemplate.from_template(
            "你是一个AI助手，擅长将用户的问题改写成多个语义相同但表达不同的搜索查询。每个查询应简洁、独立，适合用于向量检索。\n\n"
            "原始问题：{question}\n\n"
            "请生成3个不同的搜索查询，每行一个，不要编号，不要解释："
        )
        
        # 格式化提示
        query_prompt = query_prompt_template.format(question=question)
        
        # 使用ChatOpenAI生成查询变体，修复参数问题
        try:
            response = self.llm.invoke(query_prompt)
            response_text = response.content.strip()
        except Exception as e:
            print(f"⚠️  LLM调用出错: {e}")
            # 返回原始问题作为后备方案
            return [question]
        
        # 解析生成的查询
        queries = []
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) > 5:
                # 移除可能的编号标记
                if line[0].isdigit() and (line[1] == '.' or line[1] == '、'):
                    query = line[2:].strip()
                else:
                    query = line
                queries.append(query)
        
        # 确保包含原始问题并去重
        all_queries = [question] + queries
        unique_queries = []
        seen = set()
        for q in all_queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)
        
        return unique_queries[:4]  # 最多返回4个查询
    
    def _retrieve_documents_multi(self, queries: List[str], k: int = 3) -> List[List[Document]]:
        """检索所有查询的文档"""
        all_docs = []
        for query in queries:
            try:
                docs = self.vectorstore.similarity_search(query, k=k)
                all_docs.append(docs)
                print(f"   🔍 '{query}': 找到 {len(docs)} 个文档")
            except Exception as e:
                print(f"❌ 检索失败 '{query}': {e}")
                all_docs.append([])
        return all_docs
    
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
    # 测试代码 - 展示如何使用MultiQueryRAG
    print("MultiQueryRAG检索测试")
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
    
    # 创建MultiQueryRAG对象
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    rag = MultiQueryRAG(vectorstore, llm)
    
    # 执行查询
    question = "什么是滑动检测？"
    print(f"\n🔍 查询问题: {question}")
    docs = rag.retrieve(question)
    print(f"📚 检索到 {len(docs)} 个文档")
    
    # 显示检索到的文档片段
    for i, doc in enumerate(docs):
        print(f"\n[{i+1}] {doc.page_content[:200]}...")
