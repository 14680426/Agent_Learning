import os
import sys

# 获取当前文件所在的目录(src/RAG)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取src目录
src_dir = os.path.dirname(current_dir)
# 将src目录添加到系统路径中
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from dotenv import load_dotenv
# 加载项目根目录中的.env文件
project_root = os.path.dirname(src_dir)
load_dotenv(dotenv_path=os.path.join(project_root, '.env'))

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from RAG.query import MultiQueryRAG, AbstractQueryRAG, SingleQueryRAG
from RAG.re_rank.local_cross_encoder import LocalCrossEncoderReranker
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from RAG.re_rank import LocalCrossEncoderReranker


class RAG:
    """RAG主类，整合多角度查询和交叉排序功能"""
    
    def __init__(self, 
                 vectorstore: Chroma,
                 llm: BaseLanguageModel,
                 embeddings: Embeddings,
                 reranker: LocalCrossEncoderReranker = None):
        """
        初始化RAG系统
        
        Args:
            vectorstore: 向量数据库
            llm: 语言模型
            embeddings: 嵌入模型
            reranker: 重排序模型
        """
        self.vectorstore = vectorstore
        self.llm = llm
        self.embeddings = embeddings
        self.reranker = reranker
        print("✅ RAG系统初始化完成")
    
    def multi_query_retrieve(self, question: str, k: int = 3) -> List[Document]:
        """
        多角度查询方法
        
        Args:
            question: 用户问题
            k: 返回文档数量
            
        Returns:
            检索到的文档列表
        """
        self.multi_query_retriever = MultiQueryRAG(self.vectorstore, self.llm)

        print("🔄 执行多角度查询...")
        docs = self.multi_query_retriever.retrieve(question, k)
        print(f"📚 多角度查询获得 {len(docs)} 个文档")
        return docs
    
    def abstract_query_retrieve(self, question: str, k: int = 3) -> List[Document]:
        """
        抽象化查询方法
        
        Args:
            question: 用户问题
            k: 返回文档数量
            
        Returns:
            检索到的文档列表
        """
        self.abstract_query_retriever = AbstractQueryRAG(self.vectorstore, self.llm)

        print("🔄 执行抽象化查询...")
        docs = self.abstract_query_retriever.retrieve(question, k)
        print(f"📚 抽象化查询获得 {len(docs)} 个文档")
        return docs
    
    def single_query_retrieve(self, question: str, k: int = 3) -> List[Document]:
        """
        单角度查询方法
        
        Args:
            question: 用户问题
            k: 返回文档数量
            
        Returns:
            检索到的文档列表
        """
        self.single_query_retriever = SingleQueryRAG(self.vectorstore, self.llm)

        print("🔄 执行普通查询...")

        docs = self.single_query_retriever.retrieve(question, k)
        print(f"📚 普通查询获得 {len(docs)} 个文档")
        return docs

    def similarity_search(self, question: str, k: int = 4) -> List[Document]:
        """
        相似度搜索方法（待实现）
        
        Args:
            question: 用户问题
            k: 返回文档数量
            
        Returns:
            检索到的文档列表
        """
        # TODO: 实现相似度搜索
        pass
    
    def hybrid_search(self, question: str, k: int = 4) -> List[Document]:
        """
        混合搜索方法（待实现）
        
        Args:
            question: 用户问题
            k: 返回文档数量
            
        Returns:
            检索到的文档列表
        """
        # TODO: 实现混合搜索
        pass
    
    def cross_encoder_rerank(self, question: str, docs: List[Document], k: int = 5) -> List[Document]:
        """
        交叉编码器重排序方法
        
        Args:
            question: 用户问题
            docs: 待排序文档列表
            k: 返回文档数量
            
        Returns:
            重排序后的文档列表
        """
        print("⚖️ 执行交叉编码器重排序...")
        reranked_docs = self.reranker.rerank(question, docs, k)
        print(f"✅ 交叉编码器重排序完成，返回 {len(reranked_docs)} 个最相关文档")
        return reranked_docs
    
    def diversity_rerank(self, question: str, docs: List[Document], k: int = 5) -> List[Document]:
        """
        多样性重排序方法（待实现）
        
        Args:
            question: 用户问题
            docs: 待排序文档列表
            k: 返回文档数量
            
        Returns:
            重排序后的文档列表
        """
        # TODO: 实现多样性重排序
        pass
    
    def reciprocal_rank_fusion(self, question: str, docs_list: List[List[Document]], k: int = 5) -> List[Document]:
        """
        reciprocal rank fusion排序方法（待实现）
        
        Args:
            question: 用户问题
            docs_list: 多个检索器返回的文档列表
            k: 返回文档数量
            
        Returns:
            融合排序后的文档列表
        """
        # TODO: 实现reciprocal rank fusion排序
        pass
    
    def query(self, question: str, initial_k: int = 20, final_k: int = 5) -> Dict[str, Any]:
        """
        执行RAG查询：多角度查询 + 重排序
        
        Args:
            question: 用户问题
            initial_k: 初始检索文档数
            final_k: 最终返回文档数
            
        Returns:
            包含检索到的文档的结果字典
        """
        # 使用LLM分析问题并决定最适合的查询方式
        queries = self._determine_query_strategy(question)
        print(f"🔍 开始处理问题: {question}")
        print(f"🔍 选择的查询策略: {queries}")
        
        # 1. 查询
        if queries == "abstract":
            queries_docs = self.abstract_query_retrieve(question, initial_k)
        elif queries == "multi":
            queries_docs = self.multi_query_retrieve(question, initial_k)
        elif queries == "single":
            queries_docs = self.single_query_retrieve(question, initial_k)
        else:
            print("无效的查询方式")
            return {"documents": []}
        
        # 2. 重排序
        print(f"🔄 对 {len(queries_docs)} 个文档进行重排序...")
        reranked_docs = self.cross_encoder_rerank(question, queries_docs, final_k)
        print(f"✅ 重排序完成，返回 {len(reranked_docs)} 个文档")
        
        return {
            "documents": reranked_docs,
            "query_strategy": queries
        }
    
    def _determine_query_strategy(self, question: str) -> str:
        """
        使用LLM分析问题并决定最适合的查询策略
        
        Args:
            question: 用户问题
            
        Returns:
            查询策略 ("single", "multi", "abstract")
        """
        # 定义提示模板
        prompt_template = """
            你是一个智能查询策略选择助手。根据用户的问题，选择最适合的检索策略：

            策略说明：
            1. single（单查询）：适用于具体、明确的问题，涉及两个及以上的词语之间的关系，如"滑动检测有什么常用的方法？"
            2. multi（多角度查询）：适用于需要从多个角度理解的问题，如"滑动检测的原理和应用？"
            3. abstract（抽象查询）：适用于问题给出的信息比较少，只有涉及一个词语，如"滑动检测是什么？"

            请分析以下问题并选择最适合的策略，只需回复策略名称（single/multi/abstract）：

            问题：{question}

            策略：
            """.strip()
        
        # 构建提示
        prompt = prompt_template.format(question=question)
        
        try:
            # 使用LLM进行分析
            response = self.llm.invoke(prompt)
            strategy = response.content.strip().lower()
            
            # 验证策略有效性
            if strategy in ["single", "multi", "abstract"]:
                return strategy
            else:
                # 默认使用多角度查询
                print("无效的查询方式，使用默认策略")
                return "multi"
        except Exception as e:
            print(f"⚠️  策略选择出错: {e}，使用默认策略")
            # 默认使用多角度查询
            return "multi"

def main():
    """主函数 - 演示RAG系统的使用"""
    print("=" * 50)
    print("RAG系统使用示例")
    print("=" * 50)
    
    import os
    from Models import ModelManager
    modelManager = ModelManager()
    llm = modelManager.get_qwen_model()

    # 嵌入模型路径
    embedding_model_path = "./Models/maidalun/bce-embedding-base_v1"
    
    # 检查嵌入模型路径是否存在
    if not os.path.exists(embedding_model_path) or not os.listdir(embedding_model_path):
        raise ValueError(f"嵌入模型路径不存在或为空: {embedding_model_path}，请先下载模型到指定路径。")

    # 初始化嵌入模型（启用 GPU）
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_path,
        model_kwargs={"device": "cuda"},               # 使用 GPU 加速
        encode_kwargs={"normalize_embeddings": True}   # 归一化便于计算余弦相似度
    )

    # 加载Chroma向量数据库，指定集合名称
    chroma_persist_directory = "./RAG/tools/chroma_db"
    from langchain_community.vectorstores import Chroma
    vectorstore = Chroma(
        persist_directory=chroma_persist_directory,
        embedding_function=embeddings,
        collection_name="local_pdf_chunks"  # 指定默认集合名称
    )

    cross_encoder_model_path = "./Models/ms-marco-MiniLM-L-6-v2/cross-encoder"
    
    # 检查交叉编码器模型路径是否存在
    if not os.path.exists(cross_encoder_model_path) or not os.listdir(cross_encoder_model_path):
        raise ValueError(f"交叉编码器模型路径不存在或为空: {cross_encoder_model_path}，请先下载模型到指定路径。")

    # 创建RAG对象
    print("\n🔧 正在初始化RAG系统...")
    from RAG.re_rank.local_cross_encoder import LocalCrossEncoderReranker
    reranker = LocalCrossEncoderReranker(embeddings=embeddings, model_path=cross_encoder_model_path)
    rag = RAG(llm=llm, embeddings=embeddings, vectorstore=vectorstore, reranker=reranker)
    print("✅ RAG系统初始化完成")

    # 执行查询
    question = "什么是滑动检测？"
    print(f"\n🔍 查询问题: {question}")
    
    try:
        result = rag.query(question)
        print(f"\n📊 相关文档数量: {len(result['documents'])}")
        
        # 显示相关文档片段
        print("\n📄 相关文档片段:")
        for i, doc in enumerate(result['documents']):
            print(f"\n[{i+1}] {doc.page_content[:200]}...")
            
    except Exception as e:
        print(f"❌ 查询过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()