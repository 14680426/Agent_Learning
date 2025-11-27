from pydantic import BaseModel, Field
from typing import Type
from langchain_core.tools import BaseTool
import traceback


class RAGQueryArgs(BaseModel):
    query: str = Field(description="用户提出的完整问题，需要使用RAG系统基于本地技术文档知识库来回答，特别适用于抓取检测、滑动检测等专业技术问题。请传递用户的完整问题，不要简化或提取关键词。注意：对于询问最新方法、最新技术等时效性问题，RAG系统可能无法提供最新的信息，需要结合网络搜索工具使用。")


class RAGTool(BaseTool):
    # 工具名字
    name: str = "rag_tool"
    
    description: str = "使用RAG系统基于本地技术文档知识库检索相关信息，特别擅长获取抓取检测、滑动检测等专业技术问题的相关文档。该工具需要接收用户的完整问题作为输入。注意：此工具主要适用于已有文档中的技术内容，对于时效性较强的问题可能无法提供最新信息。"
    
    return_direct: bool = False
    
    args_schema: Type[BaseModel] = RAGQueryArgs

    def _run(self, query: str) -> str:
        try:
            print(f"=== RAG工具开始执行 ===")
            print(f"执行RAG工具，输入的参数为: {query}")
            
            # 导入RAG相关模块
            from RAG.rag import RAG
            from RAG.re_rank.local_cross_encoder import LocalCrossEncoderReranker
            from Models import ModelManager
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import Chroma
            import os
            
            import os

            current_dir = os.getcwd()

            basename = os.path.basename(current_dir)

            if basename == "src":
                src_dir = current_dir  # 已经在 src 目录下
            else:
                src_dir = os.path.join(current_dir, "src")  # 否则拼接 src

            print("src_dir:", src_dir)
            
            # 初始化模型管理器并获取LLM
            model_manager = ModelManager()
            llm = model_manager.get_qwen_model()
            
            # 初始化嵌入模型（使用基于src目录的路径）
            embedding_model_path = os.path.join(src_dir, "Models/maidalun/bce-embedding-base_v1")
            
            # 检查模型是否存在
            if not os.path.exists(embedding_model_path) or not os.listdir(embedding_model_path):
                print("RAG系统所需的本地嵌入模型路径错误或尚未下载")
                return "RAG系统所需的本地嵌入模型路径错误或尚未下载"
            
            print(f"使用嵌入模型路径: {embedding_model_path}")
            embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model_path,
                model_kwargs={"device": "cuda"},
                encode_kwargs={"normalize_embeddings": True}
            )
            
            # 加载Chroma向量数据库（使用基于src目录的路径）
            chroma_db_path = os.path.join(src_dir, "RAG/tools/chroma_db")
            print(f"使用Chroma数据库路径: {chroma_db_path}")
            vectorstore = Chroma(
                persist_directory=chroma_db_path,
                embedding_function=embeddings,
                collection_name="local_pdf_chunks"
            )
            
            # 初始化交叉编码器重排序器
            cross_encoder_model_path = os.path.join(src_dir, "Models/ms-marco-MiniLM-L-6-v2/cross-encoder")
            
            # 检查交叉编码器模型是否存在
            if not os.path.exists(cross_encoder_model_path) or not os.listdir(cross_encoder_model_path):
                print("RAG系统所需的交叉编码器模型路径错误或尚未下载")
                return "RAG系统所需的交叉编码器模型路径错误或尚未下载"
            
            print(f"使用交叉编码器模型路径: {cross_encoder_model_path}")
            reranker = LocalCrossEncoderReranker(embeddings=embeddings, model_path=cross_encoder_model_path)
            
            # 创建RAG对象
            print("创建RAG对象")
            rag = RAG(llm=llm, embeddings=embeddings, vectorstore=vectorstore, reranker=reranker)
            
            # 执行查询（确保传递完整的查询）
            print(f"向RAG系统传递完整查询: {query}")
            result = rag.query(query)

            print(f"\n📝 RAG返回文档数量: {len(result['documents'])}")
            
            # 将检索到的文档内容组合成字符串返回
            if result['documents']:
                documents_content = "\n\n".join([
                    f"文档 {i+1}:\n{doc.page_content}" 
                    for i, doc in enumerate(result['documents'])
                ])
                print("=== RAG工具执行完成，找到相关文档 ===")
                return f"检索到以下相关文档:\n\n{documents_content}"
            else:
                # 明确返回未找到相关信息的状态
                print("=== RAG工具执行完成，未找到相关文档 ===")
                return "error: 文档中未找到相关信息。此工具主要适用于已有文档中的技术内容，对于时效性较强的问题可能无法提供最新信息，建议结合网络搜索工具使用。"
            
        except Exception as e:
            print(f"RAG工具执行出错: {e}")
            traceback.print_exc()
            return f"error: RAG工具执行出错: {str(e)}"