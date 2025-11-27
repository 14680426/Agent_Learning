#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import hashlib
import datetime
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from modelscope.hub.snapshot_download import snapshot_download
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import sys

# 获取当前文件所在的目录(src/RAG/tools)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取src目录
src_dir = os.path.dirname(os.path.dirname(current_dir))
# 将src目录添加到系统路径中
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from dotenv import load_dotenv
# 加载项目根目录中的.env文件
project_root = os.path.dirname(src_dir)
load_dotenv(dotenv_path=os.path.join(project_root, '.env'))


# 创建本地模型目录
local_models_dir = "Models"
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
embeddings = HuggingFaceEmbeddings(
    model_name=local_model_path,
    model_kwargs={"device": "cuda"},               # 使用 GPU 加速
    encode_kwargs={"normalize_embeddings": True}   # 归一化便于计算余弦相似度
)

def clean_text(text: str) -> str:
    """
    优化的文本清理函数，更好地处理中英文混合文本
    移除无法编码为 UTF-8 的非法字符，并清理首尾空白
    """
    # 移除无法编码为 UTF-8 的非法字符
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    # 定义中文字符的正则表达式
    chinese_char = r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]'
    
    # 定义各种空白字符（包括全角空格、不间断空格等）
    any_whitespace = r'[\s\u00A0\u2000-\u200F\u2028-\u202F\u3000]+'
    
    # 删除中文字符之间的空白字符
    pattern = f'({chinese_char}){any_whitespace}(?={chinese_char})'
    text = re.sub(pattern, r'\1', text)
    
    # 处理换行符，将单个换行符替换为空格
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # 将多个连续换行符合并为两个换行符（表示段落分隔）
    text = re.sub(r'\n{2,}', '\n\n', text)
    
    # 合并多个空格为单个空格
    text = re.sub(r' +', ' ', text)
    
    # 处理特殊字符
    # 删除零宽字符
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    
    # 清理首尾空白
    return text.strip()

def get_text_splitter(split_type: str = "recursive", embeddings=None):
    """
    获取文本分割器
    
    Args:
        split_type: 分割类型，"recursive" 或 "semantic"
        embeddings: 嵌入模型，仅在使用语义分割时需要
        
    Returns:
        TextSplitter: 文档分割器实例
    """
    # 导入我们新创建的分块方法
    from RAG.tools.splits import RecursiveTextSplitter, SemanticTextSplitter
    
    if split_type == "semantic":
        return SemanticTextSplitter(embeddings, similarity_threshold=0.75)
    else:
        return RecursiveTextSplitter(chunk_size=300, chunk_overlap=30)

def load_pdfs_from_directory(directory_path: str) -> List:
    """
    从指定目录加载所有PDF文件
    
    Args:
        directory_path: PDF文件目录路径
        
    Returns:
        List: 文档列表
    """
    documents = []
    # 检查路径是文件还是目录
    if os.path.isfile(directory_path) and directory_path.lower().endswith('.pdf'):
        # 如果是单个PDF文件
        loader = PyPDFLoader(directory_path)
        documents.extend(loader.load())
    elif os.path.isdir(directory_path):
        # 如果是目录，加载目录中所有PDF文件
        for file_name in os.listdir(directory_path):
            if file_name.lower().endswith('.pdf'):
                file_path = os.path.join(directory_path, file_name)
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
    else:
        print(f"路径 '{directory_path}' 既不是PDF文件也不是目录")
        
    return documents

def compute_document_hash(document) -> str:
    """
    计算文档内容的哈希值，用于去重
    
    Args:
        document: 文档对象
        
    Returns:
        str: 文档哈希值
    """
    content = document.page_content
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def add_documents_to_chroma(documents: List, 
                           persist_directory: str = "./chroma_db",
                           collection_name: str = "local_pdf_chunks",
                           split_type: str = "recursive") -> None:
    """
    将文档添加到Chroma向量数据库
    
    Args:
        documents: 文档列表
        persist_directory: 持久化目录
        collection_name: 集合名称
        split_type: 分块类型 ("recursive" 或 "semantic")
    """
    # 获取文本分割器
    if split_type == "semantic":
        text_splitter = get_text_splitter(split_type, embeddings)
    else:
        text_splitter = get_text_splitter(split_type)
    
    # 分割文档
    split_documents = text_splitter.split_documents(documents)
    
    # 增强元数据并清洗文本
    for i, doc in enumerate(split_documents):
        # 确保doc是文档对象而不是字符串
        if isinstance(doc, str):
            print(f"警告: 发现字符串类型的文档块，跳过处理: {doc[:50]}...")
            continue
            
        # 确保有元数据字典
        if doc.metadata is None:
            doc.metadata = {}
            
        # 添加分块相关信息
        doc.metadata["chunk_index"] = i
        doc.metadata["total_chunks"] = len(split_documents)
        
        # 添加处理时间戳
        doc.metadata["processed_at"] = datetime.datetime.now().isoformat()
        
        # 清洗文本内容
        doc.page_content = clean_text(doc.page_content)
        
        # 为文档对象添加ID属性（Chroma需要）
        if not hasattr(doc, 'id'):
            # 使用源文件信息和索引生成唯一ID
            source = doc.metadata.get("source", "unknown")
            doc.id = f"{source}_chunk_{i}"
    
    # 创建或加载Chroma数据库
    if os.path.exists(persist_directory):
        # 加载现有数据库
        db = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        
        # 获取现有文档的IDs和元数据
        existing_docs = db.get(include=["documents", "metadatas"])
        existing_contents = existing_docs["documents"]
        
        # 过滤掉已存在的文档
        new_documents = []
        for i, doc in enumerate(split_documents):
            # 确保doc是文档对象而不是字符串
            if isinstance(doc, str):
                continue
                
            if doc.page_content not in existing_contents:
                new_documents.append(doc)
                
        if new_documents:
            db.add_documents(new_documents)
            print(f"添加了 {len(new_documents)} 个新文档块到数据库")
        else:
            print("没有新的文档块需要添加")
    else:
        # 过滤掉字符串类型的文档
        filtered_documents = [doc for doc in split_documents if not isinstance(doc, str)]
        
        # 创建新的数据库
        db = Chroma.from_documents(
            filtered_documents,
            embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        print(f"创建了新的Chroma数据库，包含 {len(filtered_documents)} 个文档块")

def show_database_info(persist_directory: str = "./RAG/tools/chroma_db",
                      collection_name: str = "local_pdf_chunks"):
    """
    展示数据库结构和部分文本块信息
    
    Args:
        persist_directory: 持久化目录
        collection_name: 集合名称
    """
    if not os.path.exists(persist_directory):
        print(f"数据库目录 {persist_directory} 不存在")
        return
    
    # 加载数据库
    db = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    # 获取数据库信息
    docs = db.get(include=["documents", "metadatas"])
    
    print(f"\n=== 数据库信息 ===")
    print(f"集合名称: {collection_name}")
    print(f"存储路径: {persist_directory}")
    print(f"文档总数: {len(docs['ids'])}")
    
    if len(docs['ids']) > 0:
        print(f"\n=== 文档示例 (显示前3个) ===")
        for i in range(min(3, len(docs['ids']))):
            print(f"\n--- 文档 {i+1} ---")
            print(f"ID: {docs['ids'][i]}")
            print(f"内容预览: {docs['documents'][i][:200]}...")
            print(f"元数据: {docs['metadatas'][i]}")
    else:
        print("\n数据库中没有文档")

def show_all_collections(persist_directory: str = "./RAG/tools/chroma_db"):
    """
    显示所有集合名称
    
    Args:
        persist_directory: 持久化目录
    """
    if not os.path.exists(persist_directory):
        print(f"数据库目录 {persist_directory} 不存在")
        return
    
    # 加载数据库
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    # 获取所有集合
    try:
        # 注意：Chroma的新版本可能没有list_collections方法
        collections = db._client.list_collections()
        print(f"\n=== 所有集合 ===")
        for collection in collections:
            print(f"- {collection.name}")
    except AttributeError:
        print("当前Chroma版本不支持列出所有集合")

def clear_collection(persist_directory: str = "./chroma_db",
                    collection_name: str = "local_pdf_chunks"):
    """
    清空指定的Chroma集合（保留集合结构）
    
    Args:
        persist_directory: 持久化目录
        collection_name: 集合名称
    """
    if not os.path.exists(persist_directory):
        print(f"数据库目录 {persist_directory} 不存在")
        return False
    
    try:
        # 加载数据库
        db = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        
        # 获取当前文档数量
        docs = db.get(include=["documents"])
        doc_count = len(docs["ids"])
        
        if doc_count > 0:
            # 删除所有文档
            db.delete(ids=docs["ids"])
            print(f"已从集合 {collection_name} 中清空 {doc_count} 个文档")
        else:
            print(f"集合 {collection_name} 中没有文档需要清空")
        
        return True
        
    except Exception as e:
        print(f"清空集合时出错: {e}")
        return False

def delete_collection(persist_directory: str = "./chroma_db",
                     collection_name: str = "local_pdf_chunks"):
    """
    删除指定的Chroma集合（包括集合结构和所有数据）
    
    Args:
        persist_directory: 持久化目录
        collection_name: 集合名称
    """
    if not os.path.exists(persist_directory):
        print(f"数据库目录 {persist_directory} 不存在")
        return False
    
    try:
        # 加载数据库
        db = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        
        # 删除集合
        db._client.delete_collection(collection_name)
        print(f"已删除集合: {collection_name}")
        
        return True
        
    except Exception as e:
        print(f"删除集合时出错: {e}")
        return False

def process_pdfs(pdf_directory: str = "RAG/Dataset/PDF",
                 chunk_size: int = 350,
                 chunk_overlap: int = 40,
                 persist_directory: str = "./RAG/tools/chroma_db",
                 collection_name: str = "local_pdf_chunks",
                 split_type: str = "recursive"):
    """
    处理PDF文件并将其分块存储到Chroma向量数据库
    
    Args:
        pdf_directory: PDF文件或目录路径
        chunk_size: 文本分块大小
        chunk_overlap: 文本重叠大小
        collection_name: Chroma集合名称
    """
    # 检查路径是否存在
    if not os.path.exists(pdf_directory):
        print(f"路径 '{pdf_directory}' 不存在")
        return
    
    print(f"开始处理PDF: {pdf_directory}")
    
    # 加载PDF文档
    documents = load_pdfs_from_directory(pdf_directory)
    print(f"加载了 {len(documents)} 个PDF文档")
    
    if not documents:
        print("未找到任何PDF文档")
        return
    
    # 添加文档到Chroma数据库
    add_documents_to_chroma(documents, persist_directory, collection_name, split_type)
    
    print("PDF处理完成!")

if __name__ == "__main__":
    # 默认使用递归分块处理PDF
    process_pdfs(pdf_directory="RAG/Dataset/PDF/基于视-触觉融合感知的机器人抓取滑动检测与力控研究_闫腾.pdf", 
                 chunk_size=350, 
                 chunk_overlap=40, 
                 split_type="semantic"
                 )

    # show_all_collections()

    # show_database_info()
    
    # clear_collection("RAG/tools/chroma_db", "local_pdf_chunks")