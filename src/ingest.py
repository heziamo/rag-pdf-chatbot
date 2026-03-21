import os
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus


def process_and_store_pdf(file_path: str, collection_name: str = "rag_collection"):
    """
    读取 PDF 文件，进行语义分块，并存入本地 Milvus 数据库。
    """
    print(f"正在使用 Unstructured 解析文件: {file_path} ...")
    # 1. 使用 Unstructured 加载 PDF
    # strategy="fast" 速度快，适合纯文本；如果有复杂图片/表格，可改为 "hi_res"
    loader = UnstructuredLoader(
        file_path=file_path,
        strategy="fast",
        mode="elements"
    )
    docs = loader.load()

    # 2. 文本分块 (Chunking)
    print("正在进行文本分块...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)
    print(f"共生成 {len(chunks)} 个文本块。")

    # 3. 向量化并存入 Milvus
    print("正在生成向量并存入 Milvus Lite...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 使用本地文件作为 Milvus 数据库 (Milvus Lite)，无需安装 Docker
    URI = "./milvus_demo.db"

    vector_db = Milvus.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection_args={"uri": URI},
        collection_name=collection_name,
        drop_old=True  # 每次运行清空旧数据，方便测试；生产环境请设为 False
    )

    print("数据入库完成！")
    return vector_db