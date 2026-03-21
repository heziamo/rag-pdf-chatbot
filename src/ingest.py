import os
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus


def process_and_store_pdf(file_path: str, collection_name: str = "rag_pdf_collection"):
    # 1. 解析 PDF
    loader = UnstructuredLoader(
        file_path=file_path,
        #切换到高精度模式
        strategy="hi_res",
        #开启表格结构提取
        infer_table_structure=True,
        #设置 chunking_strategy，确保表格作为一个整体，不被切断
        chunking_strategy="by_title",
        mode="elements",
        languages=["chi_sim","eng"]
    )
    docs = loader.load()

    # 对提取出的元素进行“清洗”，确保表格是以 HTML 形式存在的
    refined_docs = []
    for doc in docs:
        # 如果 Unstructured 识别出这是一个表格
        if doc.metadata.get("category") == "Table":
            # 获取表格的 HTML 表示形式（这是 DeepSeek 最喜欢的格式）
            table_html = doc.metadata.get("text_as_html")
            if table_html:
                doc.page_content = f"【数据表格开始】\n{table_html}\n【数据表格结束】"

        refined_docs.append(doc)

    # 2. 分块
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # 3. Embedding
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. 连接 Docker 里的 Milvus
    # 如果你的 PyCharm 在宿主机运行，Docker 映射端口通常是 19530
    MILVUS_URI = "http://localhost:19530"

    print(f"正在连接 Docker Milvus: {MILVUS_URI}...")
    vector_db = Milvus.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection_args={"uri": MILVUS_URI},
        collection_name=collection_name,
        drop_old=True
    )
    return vector_db