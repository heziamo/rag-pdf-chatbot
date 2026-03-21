import os
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus


def process_and_store_pdf(file_path: str, collection_name: str = "rag_pdf_collection"):
    # 1. 使用本地 Unstructured hi_res 模式解析（支持表格结构化提取）
    loader = UnstructuredLoader(
        file_path=file_path,
        strategy="hi_res",
        infer_table_structure=True,  # 关键：让表格输出成 HTML
        chunking_strategy="by_title",  # 尽量让表格保持完整
        mode="elements",
        languages=["chi_sim", "eng"]
    )
    docs = loader.load()

    # 2. 过滤：只保留普通文字 + 表格，丢弃图片/图表/其他不需要的元素
    refined_docs = []
    kept_categories = {"NarrativeText", "ListItem", "Title", "Table"}  # 可以根据需要再加 "UncategorizedText" 等

    for doc in docs:
        category = doc.metadata.get("category", "")

        if category in kept_categories:
            # 特别处理表格：转成 DeepSeek 友好的 HTML 格式
            if category == "Table":
                table_html = doc.metadata.get("text_as_html")
                if table_html:
                    doc.page_content = f"【表格开始】\n{table_html}\n【表格结束】"
                else:
                    # 极少数情况没有 html，就用原始文本
                    doc.page_content = f"[表格（无 HTML 结构）]\n{doc.page_content}"

            # 普通文字、标题、列表项直接保留原样
            refined_docs.append(doc)

        # 可选：打印被丢弃的元素，便于调试
        # else:
        #     print(f"丢弃: {category} → {doc.page_content[:60]}...")

    print(f"提取到 {len(refined_docs)} 个有效元素（文字 + 表格），将进行向量化存储")

    if not refined_docs:
        raise ValueError("没有提取到任何文字或表格内容，请检查 PDF 是否可解析")

    # 3. 分块（文字和表格一起分）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,  # 稍微多一点重叠，对论文这种长句子更有帮助
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    chunks = text_splitter.split_documents(refined_docs)

    # 4. Embedding + 存 Milvus（保持原样）
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    MILVUS_URI = "http://localhost:19530"
    print(f"连接 Milvus: {MILVUS_URI} ...")

    vector_db = Milvus.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection_args={"uri": MILVUS_URI},
        collection_name=collection_name,
        drop_old=True,  # 开发阶段建议每次都重建，生产环境可以改成 False
    )

    print(f"入库完成！共 {len(chunks)} 个分块")
    return vector_db