import os
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# 【全新导入】引入 LCEL 的核心组件，彻底抛弃旧版 chains
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def get_answer_from_milvus(question: str, collection_name: str = "rag_pdf_collection"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 连接 Docker 里的 Milvus
    MILVUS_URI = "http://localhost:19530"

    vector_db = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": MILVUS_URI},
        collection_name=collection_name
    )

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # 2. 接入 DeepSeek 大模型
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        raise ValueError("未找到 DEEPSEEK_API_KEY，请检查 .env 文件。")

    llm = ChatOpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        temperature=0,
        max_tokens=2048
    )

    # 3. 构建提示词模板
    system_prompt = (
        "你是一个有用的企业知识库助手。请使用以下检索到的上下文来回答用户的问题。"
        "如果你不知道答案，请直接说不知道，不要编造内容。\n\n"
        "上下文片段：\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 4. 辅助函数：将检索到的文档列表合并成一个长字符串
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 5. 【核心高光时刻】使用 LCEL 构建现代 RAG 链
    # 逻辑：组装数据字典 -> 填入 Prompt -> 交给大模型 -> 输出纯文本
    rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    # 6. 执行调用
    # 为了在前端展示检索到的参考片段，我们手动抓取一次 docs
    docs = retriever.invoke(question)
    # 调用 LCEL 链生成最终回答
    answer = rag_chain.invoke(question)

    return answer, docs