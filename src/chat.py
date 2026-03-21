from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
#LangChain 版本更新导致的模块路径变更问题（从 v0.1 到 v0.3 的迁移）
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def get_answer_from_milvus(question: str, collection_name: str = "rag_collection"):
    """
    从 Milvus 检索相关文档，并生成回答。
    """
    # 1. 连接到已有的本地 Milvus 数据库
    URI = "./milvus_demo.db"
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_db = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI},
        collection_name=collection_name
    )

    # 2. 设置检索器 (每次找最相关的 3 个片段)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # 3. 设置大模型和 Prompt
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    system_prompt = (
        "你是一个有用的企业知识库助手。请使用以下检索到的上下文来回答用户的问题。"
        "如果你不知道答案，请直接说不知道，不要编造内容。\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 4. 构建并运行 RAG 链 - 使用LCEL风格的现代实现
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"\n正在思考问题: {question} ...")
    answer = rag_chain.invoke(question)

    # 获取上下文用于返回
    context = retriever.invoke(question)
    
    return answer, context
