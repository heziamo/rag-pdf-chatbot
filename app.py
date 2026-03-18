import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="RAG PDF 问答机器人", page_icon="📚", layout="wide")

st.title("📚 RAG PDF 问答机器人")
st.markdown("**技术栈**：LangChain + Ollama (qwen3) + Chroma | 纯本地运行 | 面试展示项目")


# ====================== 加载向量数据库 ======================
@st.cache_resource
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    return Chroma(persist_directory="chroma_db", embedding_function=embeddings)


vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})  # 增加到6个片段

# ====================== LLM ======================
llm = Ollama(model="qwen3:8b", temperature=0.6)  # ← 请确认你的模型名，如果是 qwen2.5:7b-instruct 请修改

# ====================== 优化后的 Prompt（重点） ======================
template = """你是一个清晰、专业、善于讲解的AI助手。
请基于下面提供的上下文，用**自然流畅的中文**回答用户的问题。
回答时要条理清晰、重点突出，可以适当举例或对比说明。
如果上下文无法回答，请直接说“根据当前文档我无法确定答案”。

上下文：
{context}

问题：{question}

回答："""

prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join([f"片段{i + 1}: {doc.page_content}" for i, doc in enumerate(docs)])


rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

# ====================== 界面 ======================
with st.sidebar:
    st.header("项目信息")
    st.write("已加载 15 个 PDF，共 93 个文本片段")
    if st.button("🔄 重新构建向量数据库"):
        st.info("请在终端运行：python ingest.py")

question = st.text_input("💬 请输入你的问题：",
                         placeholder="例如：什么是注意力机制？Transformer有哪些创新点？")

if st.button("🚀 获取答案", type="primary"):
    if question.strip():
        with st.spinner("正在检索文档并思考..."):
            answer = rag_chain.invoke(question)

        st.markdown("### ✅ 回答：")
        st.markdown(answer)

        with st.expander("🔍 查看本次检索到的文档片段（面试讲解用）"):
            docs = retriever.invoke(question)
            for i, doc in enumerate(docs, 1):
                st.write(f"**片段 {i}**")
                st.caption(doc.page_content[:500] + "...")
                st.divider()
    else:
        st.warning("请输入问题")

st.caption("💡 使用建议：把 PDF 放入 data 文件夹 → 运行 ingest.py → 提问")