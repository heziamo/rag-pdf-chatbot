import streamlit as st
import os
from dotenv import load_dotenv
from src.ingest import process_and_store_pdf
from src.chat import get_answer_from_milvus

# 加载环境变量 (API Key)
load_dotenv()

# 1. 页面配置 (设置标题、图标和宽屏模式)
st.set_page_config(page_title="企业级 RAG 知识库", page_icon="🤖", layout="wide")

st.title("🤖 企业级 RAG 智能助手")
st.markdown("基于 **Milvus** 向量数据库与 **Unstructured** 智能解析构建")

# 2. 初始化聊天历史记录
# Streamlit 会在每次交互时重新运行代码，所以我们需要用 session_state 来保存聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. 侧边栏：文档上传与数据入库管理
with st.sidebar:
    st.header("📄 知识库管理")
    uploaded_file = st.file_uploader("上传 PDF 文档", type=["pdf"])

    if st.button("处理并入库 (ETL)"):
        if uploaded_file is not None:
            with st.spinner("正在使用 Unstructured 解析文档并存入 Milvus..."):
                # 确保 data 目录存在
                os.makedirs("./data", exist_ok=True)

                # 将前端上传的文件保存到本地 data 目录
                temp_pdf_path = os.path.join("./data", uploaded_file.name)
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # 调用我们之前写的入库逻辑！
                try:
                    process_and_store_pdf(temp_pdf_path)
                    st.success("入库成功！现在您可以开始提问了。")
                except Exception as e:
                    st.error(f"处理失败: {e}")
        else:
            st.warning("请先点击上方按钮上传 PDF 文件。")

    st.divider()
    st.markdown("### 系统架构说明")
    st.markdown("- **解析引擎**: Unstructured")
    st.markdown("- **向量引擎**: Milvus Lite")
    st.markdown("- **大语言模型**: OpenAI deepseek-chat")

# 4. 主聊天界面：渲染历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. 处理用户输入
# st.chat_input 会在页面底部生成一个固定输入框
if prompt := st.chat_input("请输入您的问题，例如：这份文档的核心结论是什么？"):

    # 将用户问题存入历史记录并显示在界面上
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 助手开始思考并生成回答
    with st.chat_message("assistant"):
        with st.spinner("正在检索企业知识库..."):
            try:
                # 调用我们之前写的检索逻辑！
                answer, context = get_answer_from_milvus(prompt)

                # 显示生成的回答
                st.markdown(answer)

                # 面试加分项：使用折叠面板 (Expander) 优雅地展示参考文档片段
                with st.expander("👀 查看底层检索到的文档片段 (Source Context)"):
                    for i, doc in enumerate(context):
                        st.markdown(f"**片段 {i + 1}**: {doc.page_content[:200]}...")

                # 将助手的回答存入历史记录
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error("无法获取回答。请确保您已经先在左侧侧边栏上传并处理了文档。")
                st.error(f"详细错误: {e}")