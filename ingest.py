import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ====================== 配置 ======================
DATA_DIR = "data"
DB_DIR = "chroma_db"

# 使用国内镜像加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("🚀 RAG PDF 问答系统 - 向量数据库构建工具")
print("=" * 60)

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"✅ 已创建 {DATA_DIR}/ 文件夹，请放入PDF文件后重新运行")
    exit()

# 1. 加载PDF
loader = PyPDFDirectoryLoader(DATA_DIR)
docs = loader.load()

print(f"📄 成功加载 {len(docs)} 个PDF文档")

# 2. 文本切分（优化参数，更适合论文）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
)

chunks = text_splitter.split_documents(docs)
print(f"✂️  文档切分成 {len(chunks)} 个文本片段")

# 3. 构建向量数据库
print("🔍 正在生成嵌入向量并存入 Chroma...（首次较慢）")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_DIR
)

print("🎉 向量数据库构建成功！")
print(f"   项目路径: {os.getcwd()}")
print(f"   数据库位置: {DB_DIR}/")
print(f"   总片段数: {len(chunks)}")
print("\n✅ 现在可以运行：streamlit run app.py 进行测试")