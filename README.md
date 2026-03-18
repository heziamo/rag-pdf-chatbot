# 📚 RAG PDF 问答机器人

一个**纯本地**运行的 RAG（Retrieval-Augmented Generation）项目，基于经典论文《Attention Is All You Need》实现 PDF 智能问答。

适合作为**大模型/LLM/RAG方向实习面试**的项目作品。

![界面预览](screenshot.png)   <!-- 后面你可以自己截图放进去 -->

## ✨ 项目亮点

- 完全本地化运行（无需 OpenAI API Key）
- 使用 Ollama + Qwen3 + LangChain 最新 LCEL 语法
- 支持任意 PDF 文档，只需放入 `data/` 文件夹即可
- 基于 Chroma 向量数据库，检索效率高
- Streamlit 美观前端，交互友好
- 代码结构清晰，适合面试讲解

## 🛠 技术栈

- **LLM**：Ollama (qwen:8b / qwen2.5:7b-instruct)
- **框架**：LangChain + LangChain-Community
- **向量数据库**：Chroma
- **嵌入模型**：sentence-transformers/all-MiniLM-L6-v2
- **前端**：Streamlit
- **文档加载**：PyPDF

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/你的用户名/rag-pdf-chatbot.git
cd rag-pdf-chatbot
2. 创建虚拟环境并安装依赖
Bashpython -m venv venv
venv\Scripts\activate     # Windows
pip install -r requirements.txt
3. 启动 Ollama 服务
Bashollama serve
# 新终端
ollama pull qwen:8b        # 或你使用的模型
4. 放入 PDF 并构建向量库
Bash# 把 attention.pdf 等论文放入 data/ 文件夹
python ingest.py
5. 启动 Web 界面
Bashstreamlit run app.py
📁 项目结构
textrag-pdf-chatbot/
├── data/                 # 放入你的PDF文件
├── chroma_db/            # 向量数据库（自动生成）
├── app.py                # Streamlit 前端主程序
├── ingest.py             # 构建向量数据库
├── README.md
├── requirements.txt
└── .gitignore
🎯 面试可讲解的知识点

RAG 的工作原理（Retrieval + Generation）
Chunk Size 与 Overlap 的权衡
为什么选择本地嵌入模型而非 OpenAI embeddings
LangChain LCEL 链式调用写法
自注意力机制（结合论文内容）
如何解决幻觉问题（Hallucination）

后续可扩展方向

支持多文件对话 + 历史记录
添加网页URL加载
使用更强的本地模型（qwen2.5:14b 等）
部署到 Docker