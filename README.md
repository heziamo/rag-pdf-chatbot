# 🚀 DeepSeek-Milvus-PDF-RAG

一个基于 **DeepSeek-V3/R1** 和 **Milvus** 构建的高精度 PDF 对话机器人。支持本地 OCR 识别与复杂的表格结构化解析。

## 🌟 项目亮点
- **表格高保真**：采用 `Unstructured` 的 `hi_res` 策略，将 PDF 表格转化为 HTML 喂给 DeepSeek，保留行列关系。
- **全本地解析**：集成 Tesseract OCR 和 Poppler，支持扫描件解析，数据不流向第三方解析平台。
- **向量存储**：使用 Docker 部署的 Milvus Standalone，具备企业级检索性能。

## 🏗️ 系统架构

graph TD
    User((用户)) -->|上传PDF / 提问| UI[Streamlit 界面]
    
    subgraph "数据解析层"
        UI -->|文件| Loader[Unstructured + Tesseract]
        Loader -->|表格提取| Table[HTML 结构化表格]
        Loader -->|文本提取| Text[清洗后的文本]
    end

    subgraph "存储与检索"
        Table & Text -->|Embedding| Milvus[(Milvus 向量库)]
    end

    subgraph "大脑推理"
        UI -->|查询| Chain[LangChain 检索链]
        Milvus -->|背景知识| Chain
        Chain -->|Prompt| LLM[DeepSeek API]
        LLM -->|回答| UI
    end
##🛠️ 环境准备
1. 软件依赖 (Windows)
Tesseract OCR: 安装地址，需下载 chi_sim 语言包。

Poppler: 下载地址，解压并记录 bin 路径。

Docker Desktop: 用于运行 Milvus。

2. 启动 Milvus
Bash
# 在项目目录下运行 (确保已下载 milvus 的 docker-compose.yml)
docker-compose up -d
##🚀 快速开始
1. 安装 Python 依赖
Bash
pip install -r requirements.txt
2. 配置环境变量 (.env)
在项目根目录新建 .env 文件，填入以下内容：

Code snippet
DEEPSEEK_API_KEY=你的API密钥
MILVUS_URI=http://localhost:19530
TESSERACT_PATH=D:\tesseract\tesseract.exe
TESSDATA_PREFIX=D:\tesseract\tessdata
3. 运行应用
Bash
streamlit run app.py
📝 注意事项
中文路径优化：代码已自动处理 Windows 中文用户名导致的临时文件报错。

解析速度：由于开启了 hi_res 表格识别，解析 10 页以上的 PDF 可能需要较长时间。
