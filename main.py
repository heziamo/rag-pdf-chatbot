import os
from dotenv import load_dotenv
from src.ingest import process_and_store_pdf
from src.chat import get_answer_from_milvus

# 加载 .env 文件中的环境变量 (如 API Key)
load_dotenv()


def main():
    # 1. 指定你的 PDF 文件路径
    pdf_path = "./data/sample.pdf"

    if not os.path.exists(pdf_path):
        print(f"找不到文件: {pdf_path}。请先在 data 目录下放入一个 PDF 文件。")
        return

    # 2. 将 PDF 处理并存入 Milvus 数据库
    # 注意：在实际项目中，入库通常只需执行一次。这里为了演示流程写在一起。
    process_and_store_pdf(file_path=pdf_path)

    # 3. 开启问答循环
    print("\n" + "=" * 50)
    print("RAG 助手已就绪！可以开始提问了 (输入 'quit' 退出)")
    print("=" * 50)

    while True:
        user_input = input("\n你的问题: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("再见！")
            break

        if not user_input.strip():
            continue

        # 检索并回答
        answer, context = get_answer_from_milvus(user_input)

        print("\n🤖 助手回答:")
        print(answer)

        # 面试加分：展示你确实从文档中取到了内容
        print("\n[参考的文档片段片段 (Snippets)]:")
        for i, doc in enumerate(context):
            # 打印每个片段的前 50 个字符
            print(f"- 片段 {i + 1}: {doc.page_content[:50]}...")


if __name__ == "__main__":
    main()