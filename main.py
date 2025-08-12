import asyncio
import logging
from src.readers.pdf import PDFReader
from src.readers.web import WebReader
from src.utils.helpers import list_pdf_files

def main():
    """
    脚本主入口，演示 PDFReader 和 WebReader 的用法。
    Main script entry, demonstrates usage of PDFReader and WebReader.
    """
    # 获取可用的 PDF 文件列表
    pdf_files_list = list_pdf_files()
    # 提示用户选择 PDF 文件或输入其他内容
    input_data_str = input(f"可选择 PDF 文件名:\n{pdf_files_list}\nYou: ").strip()

    # 判断用户是否选择退出
    if input_data_str.lower() in ["退出", "再见", "bye", "exit", "quit"]:
        print("Chatbot: 再见！期待下次与您对话。")
        return

    # 判断输入是否为 PDF 文件
    if input_data_str.lower().endswith('.pdf'):
        logging.info("启动PDF处理示例...")
        pdf_obj = PDFReader()
        pdf_obj.main(input_data_str, save_data_flag=True)
    else:
        # 非 PDF 文件，使用 WebReader 处理
        logging.info("启动Web处理示例...")
        web_obj = WebReader()
        # 使用 asyncio 运行异步主流程
        asyncio.run(web_obj.main(input_data_str))

if __name__ == "__main__":
    main()
