import asyncio
from src.readers.pdf import PDFReader
from src.readers.web import WebReader
import logging

if __name__ == "__main__":
    """
    脚本主入口，演示 PDFReader 的用法。
    Main script entry, demonstrates usage of PDFReader.
    """
    pdf_obj = PDFReader()
    logging.info("启动PDF处理示例...")
    pdf_obj.main("Securing-Agentic-Applications-Guide-1.0.pdf", save_data_flag=True)

    #web_obj = WebReader()
    #logging.info("启动Web处理示例...")
    #url = "https://sonraisecurity.com/blog/aws-agentcore-privilege-escalation-bedrock-scp-fix"
    #asyncio.run(web_obj.main(url))
