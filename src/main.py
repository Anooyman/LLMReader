from pdf_reader import PDFReader
import logging

if __name__ == "__main__":
    """
    脚本主入口，演示 PDFReader 的用法。
    Main script entry, demonstrates usage of PDFReader.
    """
    pdf_obj = PDFReader()
    logging.info("启动PDF处理示例...")
    pdf_obj.main("1706.03762v7.pdf")
