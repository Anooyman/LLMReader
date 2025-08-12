import base64
import json
import os
import re
import logging
from typing import List, Dict, Any
from tqdm import tqdm

from langchain_core.messages import HumanMessage, AIMessage

from src.readers.base import ReaderBase
from src.config.settings import (
    ReaderRole,
    SYSTEM_PROMPT_CONFIG,
    PDF_IMAGE_PATH,
    PDF_PATH,
)
from src.utils.helpers import *
from src.core.vector_db.vector_db_client import VectorDBClient

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class PDFReader(ReaderBase):
    """
    PDFReader 类用于处理 PDF 文件，包括：
    1. PDF 转图片
    2. 图片内容提取
    3. 调用 LLM 进行内容分析与总结
    4. 构建和使用向量数据库
    5. 支持交互式问答

    This class provides a full pipeline for PDF document analysis, including image conversion, content extraction, LLM-based summarization, vector DB construction, and interactive Q&A.
    """
    def __init__(self, provider: str = "azure") -> None:
        """
        初始化 PDFReader 对象，支持多 LLM provider。
        provider: 'azure'（默认）、'openai'、'ollama'。
        """
        super().__init__(provider)
        self.pdf_image_path = PDF_IMAGE_PATH
        self.pdf_path = PDF_PATH
        self.pdf_raw_data = None
        self.chunk_count = 20  # 每个分块的大小

        for path in [self.pdf_image_path, self.pdf_path]:
            makedir(path)

    def extract_pdf_data(self, pdf_file_path: str) -> List[Dict[str, Any]]:
        """
        将 PDF 转为图片并用 LLM 提取每页内容，结果保存为 JSON。
        Convert PDF to images and extract content from each page using LLM, save as JSON.
        Args:
            pdf_file_path (str): PDF 文件名（不含路径）。
        Returns:
            List[Dict[str, Any]]: 每页图片的内容和页码。
        """
        output_folder_path = os.path.join(self.pdf_image_path, pdf_file_path)
        pdf_path = os.path.join(self.pdf_path, f"{pdf_file_path}.pdf")
        logger.info(f"开始处理PDF: {pdf_path}")
        pdf_to_images(pdf_path, output_folder_path)
        image_paths = read_images_in_directory(output_folder_path)
        sorted_list = sorted(image_paths, key=lambda x: int(re.search(r'page_(\d+)\.png', x).group(1)))
        image_content_list = []
        error_pages_list = []
        for path in tqdm(sorted_list, desc="[INFO] 正在处理图片并提取内容"):
            try:
                with open(path, 'rb') as img_file:
                    encoded_image = base64.b64encode(img_file.read()).decode('ascii')
                message = [HumanMessage(
                    content=[
                        {"type": "text", "text": SYSTEM_PROMPT_CONFIG.get(ReaderRole.IMAGE_EXTRACT) },
                        { "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                    ],
                )]
                response = self.chat_model.invoke(message)
                image_content_list.append(
                    {
                        "data": response.content,
                        "page": extract_page_num(path)
                    }
                )
                logger.info(f"图片{path}内容提取成功")
            except Exception as e:
                logger.error(f"Get error for {path}: {e}")
                error_pages_list.append(path)
                continue
        with open(os.path.join(self.json_data_path, f"{pdf_file_path}.json"), 'w', encoding='utf-8') as file:
            json.dump(image_content_list, file, ensure_ascii=False )
        logger.info(f"数据已保存到本地JSON文件 {self.json_data_path}/{pdf_file_path}.json 中。")
        if len(error_pages_list) > 0:
            logger.error(f"部分图片提取内容失败{error_pages_list}.\n请用其他手段提取数据并写入 JSON 文件中.")
        return image_content_list
   
    def split_pdf_raw_data(self):
        """
        将 self.pdf_raw_data 按照 self.chunk_count 进行切分。
        Split self.pdf_raw_data into chunks of size self.chunk_count.
        Returns:
            List[List[Any]]: 切分后的数据块列表。
        """
        if not isinstance(self.pdf_raw_data, list):
            logger.error("pdf_raw_data 不是 list，无法切分。")
            return []
        chunks = [self.pdf_raw_data[i:i + self.chunk_count] for i in range(0, len(self.pdf_raw_data), self.chunk_count)]
        logger.info(f"已将 pdf_raw_data 切分为 {len(chunks)} 个块，每块最多 {self.chunk_count} 条。")
        return chunks

    def process_pdf(self, pdf_file_path: str, save_data_flag: bool=True) -> Any:
        """
        主流程：读取 PDF 数据，提取结构，分章节总结，最终生成详细回答。
        Main pipeline: read PDF data, extract structure, summarize by section, and generate final answer.
        Args:
            pdf_file_path (str): PDF 文件名（不含路径）。
        Returns:
            None
        """
        vector_db_path = os.path.join(f"{self.vector_db_path}/{pdf_file_path}_data_index")
        self.vector_db_obj = VectorDBClient(vector_db_path)
        logger.info(f"开始处理PDF主流程: {pdf_file_path}")
        try:
            with open(f"{self.json_data_path}/{pdf_file_path}.json", 'r', encoding='utf-8') as f:
                self.pdf_raw_data = json.load(f)
            logger.info(f"成功读取本地JSON数据: {self.json_data_path}/{pdf_file_path}.json")
        except Exception as e:
            logger.warning(f"读取本地JSON失败，将重新提取: {e}")
            self.pdf_raw_data = self.extract_pdf_data(pdf_file_path)

        if os.path.exists(vector_db_path):
            self.get_data_from_vector_db()
        else:
            # 按 chunk_count 切分 pdf_raw_data，便于大文件分批处理
            chunks = self.split_pdf_raw_data()
            logger.info(f"开始分块处理 PDF，每块大小为 {self.chunk_count}，共 {len(chunks)} 块。")
            self.get_data_from_json_dict(chunks, self.pdf_raw_data)

        if save_data_flag:
            self.generate_output_file(pdf_file_path, self.common_data_dict, self.raw_data_dict)

        logger.info(f"PDF处理流程结束。")

    def chat(self, input_prompt: str) -> Any:
        """
        针对用户输入进行对话。
        Interactive chat for user input.
        Args:
            input_prompt (str): 用户输入。
        Returns:
            Any: 回答内容。
        """
        response = self.call_llm_chain(
            ReaderRole.CHAT,
            input_prompt,
            "chat",
            system_format_dict={
                "agenda_dict": self.agenda_dict
            }
        )
        print("====="*10)
        print(self.agenda_dict)
        print("====="*10)
        print(response)
        try:
            extract_response = extract_data_from_LLM_res(response)
            context_data  = self.retrieval_data(extract_response)
        except Exception as e:
            logger.warning(f"LLM 响应解析失败，直接返回原始响应: {e}")
            context_data = response

        answer = self.get_answer(context_data, input_prompt)
        logger.info(f"对话回答生成完毕。")
        return answer

    def main(self, pdf_file_path: str, save_data_flag: bool=True) -> None:
        """
        主入口，启动 PDF 处理和对话。
        Main entry point, starts PDF processing and interactive chat.
        Args:
            pdf_file_path (str): PDF 文件名。
            save_data_flag (bool): 是否需要存储文件
        Returns:
            None
        """
        logger.info(f"启动主流程，处理 PDF 文件: {pdf_file_path}")

        self.process_pdf(get_pdf_name(pdf_file_path), save_data_flag)
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["退出", "再见", "bye", "exit", "quit"]:
                print("Chatbot: 再见！期待下次与您对话。")
                logger.info("用户主动退出对话。")
                break

            answer = self.chat(user_input)
            self.add_message_to_history(session_id="chat", message=AIMessage(answer))
            print(f"User: {user_input}")
            print(f"ChatBot: {answer}")
            print("======"*10)