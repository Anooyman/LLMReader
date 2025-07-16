import base64
import json
import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from tqdm import tqdm
import fitz  # 导入pymupdf库，它在导入时别名为fitz

from llm import LLMBase
from config import SYSTEM_PROMPT_CONFIG, PDFReaderRole, PDF_IMAGE_PATH, JSON_DATA_PATH, PDF_PATH, VECTOR_DB_PATH
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

from utility import group_data_by_sections_with_titles, extract_data_from_LLM_res

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class PDFReader(LLMBase):
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
        self.json_data_path = JSON_DATA_PATH
        self.pdf_path = PDF_PATH
        self.vector_db_path = VECTOR_DB_PATH
        self.agenda_dict = {}
        self.pdf_raw_data = None
        self.vector_db = None
        self.chunk_count = 30  # 每个分块的大小

        for path in [self.pdf_image_path, self.json_data_path]:
            if not os.path.exists(path):
                os.makedirs(path)
                logger.info(f"Folder {path} created")
            else:
                logger.debug(f"Folder {path} already exists")

    def extract_page_num(self, path: str) -> Optional[str]:
        """
        从图片路径中提取页码数字。
        Extract page number from image file path.
        Args:
            path (str): 图片文件路径。
        Returns:
            Optional[str]: 提取到的页码数字，未找到则为 None。
        """
        file_name = os.path.basename(path)
        pattern = r'\d+'
        match = re.search(pattern, file_name)

        if match:
            number = match.group(0)
            logger.debug(f"Extracted page number {number} from {file_name}")
            return number
        else:
            logger.warning(f"未找到数字 in {file_name}。")
            return None

    def pdf_to_images(self, pdf_path: str, output_folder: str) -> None:
        """
        将 PDF 文件每一页转换为图片并保存到指定文件夹。
        Convert each page of a PDF to an image and save to output folder.
        Args:
            pdf_path (str): PDF 文件路径。
            output_folder (str): 图片保存文件夹。
        Returns:
            None
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            logger.info(f"Output folder created: {output_folder}")
        else:
            logger.debug(f"Output folder already exists: {output_folder}")

        logger.info(f"开始将PDF转为图片: {pdf_path}")
        doc = fitz.open(pdf_path)  # 打开 PDF 文档
        for page_num in range(doc.page_count):
            page = doc[page_num]
            pix = page.get_pixmap()
            image_path = f"{output_folder}/page_{page_num + 1}.png"
            pix.save(image_path)
            logger.info(f"Saved image: {image_path}")
        logger.info(f"PDF 转图片完成，共 {doc.page_count} 页。")
        doc.close()

    def read_images_in_directory(self, directory_path: str) -> List[str]:
        """
        读取指定目录下所有支持格式的图片文件路径。
        Read all supported image files in a directory.
        Args:
            directory_path (str): 目录路径。
        Returns:
            List[str]: 图片文件路径列表。
        """
        image_files = []
        valid_image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_extension = os.path.splitext(file)[1].lower()
                if file_extension in valid_image_extensions:
                    image_path = os.path.join(root, file)
                    image_files.append(image_path)
        logger.info(f"读取到{len(image_files)}张图片 in {directory_path}")
        return image_files

    def get_pdf_name(self, file_name: str) -> str:
        """
        获取去除扩展名后的文件名。
        Get file name without extension.
        Args:
            file_name (str): 文件名。
        Returns:
            str: 去除扩展名后的文件名。
        """
        dot_index = file_name.rfind('.')
        if dot_index != -1:
            file_name_without_ext = file_name[:dot_index]
            logger.debug(f"PDF文件名去后缀: {file_name_without_ext}")
        else:
            file_name_without_ext = file_name
            logger.debug(f"PDF文件名无后缀: {file_name}")
        return file_name_without_ext

    def extract_pdf_data(self, pdf_file_path: str) -> List[Dict[str, Any]]:
        """
        将 PDF 转为图片并用 LLM 提取每页内容，结果保存为 JSON。
        Convert PDF to images and extract content from each page using LLM, save as JSON.
        Args:
            pdf_file_path (str): PDF 文件名（不含路径）。
        Returns:
            List[Dict[str, Any]]: 每页图片的内容和页码。
        """
        output_folder_path = os.path.join(self.pdf_image_path, self.get_pdf_name(pdf_file_path))
        pdf_path = os.path.join(self.pdf_path, pdf_file_path)
        logger.info(f"开始处理PDF: {pdf_path}")
        self.pdf_to_images(pdf_path, output_folder_path)
        image_paths = self.read_images_in_directory(output_folder_path)
        sorted_list = sorted(image_paths, key=lambda x: int(re.search(r'page_(\d+)\.png', x).group(1)))
        image_content_list = []
        error_pages_list = []
        for path in tqdm(sorted_list, desc="[INFO] 正在处理图片并提取内容"):
            try:
                with open(path, 'rb') as img_file:
                    encoded_image = base64.b64encode(img_file.read()).decode('ascii')
                message = [HumanMessage(
                    content=[
                        {"type": "text", "text": SYSTEM_PROMPT_CONFIG.get(PDFReaderRole.IMAGE_EXTRACT) },
                        { "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                    ],
                )]
                response = self.chat_model.invoke(message)
                image_content_list.append(
                    {
                        "data": response.content,
                        "page": self.extract_page_num(path)
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
   
    def call_llm_chain(
            self,
            role: str,
            input_prompt: str,
            session_id: str,
            output_parser=StrOutputParser(),
            system_format_dict: dict={}
        ) -> Any:
        """
        通用 LLM 调用方法，按不同角色和 session_id 调用链。
        General LLM call method, invokes chain with different roles and session IDs.
        Args:
            role (str): PDFReaderRole 枚举值。
            input_prompt (str): 输入提示。
            session_id (str): 会话 ID。
            output_parser: 输出解析器。
        Returns:
            Any: LLM 响应对象。
        """
        logger.info(f"调用LLM: role={role}, session_id={session_id}")

        system_prompt = SYSTEM_PROMPT_CONFIG.get(role)

        if system_format_dict:
            system_prompt = system_prompt.format(**system_format_dict)

        chain = self.build_chain(
            client=self.chat_model,
            system_prompt=system_prompt,
            output_parser=output_parser
        )
        response = chain.invoke(
            {"input_prompt": input_prompt},
            config={"configurable": {"session_id": session_id}}
        )
        logger.info(f"LLM调用完成: role={role}, session_id={session_id}")
        return response

    def get_basic_info(self, pdf_raw_data) -> Dict[str, Any]:
        """
        获取 PDF 的基本信息摘要。
        Get basic summary information of the PDF.
        Args:
            pdf_raw_data: PDF 原始数据。
        Returns:
            Dict[str, Any]: 基本信息摘要。
        """
        input_prompt = f"这里是文章的完整内容: {pdf_raw_data}"
        response = self.call_llm_chain(PDFReaderRole.PDF_COMMON, input_prompt, "common")
        logger.info("已获取 PDF 基本信息摘要。")
        return extract_data_from_LLM_res(response)

    def get_pdf_agenda(self, pdf_raw_data) -> List[Dict[str, Any]]:
        """
        获取 PDF 的目录结构或议程。
        Get the agenda or table of contents of the PDF.
        Args:
            pdf_raw_data: PDF 原始数据。
        Returns:
            List[Dict[str, Any]]: 目录结构列表。
        """
        input_prompt = f"这里是文章的完整内容: {pdf_raw_data}"
        response = self.call_llm_chain(PDFReaderRole.PDF_AGENDA, input_prompt, "agenda")
        logger.info(f"PDF Directory Structure: {response}")
        return extract_data_from_LLM_res(response)

    def summary_content(self, title: str, content: Any) -> Any:
        """
        针对某一章节内容进行总结。
        Summarize the content of a specific section.
        Args:
            title (str): 章节标题。
            content (Any): 章节内容。
        Returns:
            Any: 总结内容。
        """
        input_prompt = f"请总结{title}的内容，上下文如下：{content}"
        response = self.call_llm_chain(PDFReaderRole.PDF_SUMMARY, input_prompt, "summary")
        logger.info(f"章节 {title} 总结完成。")
        return response
 
    def get_answer(self, context_data: Any, query: str, common_data: Any = "") -> Any:
        """
        综合所有摘要和基本信息，生成最终详细回答。
        Generate a detailed answer based on all summaries and basic info.
        Args:
            context_data (Any): 上下文数据。
            query (str): 问题。
            common_data (Any, optional): 背景信息。
        Returns:
            Any: 最终回答。
        """
        if common_data:
            input_prompt = f"请结合背景信息回答客户问题\nBackground info:{common_data}\n{context_data}\nQuestion:{query}"
        else:
            input_prompt = f"请结合背景信息回答客户问题\nBackground info:{context_data}\nQuestion:{query}"
        logger.info("开始生成最终详细回答...")
        response = self.call_llm_chain(PDFReaderRole.PDF_ANSWER, input_prompt, "answer")
        logger.info("最终详细回答生成完毕。")
        return response

    def build_vector_db(self, vector_db_path, content_docs: List[Document]) -> FAISS:
        """
        构建向量数据库。
        Build a vector database from document list.
        Args:
            content_docs (List[Document]): 文档列表。
        Returns:
            FAISS: 构建的向量数据库对象。
        """
        logger.info(f"开始构建向量数据库，文档数: {len(content_docs)}")
        vector_db = FAISS.from_documents(content_docs, self.embbeding_model)
        vector_db.save_local(vector_db_path)
        logger.info(f"向量数据库已保存到: {vector_db_path}")
        return vector_db 
 
    def load_vector_db(self, vector_db_path) -> FAISS:
        """
        加载本地向量数据库。
        Load local vector database.
        Args:
            vector_db_path: 向量数据库路径。
        Returns:
            FAISS: 加载的向量数据库对象。
        """
        logger.info(f"加载本地向量数据库: {vector_db_path}")
        return FAISS.load_local(vector_db_path, self.embbeding_model, allow_dangerous_deserialization=True)

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

    def process_pdf(self, pdf_file_path: str, query: str = "请按照文章本身的叙事结构详细整理这篇文章的主要内容") -> Any:
        """
        主流程：读取 PDF 数据，提取结构，分章节总结，最终生成详细回答。
        Main pipeline: read PDF data, extract structure, summarize by section, and generate final answer.
        Args:
            pdf_file_path (str): PDF 文件名（不含路径）。
            query (str, optional): 问题。
        Returns:
            Any: 最终回答。
        """
        vector_db_path = os.path.join(self.vector_db_path, f"{pdf_file_path}_data_index")
        logger.info(f"开始处理PDF主流程: {pdf_file_path}")
        try:
            with open(os.path.join(self.json_data_path, f"{pdf_file_path}.json"), 'r', encoding='utf-8') as f:
                self.pdf_raw_data = json.load(f)
            logger.info(f"成功读取本地JSON数据: {os.path.join(self.json_data_path, f'{pdf_file_path}.json')}")
        except Exception as e:
            logger.warning(f"读取本地JSON失败，将重新提取: {e}")
            self.pdf_raw_data = self.extract_pdf_data(pdf_file_path)

        # 初始化存储变量
        common_data_dict = {}
        vector_db_content_docs = []

        if os.path.exists(vector_db_path):
            self.vector_db = self.load_vector_db(vector_db_path)
            all_db_res = self.vector_db.similarity_search_with_score("", k=99999999)
            # 对检索回来的章节内容按 pages 的第一个页码进行排序，保证章节顺序与原文一致
            all_db_res_sorted = sorted(
                all_db_res,
                key=lambda x: x[0].metadata.get("pages", [float('inf')])[0] if x[0].metadata.get("pages") else float('inf')
            )
            total_summary = {}
            logger.info(f"检索回 {len(all_db_res)} 章节内容（已按页码排序）")
            for db_res in all_db_res_sorted:
                metadata = db_res[0].metadata  # db_res[0] 是 Document
                title = db_res[0].page_content  # 章节标题
                summary = metadata.get("summary")  # 章节摘要
                pages = metadata.get("pages")     # 章节页码列表
                # "PDF common info" 是全局信息，单独处理
                if title == "PDF common info":
                    common_data_dict = summary
                else:
                    total_summary[title] = summary
                    self.agenda_dict[title] = pages
            logger.info(f"当前文章目录结构如下: {self.agenda_dict}")
            # 返回值说明：
            # total_summary: 按章节顺序有序的 dict，key 为章节标题，value 为摘要
            # self.agenda_dict: 按章节顺序有序的 dict，key 为章节标题，value 为页码列表

        else:
            # 按 chunk_count 切分 pdf_raw_data，便于大文件分批处理
            chunks = self.split_pdf_raw_data()
            logger.info(f"开始分块处理 PDF，每块大小为 {self.chunk_count}，共 {len(chunks)} 块。")

            agenda_list = []
            for index, chunk in enumerate(chunks):
                logger.info(f"处理第 {index+1}/{len(chunks)} 块...")
                # 只在第一个 chunk 提取基本信息
                if index == 0:
                    logger.info(f"获取PDF基本信息（第一个chunk）...")
                    common_data_dict = self.get_basic_info(chunk)
                    logger.info(f"PDF 基本信息: {common_data_dict}")
                logger.info(f"获取PDF目录结构（第 {index+1} 块）...")
                agenda = self.get_pdf_agenda(chunk)
                logger.info(f"第 {index+1} 块目录结构: {agenda}")
                agenda_list.extend(agenda)

            vector_db_content_docs.append(
                Document(
                    page_content="PDF common info",
                    metadata={
                        "pages": [0],
                        "raw_data": common_data_dict,
                        "summary": common_data_dict,
                    }
                )   
            )
            # 合并所有 chunk 的目录结构后，按章节分组
            logger.info(f"合并所有 chunk 的目录结构，准备分组...")
            agenda_data_list, self.agenda_dict = group_data_by_sections_with_titles(agenda_list, self.pdf_raw_data)
            logger.info(f"章节分组完成，章节数: {len(agenda_data_list)}")
            logger.info(f"最终章节信息如下: {self.agenda_dict}")
            logger.info(f"开始分章节总结...")
            total_summary = {}
            for agenda_data in agenda_data_list:
                title = agenda_data.get("title")
                data = agenda_data.get("data")
                logger.info(f"正在总结章节: {title}")
                summary = self.summary_content(title, data)
                total_summary[title] = summary
                vector_db_content_docs.append(
                    Document(
                        page_content=agenda_data.get("title"),
                        metadata={
                            "pages": agenda_data.get("pages"),
                            "raw_data": agenda_data.get("data"),
                            "summary": summary,
                        }
                    )
                )
            logger.info(f"所有章节摘要已完成，正在构建向量数据库...")
            self.vector_db = self.build_vector_db(vector_db_path, vector_db_content_docs)
            logger.info(f"向量数据库构建完成。")

        logger.info(f"生成最终详细回答...")
        final_answer = self.get_answer(total_summary, query, common_data_dict)
        logger.info(f"PDF处理流程结束。")
        return final_answer

    def retrieval_data(self, response, last_retrieval_data):
        """
        检索用户请求相关的章节内容。
        Retrieve relevant section content for user queries.
        Args:
            response: LLM 返回的章节标题列表。
            last_retrieval_data: 上一次检索的数据缓存。
        Returns:
            context_data: 当前检索到的上下文内容。
            retrieval_data: 检索数据缓存。
        """
        retrieval_data = {}
        title_list = response.get("title", [])
        context_data = []
        for title in title_list:
            if title not in last_retrieval_data.keys():
                logger.info(f"通过向量数据库检索章节: '{title}' ...")
                doc_res = self.vector_db.similarity_search_with_score(title, k=1)
                if doc_res:
                    title_context = str(doc_res[0][0].metadata.get("raw_data"))
                    logger.info(f"检索到章节 '{title}' 内容")
                else:
                    logger.warning(f"章节 '{title}' 未在向量数据库中检索到相关内容。")
                    title_context = ''
                retrieval_data[title] = title_context

            else:
                logger.info(f"'{title}'已经被检索过，跳过当前章节检索")
                title_context = last_retrieval_data.get(title)

            #if not self.is_content_in_history(title_context, "answer"):
            #    context_data.append(title_context)
            context_data.append(title_context)

        logger.info(f"检索数据完成，涉及章节数: {len(title_list)}")
        return context_data, retrieval_data

    def chat(self, input_prompt: str, last_retrieval_data: dict) -> Any:
        """
        针对用户输入进行对话。
        Interactive chat for user input.
        Args:
            input_prompt (str): 用户输入。
            last_retrieval_data (dict): 上一次检索的数据缓存。
        Returns:
            Any: 回答内容。
        """
        response = self.call_llm_chain(
            PDFReaderRole.PDF_CHAT,
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
        retrieval_data = {}
        try:
            response = extract_data_from_LLM_res(response)
            context_data, retrieval_data = self.retrieval_data(response, last_retrieval_data)
        except Exception as e:
            logger.warning(f"LLM 响应解析失败，直接返回原始响应: {e}")
            context_data = response

        answer = self.get_answer(context_data, input_prompt)
        logger.info(f"对话回答生成完毕。")
        return answer, retrieval_data

    def main(self, pdf_file_path: str) -> None:
        """
        主入口，启动 PDF 处理和对话。
        Main entry point, starts PDF processing and interactive chat.
        Args:
            pdf_file_path (str): PDF 文件名。
        Returns:
            None
        """
        chat_count = 0
        last_retrieval_dict = {}
        logger.info(f"启动主流程，处理 PDF 文件: {pdf_file_path}")
        while True:
            if chat_count == 0:
                answer = self.process_pdf(pdf_file_path)
                print(f"ChatBot: {answer}")

            user_input = input("You: ")

            if user_input.lower() in ["退出", "再见", "bye", "exit", "quit"]:
                print("Chatbot: 再见！期待下次与您对话。")
                logger.info("用户主动退出对话。")
                break

            answer, last_retrieval_dict = self.chat(user_input, last_retrieval_dict)
            self.add_message_to_history(session_id="chat", message=AIMessage(answer))
            print(f"User: {user_input}")
            print(f"ChatBot: {answer}")
            print("======"*10)
            chat_count += 1