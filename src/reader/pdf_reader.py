import base64
import json
import os
import re
import logging
from typing import List, Dict, Any
from tqdm import tqdm
import markdown
from weasyprint import HTML

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

from src.common.llm import LLMBase
from src.common.config import (
    SYSTEM_PROMPT_CONFIG,
    PDFReaderRole,
    PDF_IMAGE_PATH,
    JSON_DATA_PATH,
    PDF_PATH,
    VECTOR_DB_PATH,
    OUTPUT_PATH
)
from src.common.utility import *

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
        self.output_path = OUTPUT_PATH
        self.agenda_dict = {}
        self.pdf_raw_data = None
        self.vector_db = None
        self.chunk_count = 30  # 每个分块的大小
        self.retrieval_dict = {}

        for path in [self.pdf_image_path, self.json_data_path, self.pdf_path, self.vector_db_path, self.output_path]:
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
        output_folder_path = os.path.join(self.pdf_image_path, get_pdf_name(pdf_file_path))
        pdf_path = os.path.join(self.pdf_path, pdf_file_path)
        logger.info(f"开始处理PDF: {pdf_path}")
        self.pdf_to_images(pdf_path, output_folder_path)
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
                        {"type": "text", "text": SYSTEM_PROMPT_CONFIG.get(PDFReaderRole.IMAGE_EXTRACT) },
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
        logger.info("开始生成回答...")
        response = self.call_llm_chain(PDFReaderRole.PDF_ANSWER, input_prompt, "answer")
        logger.info("回答生成完毕。")
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

    def save_data_to_file(self, data: str, file_name: str, file_type_list: list=["md", "pdf"]):
        """
        将数据保存为指定类型的文件

        参数:
            data: str - 需要保存的数据内容
            file_name: str - 文件名（不包含扩展名）
            file_type_list: list - 要保存的文件类型列表，默认为["md", "pdf"]
        """
        # 遍历需要保存的文件类型列表
        for file_type in file_type_list:
            # 构建完整的文件路径
            path = f"{self.output_path}/{file_name}.{file_type}"

            # 处理PDF文件类型
            if file_type == 'pdf':
                # 将markdown格式数据转换为HTML
                html = markdown.markdown(data)
                # 将HTML内容写入PDF文件
                HTML(string=html).write_pdf(path)
                # 记录日志，提示PDF文件已生成
                logger.info(f"{path}已生成")

            # 处理Markdown文件类型
            elif file_type == 'md':
                # 以写入模式打开文件，指定编码为utf-8
                with open(path, 'w', encoding='utf-8') as file:
                    # 将数据写入文件，确保中文等特殊字符正常显示
                    json.dump(data, file, ensure_ascii=False)
                # 记录日志，提示Markdown文件已生成
                logger.info(f"{path}已生成")

            # 处理未知文件类型
            else:
                # 记录错误日志，提示不支持的文件类型
                logger.error(f"Unknown file type {file_type}")

    def get_brief_summary(self, common_data_dict: dict, file_type_list: list=["md", "pdf"]):
        """
        生成文章的简要摘要，整合文章叙事结构、章节信息和背景知识

        参数:
            common_data_dict (dict): 包含通用数据和背景知识的字典，用于辅助生成摘要
            file_type_list: list - 要保存的文件类型列表，默认为["md", "pdf"]

        功能说明:
            1. 构造查询指令，要求按照文章原有结构整理主要内容
            2. 调用get_answer方法获取摘要结果
            3. 将生成的摘要保存到指定位置

        无返回值，结果通过save_data_to_file方法保存
        """
        if len(file_type_list) == 0:
            logger.info("文章简要摘要已经生成，无需重复生成！")
            return

        # 构造用于生成简要摘要的查询指令
        query = "请按照文章本身的叙事结构，整理这篇文章的主要内容，每个章节都需要有一定的简单介绍。如果背景知识中有一些文章的基本信息也需要一并总结。仅需要返回相关内容，多余的话无需返回。"

        # 记录开始生成简要摘要的日志
        logger.info(f"开始生成文章简要摘要...")

        # 调用回答生成方法获取简要摘要
        answer = self.get_answer(self.total_summary, query, common_data_dict)

        # 记录摘要生成完成的日志
        logger.info(f"文章简要摘要生成完成，长度: {len(answer)} 字符")

        # 保存生成的简要摘要
        self.save_data_to_file(answer, "brief_summary", file_type_list)

    def get_detail_summary(self, raw_data_dict: dict, file_type_list: list=["md", "pdf"]):
        """
        生成文章的详细摘要，按原始数据结构逐部分处理

        参数:
            raw_data_dict (dict): 原始数据字典，键为标题，值为包含分页内容的字典
            file_type_list: list - 要保存的文件类型列表，默认为["md", "pdf"]

        功能说明:
            1. 遍历原始数据中的每个标题对应的内容
            2. 去重整合同一标题下的所有分页内容
            3. 按标题生成对应部分的详细摘要
            4. 合并所有部分摘要并保存

        无返回值，结果通过save_data_to_file方法保存
        """
        if len(file_type_list) == 0:
            logger.info("文章详细摘要已经生成，无需重复生成！")
            return

        # 初始化总摘要字符串
        total_answer = ""

        # 构造用于生成详细摘要的查询模板
        query = "按照人类的习惯理解并且总结 {title} 的内容。最后以 blog 的格式返回，需要注意换行。仅需要返回相关内容，多余的话无需返回。不需要对章节进行单独总结。"

        # 记录开始生成详细摘要的日志
        logger.info(f"开始生成详细摘要，共包含 {len(raw_data_dict)} 个部分...")

        # 遍历每个标题对应的内容
        for title, data_dict in raw_data_dict.items():
            logger.info(f"正在处理: {title}")

            # 收集并去重同一标题下的所有页面内容
            context_data = []
            for page, raw_data in data_dict.items():
                if raw_data not in context_data:
                    context_data.append(raw_data)

            # 记录当前部分的内容数量
            logger.info(f"{title} 包含 {len(context_data)} 个非重复内容块")

            # 生成当前标题对应的详细摘要
            answer = self.get_answer(context_data, query.format(title=title))

            # 将当前部分的摘要添加到总摘要中
            total_answer += "\n\n" + answer

        # 保存合并后的详细摘要
        self.save_data_to_file(total_answer, "detail_summary", file_type_list)

    def generate_output_file(self, pdf_file_path: str, common_data_dict: dict, raw_data_dict: dict):
        # 当保存数据标志为True时，执行文章总结和文件导出流程
        logger.info(f"开始总结文章信息并导出文件")
    
        # 构建完整输出路径，拼接PDF文件路径作为子目录
        self.output_path += f"/{pdf_file_path}"
        logger.debug(f"输出目录路径: {self.output_path}")
    
        # 创建输出目录（如果不存在）
        makedir(self.output_path)
        
        # 定义需要处理的摘要类型列表
        summary_type_list = ["brief_summary", "detail_summary"]
        # 定义需要生成的文件格式列表
        file_type_list = ["md", "pdf"]
    
        # 遍历每种摘要类型，检查并生成缺失的文件
        for summary_type in summary_type_list:
            # 记录当前正在处理的摘要类型
            logger.info(f"开始处理摘要类型: {summary_type}")
    
            # 存储当前摘要类型下缺失的文件格式
            mis_type = []
    
            # 检查每种文件格式是否已存在
            for file_type in file_type_list:
                # 构建完整的文件路径
                file_path = f"{self.output_path}/{summary_type}.{file_type}"
                logger.debug(f"检查文件是否存在: {file_path}")
    
                # 如果文件不存在，添加到缺失列表
                if not is_file_exists(file_path):
                    mis_type.append(file_type)
    
            # 如果存在缺失的文件格式，调用对应方法生成
            if mis_type:
                logger.info(f"摘要类型 {summary_type} 存在缺失的文件格式: {mis_type}")
    
                # 根据摘要类型调用对应的生成方法，并传入缺失的格式列表
                if summary_type == "brief_summary":
                    self.get_brief_summary(common_data_dict, mis_type)
    
                elif summary_type == "detail_summary":
                    self.get_detail_summary(raw_data_dict, mis_type)
            else:
                # 所有格式文件都已存在，无需生成
                logger.info(f"摘要类型 {summary_type} 的所有文件格式均已存在，无需生成")

    def process_pdf(self, pdf_file_path: str, save_data_flag: bool=True) -> Any:
        """
        主流程：读取 PDF 数据，提取结构，分章节总结，最终生成详细回答。
        Main pipeline: read PDF data, extract structure, summarize by section, and generate final answer.
        Args:
            pdf_file_path (str): PDF 文件名（不含路径）。
        Returns:
            Any: 最终回答。
        """
        vector_db_path = os.path.join(f"{self.vector_db_path}/{pdf_file_path}_data_index")
        logger.info(f"开始处理PDF主流程: {pdf_file_path}")
        try:
            with open(f"{self.json_data_path}/{pdf_file_path}.json", 'r', encoding='utf-8') as f:
                self.pdf_raw_data = json.load(f)
            logger.info(f"成功读取本地JSON数据: {self.json_data_path}/{pdf_file_path}.json")
        except Exception as e:
            logger.warning(f"读取本地JSON失败，将重新提取: {e}")
            self.pdf_raw_data = self.extract_pdf_data(pdf_file_path)

        # 初始化存储变量
        common_data_dict = {}
        vector_db_content_docs = []
        raw_data_dict = {}

        if os.path.exists(vector_db_path):
            self.vector_db = self.load_vector_db(vector_db_path)
            all_db_res = self.vector_db.similarity_search_with_score("", k=99999999)
            # 对检索回来的章节内容按 pages 的第一个页码进行排序，保证章节顺序与原文一致
            all_db_res_sorted = sorted(
                all_db_res,
                key=lambda x: x[0].metadata.get("pages", [float('inf')])[0] if x[0].metadata.get("pages") else float('inf')
            )
            self.total_summary = {}
            logger.info(f"检索回 {len(all_db_res)} 章节内容（已按页码排序）")
            for db_res in all_db_res_sorted:
                metadata = db_res[0].metadata  # db_res[0] 是 Document
                title = db_res[0].page_content  # 章节标题
                summary = metadata.get("summary")  # 章节摘要
                pages = metadata.get("pages")     # 章节页码列表
                data = metadata.get("raw_data")
                # "PDF common info" 是全局信息，单独处理
                if title == "PDF common info":
                    common_data_dict = summary
                else:
                    self.total_summary[title] = summary
                    self.agenda_dict[title] = pages

                raw_data_dict[title] = data

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
            self.total_summary = {}
            for agenda_data in agenda_data_list:
                title = agenda_data.get("title")
                data = agenda_data.get("data")
                page = agenda_data.get("pages")
                logger.info(f"正在总结章节: {title}")
                summary = self.summary_content(title, list(data.values()))
                self.total_summary[title] = summary
                vector_db_content_docs.append(
                    Document(
                        page_content=title,
                        metadata={
                            "pages": page,
                            "raw_data": data,
                            "summary": summary,
                        }
                    )
                )
                raw_data_dict[title] = data
            logger.info(f"所有章节摘要已完成，正在构建向量数据库...")
            self.vector_db = self.build_vector_db(vector_db_path, vector_db_content_docs)
            logger.info(f"向量数据库构建完成。")

        if save_data_flag:
            self.generate_output_file(pdf_file_path, common_data_dict, raw_data_dict)

        logger.info(f"PDF处理流程结束。")

    def retrieval_data(self, response):
        """
        检索用户请求相关的章节内容。
        Retrieve relevant section content for user queries.
        Args:
            response: LLM 返回的章节标题列表。
        Returns:
            context_data: 当前检索到的上下文内容。
        """
        title_list = response.get("title", [])
        context_data = []
        for title in title_list:
            raw_data_dict = {}
            if title not in self.retrieval_data.keys():
                logger.info(f"通过向量数据库检索章节: '{title}' ...")
                doc_res = self.vector_db.similarity_search_with_score(title, k=1)
                if doc_res:
                    raw_data_dict = doc_res[0][0].metadata.get("raw_data", {})
                    self.retrieval_data[title] = raw_data_dict
                    logger.info(f"检索到章节 '{title}' 内容")
                else:
                    logger.warning(f"章节 '{title}' 未在向量数据库中检索到相关内容。")
                    raw_data_dict = {}

            else:
                logger.info(f"'{title}'已经被检索过，跳过当前章节检索")
                raw_data_dict = self.retrieval_data.get(title)

            #if not self.is_content_in_history(title_context, "answer"):
            #    context_data.append(title_context)
            for page, raw_data in raw_data_dict.items():
                if raw_data not in context_data:
                    context_data.append(raw_data)

        logger.info(f"检索数据完成，涉及章节数: {len(title_list)}")
        logger.info(f"检索数据完成，涉及页数: {len(context_data)}")
        return context_data

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
        chat_count = 0
        logger.info(f"启动主流程，处理 PDF 文件: {pdf_file_path}")
        while True:
            if chat_count == 0:
                self.process_pdf(pdf_file_path, save_data_flag)

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
            chat_count += 1