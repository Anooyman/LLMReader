import json
import logging
import markdown
from weasyprint import HTML

from langchain.docstore.document import Document
from src.core.llm.client import LLMBase
from src.config.settings import (
    JSON_DATA_PATH,
    OUTPUT_PATH,
    VECTOR_DB_PATH
)
from src.utils.helpers import *

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
MAX_CHAPTER_LEN = 10
"""
基础阅读器类，提供PDF/网页等文档处理的通用功能，包括：
- 数据保存与加载
- 摘要生成（简要/详细）
- 向量数据库交互基础逻辑

该类继承自LLMBase，为子类（如PDFReader、WebReader）提供通用方法。
"""
class ReaderBase(LLMBase):
    """
    基础阅读器类，封装文档处理的核心流程与通用工具方法。
    """
    def __init__(self, provider: str = "azure") -> None:
        """
        初始化基础阅读器，配置LLM提供商及数据存储路径。
        
        参数:
            provider (str): LLM服务提供商，支持'azure'（默认）、'openai'、'ollama'。
        
        说明:
            数据存储路径依赖全局变量：
            - JSON_DATA_PATH: 原始数据JSON存储路径
            - OUTPUT_PATH: 生成文件（摘要等）输出路径
            - VECTOR_DB_PATH: 向量数据库存储路径
        """
    def __init__(self, provider: str = "azure") -> None:
        """
        初始化 PDFReader 对象，支持多 LLM provider。
        provider: 'azure'（默认）、'openai'、'ollama'。
        """
        super().__init__(provider)
        self.json_data_path = JSON_DATA_PATH
        self.output_path = OUTPUT_PATH
        self.vector_db_path = VECTOR_DB_PATH
        self.agenda_dict = {}
        self.retrieval_data_dict = {}
        self.vector_db = None
        self.retrieval_dict = {}
        self.total_summary = {}
        self.raw_data_dict = {}
        self.common_data_dict = {}

        for path in [self.json_data_path, self.output_path, self.vector_db_path]:
            makedir(path)

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
        query = "请按照文章本身的叙事结构，整理这篇文章的主要内容，每个章节都需要有一定的简单介绍。如果背景知识中有一些文章的基本信息也需要一并总结。仅需要返回相关内容，多余的话无需返回。返回中文，markdown格式。"

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
        query = "按照人类的习惯理解并且总结 {title} 的内容。最后以 blog 的格式返回，需要注意标题(如果标题中有数字则需要写出到结果中)，换行，加粗关键信息。仅需要返回相关内容的总结信息，多余的话无需返回。不需要对章节进行单独总结。返回中文"

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
            total_answer += "\n\n --- \n\n " + answer

        # 保存合并后的详细摘要
        self.save_data_to_file(total_answer, "detail_summary", file_type_list)

    def generate_output_file(self, file_path: str, common_data_dict: dict, raw_data_dict: dict):
        # 当保存数据标志为True时，执行文章总结和文件导出流程
        logger.info(f"开始总结文章信息并导出文件")
    
        # 构建完整输出路径，拼接文件路径作为子目录
        self.output_path += f"/{file_path}"
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

    def get_data_from_json_dict(self, chunks: list, json_data_dict: dict) -> None:
        """
        从分块数据中提取文档信息（基本信息、目录结构）并构建向量数据库内容。
        
        参数:
            chunks (list): 分块后的原始数据列表
            json_data_dict (dict): 原始JSON数据字典
        
        """
        vector_db_content_docs = []
        agenda_list = []
        for index, chunk in enumerate(chunks):
            logger.info(f"处理第 {index+1}/{len(chunks)} 块...")
            # 只在第一个 chunk 提取基本信息
            if index == 0:
                logger.info(f"获取基本信息（第一个chunk）...")
                self.common_data_dict = self.get_basic_info(chunk)
                logger.info(f"PDF 基本信息: {self.common_data_dict}")

            logger.info(f"获取目录结构（第 {index+1} 块）...")
            agenda = self.get_agenda(chunk)
            logger.info(f"第 {index+1} 块目录结构: {agenda}")
            agenda_list.extend(agenda)

        vector_db_content_docs.append(
            Document(
                page_content="PDF common info",
                metadata={
                    "pages": [0],
                    "raw_data": self.common_data_dict,
                    "summary": self.common_data_dict,
                }
            )   
        )
        # 合并所有 chunk 的目录结构后，按章节分组
        logger.info(f"合并所有 chunk 的目录结构，准备分组...")
        agenda_data_list, self.agenda_dict = group_data_by_sections_with_titles(agenda_list, json_data_dict)

        # 检查每个章节的长度，如果长度大于 MAX_CHAPTER_LEN，则需要重新获取目录结构
        agenda_list = self.check_len_of_each_chapter(agenda_list, agenda_data_list)
        logger.info(f"重新分组后的章节数: {len(agenda_list)}")

        # 重新分组
        agenda_data_list, self.agenda_dict = group_data_by_sections_with_titles(agenda_list, json_data_dict)
        logger.info(f"章节分组完成，章节数: {len(agenda_data_list)}")
        logger.info(f"最终章节信息如下: {self.agenda_dict}")
        logger.info(f"开始分章节总结...")
        for agenda_data in agenda_data_list:
            title = agenda_data.get("title")
            data = agenda_data.get("data")
            page = agenda_data.get("pages")
            logger.info(f"正在总结章节: {title}")
            summary = self.summary_content(title, list(data.values()))
            refactor_content = self.refactor_content(title, list(data.values()))
            self.total_summary[title] = summary
            vector_db_content_docs.append(
                Document(
                    page_content=title,
                    metadata={
                        "pages": page,
                        "raw_data": data,
                        "summary": summary,
                        "refactor": refactor_content,
                    }
                )
            )
            self.raw_data_dict[title] = data
        logger.info(f"所有章节摘要已完成，正在构建向量数据库...")
        self.vector_db = self.vector_db_obj.build_vector_db(vector_db_content_docs)
        logger.info(f"向量数据库构建完成。")

    def check_len_of_each_chapter(self, agenda_list: list, agenda_data_list: list):
        """
        检查每个章节的长度，如果长度大于 MAX_CHAPTER_LEN，则需要重新获取目录结构
        """
        for agenda_data in agenda_data_list:
            title = agenda_data.get("title")
            data = agenda_data.get("data")
            page = agenda_data.get("pages")
            if len(page) > MAX_CHAPTER_LEN:
                logger.info(f"章节: {title} 长度大于 {MAX_CHAPTER_LEN}，需要重新获取目录结构")
                count = 0
                while count < 5:
                    try:
                        agenda = self.get_sub_agenda(data)
                        if len(agenda) > 1:
                            agenda_list.extend(agenda)
                            break
                    except:
                        logger.info(f"子章节提取内容错误，正在重试。当前重试次数{count}")
                    count += 1
                logger.info(f"重新获取目录结构完成，目录结构: {agenda}")
        return agenda_list

    def get_data_from_vector_db(self):
        self.vector_db = self.vector_db_obj.load_vector_db()
        all_db_res = self.vector_db.similarity_search_with_score("", k=99999999)
        # 对检索回来的章节内容按 pages 的第一个页码进行排序，保证章节顺序与原文一致
        all_db_res_sorted = sorted(
            all_db_res,
            key=lambda x: x[0].metadata.get("pages", [float('inf')])[0] if x[0].metadata.get("pages") else float('inf')
        )
        logger.info(f"检索回 {len(all_db_res)} 章节内容（已按页码排序）")
        for db_res in all_db_res_sorted:
            metadata = db_res[0].metadata  # db_res[0] 是 Document
            title = db_res[0].page_content  # 章节标题
            summary = metadata.get("summary")  # 章节摘要
            pages = metadata.get("pages")     # 章节页码列表
            data = metadata.get("raw_data")
            # "Common info" 是全局信息，单独处理
            if title == "Common info":
                self.common_data_dict = summary
            else:
                self.total_summary[title] = summary
                self.agenda_dict[title] = pages

            self.raw_data_dict[title] = data

        logger.info(f"当前文章目录结构如下: {self.agenda_dict}")
        # 返回值说明：
        # total_summary: 按章节顺序有序的 dict，key 为章节标题，value 为摘要
        # self.agenda_dict: 按章节顺序有序的 dict，key 为章节标题，value 为页码列表

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
        total_page_content = []
        for title in title_list:
            if title not in self.retrieval_data_dict.keys():
                logger.info(f"通过向量数据库检索章节: '{title}' ...")
                doc_res = self.vector_db.similarity_search_with_score(title, k=1)
                if doc_res:
                    refactor_data = doc_res[0][0].metadata.get("refactor", "")
                    page_content = list(doc_res[0][0].metadata.get("raw_data", {}).keys())
                    self.retrieval_data_dict[title] = {
                        "data": refactor_data, 
                        "page": page_content
                    }
                    logger.info(f"检索到章节 '{title}' 内容")
                else:
                    logger.warning(f"章节 '{title}' 未在向量数据库中检索到相关内容。")
                    refactor_data = ""

            else:
                logger.info(f"'{title}'已经被检索过，跳过当前章节检索")
                refactor_data = self.retrieval_data_dict.get(title, {}).get("data", "")
                page_content = self.retrieval_data_dict.get(title, {}).get("page", "")
            # 按 page 的大小（页码顺序）拼接数据，确保顺序正确
            # 假设 page 是可以转换为整数的页码
            #sorted_items = sorted(raw_data_dict.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else x[0])
            #for page, raw_data in sorted_items:
            #    if raw_data not in context_data:
            #        context_data.append(raw_data)
            #    else:
            #        logger.info(f"内容已经被加入到 context 信息中!")

            if refactor_data not in context_data:
                context_data.append(refactor_data)
            else:
                logger.info(f"内容已经被加入到 context 信息中!")
            total_page_content.extend(page_content)

        logger.info(f"检索数据完成，涉及章节数: {len(title_list)}")
        logger.info(f"检索数据完成，涉及页数: {total_page_content}")
        return context_data

