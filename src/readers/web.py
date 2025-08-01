import asyncio
import json
import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

import fitz  # 导入pymupdf库，它在导入时别名为fitz
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage, AIMessage

from src.readers.base import ReaderBase
from src.core.processing.text_splitter import StrictOverlapSplitter
from src.config.settings import (
    SYSTEM_PROMPT_CONFIG,
    MCP_CONFIG,
    MCPToolName,
    WEB_MAX_TOKEN_COUNT,
    ReaderRole
)
from src.services.mcp_client import MCPClient
from src.utils.helpers import extract_name_from_url, makedir, parse_latest_plugin_call, extract_data_from_LLM_res
from src.core.vector_db.vector_db_client import VectorDBClient

# 常量定义
TOOL_STOP_FLAG = "Observation"  # MCP工具调用停止标识
VECTOR_DB_SUFFIX = "_data_index"  # 向量数据库路径后缀
FORMAT_DATA_SUFFIX = "_format_data.json"  # 格式化数据文件后缀

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class WebReader(ReaderBase):
    """
    Web内容读取器，用于从URL获取网页内容并进行处理、摘要和问答交互。

    该类继承自ReaderBase，支持通过MCP服务获取网页内容，处理大文件时分块存储到向量数据库，
    并提供与用户的交互式对话功能。

    Attributes:
        web_content (str): 存储处理后的网页内容
        spliter (StrictOverlapSplitter): 文本分块器实例，用于大文本分块
        vector_db_obj (VectorDBClient): 向量数据库客户端实例，用于存储和检索分块数据
    """

    def __init__(self, provider: str = "azure") -> None:
        """
        初始化WebReader对象，设置LLM提供商和文本分块器。

        Args:
            provider (str): LLM服务提供商，可选值为'azure'、'openai'、'ollama'，默认为'azure'
        """
        super().__init__(provider)
        self.web_content: str = ""  # 初始化网页内容为空字符串
        self.spliter = StrictOverlapSplitter(
            overlap=1,
            token_threshold=1000,
            delimiter='\n\n',  # 以空行作为文本切分符
        )
        self.vector_db_obj: Optional[VectorDBClient] = None  # 向量数据库客户端，延迟初始化

    def remove_error_blocks(self, text: str) -> Tuple[str, List[str]]:
        """
        移除文本中包含的<error>错误块内容

        错误块指被<error>和</error>标签包裹的内容，通常为MCP服务返回的错误信息。
        该方法会清理文本并记录错误块，便于后续排查问题。

        Args:
            text (str): 原始文本内容，可能包含<error>标签

        Returns:
            tuple: 包含两个元素
                - 清理后的文本（移除所有错误块）
                - 匹配到的错误块列表（保留原始错误内容）
        """
        # 正则表达式模式：匹配<error>和</error>之间的所有内容（包括换行符）
        pattern = r'<error>.*?</error>'
        # 使用re.DOTALL让.匹配包括换行符在内的所有字符
        matched_blocks = re.findall(pattern, text, flags=re.DOTALL)
        # 清理文本中的错误块
        cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
        
        if matched_blocks:
            logger.warning(f"已移除{len(matched_blocks)}个错误块内容，示例: {matched_blocks[0][:50]}...")
        return cleaned_text, matched_blocks

    async def call_mcp_server(self, input_prompt: str, mcp_config: dict, session_id: str) -> List[str]:
        """
        调用MCP服务处理输入提示并获取网页内容

        通过React模式与MCP服务交互，循环调用工具直到获取有效内容或达到最大尝试次数。
        每次工具调用结果会被检查并清理错误块，避免重复内容存入结果列表。

        Args:
            input_prompt (str): 用户输入提示，用于指导MCP服务获取内容
            mcp_config (dict): MCP服务配置，包含服务地址、认证信息等
            session_id (str): 会话ID，用于跟踪本次MCP服务调用

        Returns:
            List[str]: 网页内容片段列表，已移除错误块且无重复内容
        """
        mcp_client = MCPClient()  # 初始化MCP客户端
        await mcp_client.async_init(mcp_config)
        logger.info(f"MCP服务初始化完成，会话ID: {session_id}")

        # 自定义React模型，设置停止标识
        self.react_model = self._customize_mode(kwargs={"stop": TOOL_STOP_FLAG})
        
        # 构建系统提示词，包含工具描述和调用格式
        system_prompt = """
            Answer the following questions as best you can. You have access to the following tools:

            {tool_descs}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tools_name}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!
        """.format(tools_name=mcp_client.tools_name, tool_descs=mcp_client.tool_descs)

        # 构建React调用链
        react_chain = await self.async_build_chain(
            client=self.react_model,
            system_prompt=system_prompt,
        )

        max_attempts = 10  # 最大尝试调用工具次数
        attempt_count = 0  # 当前尝试次数
        web_result: List[str] = []  # 存储网页内容结果
        logger.info(f"开始MCP服务调用循环，最大尝试次数: {max_attempts}")

        while attempt_count <= max_attempts:
            tool_status: List[str] = []  # 记录工具调用返回的错误状态
            # 调用React链获取响应
            response = react_chain.invoke(
                {"input_prompt": input_prompt},
                config={"configurable": {"session_id": session_id}}
            )
            logger.debug(f"React链响应: {response[:200]}...")  # 日志截断，避免过长

            # 解析工具调用信息
            function_name_str, parameters_str, final_res = parse_latest_plugin_call(response)
            session_name, session = mcp_client.tool_name_to_session.get(function_name_str, ("", None))
            logger.info(
                f"工具调用信息 - 函数: {function_name_str}, "
                f"会话: {session_name}, "
                f"参数: {parameters_str[:100]}..."  # 截断参数日志
            )

            # 处理工具调用参数（替换布尔值小写为Python格式）
            parameters_str = parameters_str.replace("true", "True").replace("false", "False")
            result_content = ""

            if session:  # 如果存在有效的工具会话
                try:
                    # 安全解析参数（替换eval为json.loads，避免安全风险）
                    parameters = json.loads(parameters_str) if parameters_str else {}
                    # 调用工具并获取结果
                    result = await session.call_tool(function_name_str, parameters)
                    # 清理结果中的错误块
                    cleaned_res, tool_status = self.remove_error_blocks(result.content[0].text)
                    
                    # 检查是否为重复内容
                    if cleaned_res in web_result:
                        logger.warning(f"检测到重复内容，已跳过: {cleaned_res[:50]}...")
                        final_res += "\n\n信息已经被检索过!!"
                    else:
                        web_result.append(cleaned_res)
                        logger.info(f"成功获取网页内容片段，长度: {len(cleaned_res)}字符")

                except json.JSONDecodeError as e:
                    logger.error(f"参数解析失败（JSON格式错误）: {e}, 参数: {parameters_str}")
                except Exception as e:
                    logger.error(f"工具调用失败: {e}")

            # 记录工具错误状态
            if tool_status:
                logger.warning(f"工具返回错误: {tool_status}")
                final_res += str(tool_status)

            # 将结果添加到对话历史
            self.add_message_to_history(session_id=session_id, message=HumanMessage(final_res))

            # 如果没有有效的工具调用，退出循环
            if not function_name_str or not parameters_str:
                logger.info("未检测到有效工具调用，退出MCP服务循环")
                break

            attempt_count += 1

        # 清理MCP客户端资源
        await mcp_client.cleanup()
        logger.info(f"MCP服务调用完成，共获取{len(web_result)}个网页内容片段")
        return web_result

    async def get_web_content(self, url: str, url_name: str) -> List[str]:
        """
        从指定URL获取网页内容并保存到JSON文件

        通过调用MCP服务获取网页内容，并将原始结果保存到本地JSON文件，
        便于后续复用（避免重复网络请求）。

        Args:
            url (str): 网页URL地址（需http/https协议）
            url_name (str): URL名称（用于生成本地保存文件名，通常为URL提取的标识）

        Returns:
            list: 网页内容片段列表（MCP服务返回的原始结果，已处理错误块）
        """
        # 获取MCP服务配置
        config = MCP_CONFIG.get(MCPToolName.WEB_SEARCH)
        if not config:
            raise ValueError(f"未找到MCP工具配置: {MCPToolName.WEB_SEARCH}")

        # 构建获取网页内容的提示词（要求Markdown格式便于后续处理）
        input_prompt = f"请获取该URL的所有当前内容: {url}，并以Markdown格式返回全部信息。"
        logger.info(f"开始获取网页内容: {url}")

        # 调用MCP服务获取内容
        web_content = await self.call_mcp_server(input_prompt, config, MCPToolName.WEB_SEARCH)

        # 保存内容到本地JSON文件
        save_path = os.path.join(self.json_data_path, f"{url_name}.json")
        with open(save_path, 'w', encoding='utf-8') as file:
            json.dump(web_content, file, ensure_ascii=False)
        logger.info(f"网页内容已保存到本地: {save_path}")

        return web_content

    async def process_web(self, url: str, save_data_flag: bool = True) -> None:
        """
        处理网页内容：优先从本地加载，本地不存在则从网络获取，根据内容大小选择处理方式

        处理逻辑：
        1. 尝试加载本地缓存的网页内容（JSON文件）
        2. 本地无缓存时，调用MCP服务获取并保存
        3. 根据内容token数判断：
           - 小于等于阈值：直接生成摘要并保存
           - 大于阈值：分块后存入向量数据库，用于后续检索问答

        Args:
            url (str): 网页URL地址
            save_data_flag (bool): 是否保存处理后的数据（摘要或向量数据库），默认为True
        """
        # 从URL提取名称（用于文件命名）
        url_name = extract_name_from_url(url)
        logger.info(f"开始处理网页: {url}，提取名称: {url_name}")

        # 尝试加载本地缓存
        try:
            cache_path = os.path.join(self.json_data_path, f"{url_name}.json")
            with open(cache_path, 'r', encoding='utf-8') as f:
                web_content = json.load(f)
            logger.info(f"成功加载本地缓存: {cache_path}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"本地缓存加载失败（{type(e).__name__}），将重新获取: {e}")
            web_content = await self.get_web_content(url, url_name)

        # 计算内容总token数，判断处理方式
        content_str = ', '.join(web_content)
        token_count = self.spliter.count_tokens(content_str)
        logger.info(f"网页内容总token数: {token_count}，处理阈值: {WEB_MAX_TOKEN_COUNT}")

        if token_count <= WEB_MAX_TOKEN_COUNT:
            # 内容较小，直接生成摘要
            self.output_path += f"/{url_name}"  # 构建输出路径
            if not os.path.exists(self.output_path):  # 避免重复生成摘要
                # 构建摘要链
                summary_chain = await self.async_build_chain(
                    self.chat_model,
                    SYSTEM_PROMPT_CONFIG.get(ReaderRole.SUMMARY)
                )
                # 构建摘要提示词
                query = f"请分析总结当前web页面的内容，按照文章本身的写作顺序给出详细的总结：{content_str}"
                summary = summary_chain.invoke(
                    {"input_prompt": query},
                    config={"configurable": {"session_id": "summary"}}
                )

                if save_data_flag:
                    makedir(self.output_path)  # 创建输出目录
                    self.save_data_to_file(summary, url_name)  # 保存摘要
                    logger.info(f"摘要已保存到: {self.output_path}")

            self.web_content = content_str  # 保存完整内容用于问答

        else:
            # 内容较大，分块存入向量数据库
            vector_db_path = os.path.join(self.vector_db_path, f"{url_name}{VECTOR_DB_SUFFIX}")
            self.vector_db_obj = VectorDBClient(vector_db_path)

            if os.path.exists(vector_db_path):
                # 加载已存在的向量数据库
                self.get_data_from_vector_db()
                logger.info(f"已加载向量数据库: {vector_db_path}")
            else:
                # 文本分块处理
                raw_web_data_list = self.spliter.split_text(content_str)
                logger.info(f"文本分块完成，共{len(raw_web_data_list)}块")

                # 保存分块数据到本地
                format_data_path = os.path.join(self.json_data_path, f"{url_name}{FORMAT_DATA_SUFFIX}")
                with open(format_data_path, 'w', encoding='utf-8') as file:
                    json.dump(raw_web_data_list, file, ensure_ascii=False)
                logger.info(f"分块数据已保存到: {format_data_path}")

                # 分块入库
                chunks = self.spliter.split_into_chunks(raw_web_data_list)
                self.get_data_from_json_dict(chunks, raw_web_data_list)
                logger.info(f"分块数据已存入向量数据库: {vector_db_path}")

            if save_data_flag:
                # 生成输出文件
                self.generate_output_file(url_name, self.common_data_dict, self.raw_data_dict)
                logger.info(f"向量数据库输出文件已生成: {url_name}")

        logger.info(f"网页处理流程结束: {url}")

    def chat(self, input_prompt: str) -> str:
        """
        处理用户对话输入并生成回答

        回答逻辑：
        1. 若已加载完整网页内容（小文件），直接基于内容生成回答
        2. 若内容已分块存入向量数据库（大文件），先检索相关分块再生成回答

        Args:
            input_prompt (str): 用户输入的问题（需与网页内容相关）

        Returns:
            str: 生成的回答内容（自然语言）
        """
        if self.web_content:
            # 基于完整内容回答
            logger.info(f"使用完整网页内容回答问题: {input_prompt[:50]}...")
            answer = self.get_answer(self.web_content, input_prompt)
        else:
            # 基于向量数据库检索回答
            if not self.vector_db_obj:
                raise RuntimeError("向量数据库未初始化，请先调用process_web处理URL")

            # 调用LLM链解析问题
            response = self.call_llm_chain(
                ReaderRole.CHAT,
                input_prompt,
                "chat",
                system_format_dict={"agenda_dict": self.agenda_dict}
            )
            logger.debug(f"LLM问题解析响应: {response[:200]}...")

            try:
                # 提取检索关键词
                extract_response = extract_data_from_LLM_res(response)
                # 检索相关数据
                context_data = self.retrieval_data(extract_response)
                logger.info(f"检索到{len(context_data)}条相关分块数据")
            except Exception as e:
                logger.warning(f"LLM响应解析失败，使用原始响应检索: {e}")
                context_data = response

            # 基于检索结果生成回答
            logger.info(f"使用向量数据库检索结果回答问题: {input_prompt[:50]}...")
            answer = self.get_answer(context_data, input_prompt)

        logger.info(f"对话回答生成完毕，长度: {len(answer)}字符")
        return answer

    async def main(self, url: str) -> None:
        """
        程序主入口，处理网页URL并启动交互式对话

        流程：
        1. 调用process_web处理URL（加载/获取内容，根据大小处理）
        2. 启动交互式命令行对话，接收用户输入并返回回答
        3. 支持用户输入"退出"等指令结束对话

        Args:
            url (str): 要处理的网页URL（需完整且可访问）
        """
        await self.process_web(url)
        logger.info("网页内容处理完成，开始交互式对话（输入'退出'结束）")

        while True:
            user_input = input("You: ")

            # 检查退出指令
            if user_input.lower() in ["退出", "再见", "bye", "exit", "quit"]:
                print("Chatbot: 再见！期待下次与您对话。")
                break

            # 生成回答并记录对话历史
            answer = self.chat(user_input)
            self.add_message_to_history(session_id="chat", message=AIMessage(answer))

            # 打印对话内容
            print(f"User: {user_input}")
            print(f"ChatBot: {answer}")
            print("======" * 10)


if __name__ == "__main__":
    # 示例：创建WebReader实例并运行（实际使用需传入具体URL）
    web_reader_obj = WebReader()
    target_url = input("请输入要处理的网页URL: ")
    asyncio.run(web_reader_obj.main(target_url))