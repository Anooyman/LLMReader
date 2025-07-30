import asyncio
import base64
import json
import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from tqdm import tqdm
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
from src.utils.helpers import *
from src.core.vector_db.vector_db_client import VectorDBClient

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class WebReader(ReaderBase):

    def __init__(self, provider: str = "azure") -> None:
        """
        初始化 PDFReader 对象，支持多 LLM provider。
        provider: 'azure'（默认）、'openai'、'ollama'。
        """
        super().__init__(provider)
        self.web_content = ""
        self.spliter = StrictOverlapSplitter(
            overlap=1,
            token_threshold=1000,
            delimiter='\n\n',  # 可以指定切分符
        )

    def remove_error_blocks(self, text):
        # 正则表达式模式：匹配<error>和</error>之间的所有内容（包括换行符）
        pattern = r'<error>.*?</error>'
        # 使用re.DOTALL让.匹配包括换行符在内的所有字符
        # 先找到所有匹配的内容
        matched_blocks = re.findall(pattern, text, flags=re.DOTALL)
        # 清理文本
        cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
        # 返回清理后的文本和匹配到的错误块列表
        return cleaned_text, matched_blocks

    async def call_mcp_server(self, input_prompt: str, mcp_config: dict, session_id: str) -> Any:
        """
        针对用户输入进行对话。
        Interactive chat for user input.
        Args:
            input_prompt (str): 用户输入。
            last_retrieval_data (dict): 上一次检索的数据缓存。
        Returns:
            Any: 回答内容。
        """
        mcp_clinet = MCPClient()
        await mcp_clinet.async_init(mcp_config)
        self.react_model = self._customize_mode(kwargs={"stop": "Observation"})
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
        """
        react_chain = await self.async_build_chain(
            client=self.react_model,
            system_prompt=system_prompt.format(tools_name=mcp_clinet.tools_name, tool_descs=mcp_clinet.tool_descs),
        )
        count = 0
        max_count = 10
        web_result = []
        while count <= max_count: 
            tool_status = ''
            response = react_chain.invoke(
                {"input_prompt": input_prompt},
                config={"configurable": {"session_id": session_id}}
            )
            logger.debug(f"Chain response: {response}")
            function_name_str, parameters_str, final_res = parse_latest_plugin_call(response)
            session_name, session = mcp_clinet.tool_name_to_session.get(function_name_str, ("", None))
            logger.info(f"Function: {function_name_str}, Session: {session_name}, Function Params: {parameters_str}, Final ans lenght: {len(final_res)}")
            parameters_str = parameters_str.replace("true", "True")
            parameters_str = parameters_str.replace("false", "False")
            result = ""
            if session:
                result = await session.call_tool(function_name_str, eval(parameters_str))
                res, tool_status = self.remove_error_blocks(result.content[0].text)
                if res in web_result:
                    logger.warning(f"信息已经被检索过!!")
                else:
                    web_result.append(res)
                logger.info(f"Get {len(res)} string from web.")

            if tool_status:
                logger.info(f"tool status: {tool_status}")
                self.add_message_to_history(session_id=session_id, message=HumanMessage(str(tool_status)))

            if not function_name_str or not parameters_str:
                logger.info("No valid tool call detected, returning final answer directly.")
                break
        await mcp_clinet.cleanup()

        return web_result

    async def get_web_content(self, url, url_name):

        config = MCP_CONFIG.get(MCPToolName.WEB_SEARCH)
        input_prompt = f"Please retrieve all the current in this url: {url}."
        web_content = await self.call_mcp_server(input_prompt, config, MCPToolName.WEB_SEARCH)

        with open(os.path.join(self.json_data_path, f"{url_name}.json"), 'w', encoding='utf-8') as file:
            json.dump(web_content, file, ensure_ascii=False )

        return web_content

    async def process_web(self, url: str, save_data_flag: bool=True):

        url_name = extract_name_from_url(url)
        try:
            with open(f"{self.json_data_path}/{url_name}.json", 'r', encoding='utf-8') as f:
                web_content = json.load(f)
            logger.info(f"成功读取本地JSON数据: {self.json_data_path}/{url_name}.json")
        except Exception as e:
            logger.warning(f"读取本地JSON失败，将重新提取: {e}")
            web_content = await self.get_web_content(url, url_name)

        token_count = self.spliter.count_tokens(', '.join(web_content))
        logger.info(f"@@@@@@ total token count: {token_count}")
        if token_count <= WEB_MAX_TOKEN_COUNT:
            self.output_path += f"/{url_name}"
            if not os.path.exists(self.output_path):
                summary_chain = await self.async_build_chain(self.chat_model, SYSTEM_PROMPT_CONFIG.get(ReaderRole.SUMMARY))
                query = f"请分析总结当前web页面的内容，按照文章本身的写作顺序给出详细的总结，{web_content}"
                summary = summary_chain.invoke(
                    {"input_prompt": query},
                    config={"configurable": {"session_id": "summary"}}
                )
                if save_data_flag:
                    makedir(self.output_path)
                    self.save_data_to_file(summary, url_name)
            self.web_content = web_content
        else:
            vector_db_path = os.path.join(f"{self.vector_db_path}/{url_name}_data_index")
            self.vector_db_obj = VectorDBClient(vector_db_path)

            if os.path.exists(vector_db_path):
                self.get_data_from_vector_db()
            else:
                chunks = self.spliter.split_text(", ".join(web_content))
                self.get_data_from_json_dict(chunks, chunks)

            if save_data_flag:
                self.generate_output_file(url_name, self.common_data_dict, self.raw_data_dict)

        logger.info(f"URL 处理流程结束。")

    def chat(self, input_prompt: str) -> Any:
        """
        针对用户输入进行对话。
        Interactive chat for user input.
        Args:
            input_prompt (str): 用户输入。
            last_retrieval_data (dict): 上一次检索的数据缓存。
        Returns:
            Any: 回答内容。
        """
        if self.web_content:
            answer = self.get_answer(self.web_content, input_prompt)
        else:
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
        return  answer

    async def main(self, url) -> None:
        await self.process_web(url)

        while True:
            user_input = input("You: ")

            if user_input.lower() in ["退出", "再见", "bye", "exit", "quit"]:
                print("Chatbot: 再见！期待下次与您对话。")
                break

            answer = self.chat(user_input)
            self.add_message_to_history(session_id="chat", message=AIMessage(answer))

            print(f"User: {user_input}")
            print(f"ChatBot: {answer}")
            print("======"*10)

if __name__=="__main__":
    web_reader_obj = WebReader()
    asyncio.run(web_reader_obj.main())
