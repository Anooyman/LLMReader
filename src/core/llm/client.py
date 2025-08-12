"""
client.py - LLM provider and message history management for LLMReader

This module provides classes for managing chat message history with limits, and for abstracting over different LLM providers (Azure, OpenAI, Ollama).
"""
import logging
import tiktoken
from typing import Any, Optional
from pydantic import Field
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

from abc import ABC, abstractmethod

from src.config.settings import (
    LLM_CONFIG,
    LLM_EMBEDDING_CONFIG,
    SYSTEM_PROMPT_CONFIG,
)
logging.basicConfig(
    level=logging.INFO,  # 可根据需要改为 DEBUG
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class LimitedChatMessageHistory(InMemoryChatMessageHistory):
    """
    带有限制功能的聊天消息历史记录管理类

    注意：虽然 max_messages, max_tokens, encoding_name 使用了 Pydantic 的 Field，
    但本类并非 Pydantic BaseModel，Field 仅用于类型提示和默认值说明，实际初始化请看 __init__。

    扩展InMemoryChatMessageHistory，增加以下功能：
    - 消息数量限制：通过max_messages参数控制最大消息条数
    - Token数量限制：通过max_tokens参数控制总Token数不超过模型上下文窗口
    - 自动清理：当消息数量或Token数超出限制时，自动移除最早的消息

    Attributes:
        max_messages (int): 最大消息数量限制，默认20
        max_tokens (int): 最大Token数量限制，默认32768
        encoding_name (str): Token编码名称，默认使用o200k_base
    """
    max_messages: int = Field(default=20)
    max_tokens: int = Field(default=32768)
    encoding_name: Optional[str] = Field(default="o200k_base")  # 你可以根据模型换成合适的encoding

    def __init__(self, **kwargs):
        # NOTE: These are not Pydantic fields, just for type hinting/default doc
        super().__init__(**kwargs)
        self.max_messages = kwargs.get("max_messages", 20)
        self.max_tokens = kwargs.get("max_tokens", 32768)
        self.encoding_name = kwargs.get("encoding_name", "o200k_base")
        # 你可以在这里做额外的初始化

    def _count_tokens(self, message):
        """
        计算单条消息的Token数量
        Args:
            message: 聊天消息对象，需包含content属性
        Returns:
            int: 消息内容的Token数量
        Note:
            优先使用tiktoken进行精确计算，如未安装则使用字符数/4进行估算
        """
        try:
            import tiktoken
            encoding = tiktoken.get_encoding(self.encoding_name)
            if hasattr(message, "content"):
                return len(encoding.encode(message.content))
            else:
                return 0
        except ImportError:
            logger.warning("tiktoken not installed, using rough token estimate.")
            if hasattr(message, "content"):
                return len(message.content) // 4
            else:
                return 0
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            return 0

    def _total_tokens(self):
        """计算所有消息的总Token数"""
        return sum(self._count_tokens(m) for m in self.messages)

    def add_message(self, message):
        """
        添加消息到历史，并自动根据 max_messages 和 max_tokens 进行裁剪。
        """
        super().add_message(message)
        # 1. 限制消息条数 - 保留最新的max_messages条消息
        if len(self.messages) > self.max_messages:
            logger.info(f"[LimitedChatMessageHistory] 消息数量超出限制({self.max_messages})，已截断。")
            self.messages = self.messages[-self.max_messages:]
        # 2. 限制Token总数 - 循环移除最早消息直到Token数达标
        while self._total_tokens() > self.max_tokens and len(self.messages) > 1:
            logger.info(f"[LimitedChatMessageHistory] Token总数超出限制({self.max_tokens})，移除最早消息。")
            self.messages.pop(0)


class LLMProviderBase(ABC):
    """
    LLM Provider 抽象基类，定义统一接口。
    """
    @abstractmethod
    def get_chat_model(self, **kwargs):
        pass

    @abstractmethod
    def get_embedding_model(self, **kwargs):
        pass

class AzureLLMProvider(LLMProviderBase):

    def get_chat_model(self, **kwargs):
        return AzureChatOpenAI(
            openai_api_key=kwargs.get("openai_api_key", LLM_CONFIG.get("api_key")),
            openai_api_version=kwargs.get("openai_api_version", LLM_CONFIG.get("api_version")),
            azure_endpoint=kwargs.get("azure_endpoint", LLM_CONFIG.get("azure_endpoint")),
            deployment_name=kwargs.get("deployment_name", LLM_CONFIG.get("deployment_name")),
            model_name=kwargs.get("model_name", LLM_CONFIG.get("model_name")),
            temperature=kwargs.get("temperature", 0.7),
            max_retries=kwargs.get("max_retries", 5)
        )

    def get_embedding_model(self, **kwargs):
        return AzureOpenAIEmbeddings(
            openai_api_key=kwargs.get("openai_api_key", LLM_EMBEDDING_CONFIG.get("api_key")),
            openai_api_version=kwargs.get("openai_api_version", LLM_EMBEDDING_CONFIG.get("api_version")),
            azure_endpoint=kwargs.get("azure_endpoint", LLM_EMBEDDING_CONFIG.get("azure_endpoint")),
            deployment=kwargs.get("deployment", LLM_EMBEDDING_CONFIG.get("deployment")),
            model=kwargs.get("model", LLM_EMBEDDING_CONFIG.get("model")),
            max_retries=kwargs.get("max_retries", 5)
        )

class OpenAILLMProvider(LLMProviderBase):
    def get_chat_model(self, **kwargs):
        return ChatOpenAI(
            openai_api_key=kwargs.get("openai_api_key", LLM_CONFIG.get("openai_api_key")),
            model_name=kwargs.get("model_name", LLM_CONFIG.get("openai_model_name", "gpt-3.5-turbo")),
            temperature=kwargs.get("temperature", 0.7),
            max_retries=kwargs.get("max_retries", 5)
        )

    def get_embedding_model(self, **kwargs):
        return OpenAIEmbeddings(
            openai_api_key=kwargs.get("openai_api_key", LLM_EMBEDDING_CONFIG.get("openai_api_key")),
            model=kwargs.get("model", LLM_EMBEDDING_CONFIG.get("openai_model", "text-embedding-ada-002")),
            max_retries=kwargs.get("max_retries", 5)
        )

class OllamaLLMProvider(LLMProviderBase):
    def get_chat_model(self, **kwargs):
        return ChatOllama(
            base_url=kwargs.get("base_url", LLM_CONFIG.get("ollama_base_url", "http://localhost:11434")),
            model=kwargs.get("model", LLM_CONFIG.get("ollama_model_name", "llama3")),
            temperature=kwargs.get("temperature", 0.7)
        )

    def get_embedding_model(self, **kwargs):
        return OllamaEmbeddings(
            base_url=kwargs.get("base_url", LLM_EMBEDDING_CONFIG.get("ollama_base_url", "http://localhost:11434")),
            model=kwargs.get("model", LLM_EMBEDDING_CONFIG.get("ollama_model", "llama3")),
        )

class LLMBase:
    """
    LLMBase 统一调度各类 LLMProvider。
    管理多会话历史，支持不同 LLM provider。
    """
    def __init__(self, provider: str) -> None:
        """
        Args:
            provider (str): 'azure', 'openai', 'ollama'
        """
        self.message_histories = {}
        self.provider = provider.lower()
        self.providers = {
            "azure": AzureLLMProvider(),
            "openai": OpenAILLMProvider(),
            "ollama": OllamaLLMProvider(),
        }
        # 兼容原有 azure 默认
        self.chat_model = self.get_chat_model()
        self.embedding_model = self.get_embedding_model()

    def _customize_mode(self, **kwargs):
        """
        根据 provider 和参数动态创建对应的 chat model。
        Args:
            provider (str): 'azure'（默认）、'openai'、'ollama'。
            其他参数通过 kwargs 传递。
        Returns:
            对应 provider 的 chat model 实例。
        """
        if self.provider not in self.providers:
            logger.error(f"Unknown provider: {self.provider}")
            raise ValueError(f"Unknown provider: {self.provider}")
        return self.providers[self.provider].get_chat_model(**kwargs)

    def get_message_history(self, session_id=None):
        """
        获取指定 session_id 的消息历史，没有则自动创建。
        """
        if session_id not in self.message_histories:
            if session_id in ["chat"]:
                self.message_histories[session_id] = LimitedChatMessageHistory()
            else:
                self.message_histories[session_id] = LimitedChatMessageHistory(max_messages=3)
            logger.info(f"[get_message_history] 新建 session_id: {session_id}")
        return self.message_histories[session_id]

    def add_message_to_history(self, session_id=None, message=None):
        """
        向指定 session_id 的历史添加消息。
        """
        if message is None:
            message = HumanMessage("")  # 或 SystemMessage("")，根据你的业务场景
        if session_id not in self.message_histories:
            logger.warning(f"Can't find {session_id}, in current history. Create a new history.")
            if session_id in ["chat"]:
                self.message_histories[session_id] = LimitedChatMessageHistory()
            else:
                self.message_histories[session_id] = LimitedChatMessageHistory(max_messages=3)
        self.message_histories[session_id].add_message(message)
        logger.debug(f"[add_message_to_history] session_id={session_id}, message={message}")

    def is_content_in_history(self, content, session_id=None, exact_match=False):
        """
        判断 content 是否在 session_id 的历史消息中出现过。
        Args:
            content (str): 要查找的内容。
            session_id (Any): 会话ID。
            exact_match (bool): 是否要求完全匹配（默认False，表示只要包含即可）。
        Returns:
            bool: True 表示找到匹配内容，False 表示未找到。
        """
        history = self.get_message_history(session_id)
        for idx, msg in enumerate(history.messages):
            if hasattr(msg, "content"):
                if exact_match:
                    if msg.content == content:
                        logger.info(f"[is_content_in_history] 完全匹配成功，索引: {idx}")
                        return True
                else:
                    if content in msg.content:
                        logger.info(f"[is_content_in_history] 包含关系匹配成功，索引: {idx}")
                        return True
        logger.info("[is_content_in_history] 未找到匹配内容。")
        return False

    def build_chain(
        self,
        client: AzureChatOpenAI,
        system_prompt: str = "",
        output_parser=None,
        tools=None,
    ):
        """
        构建带有 system_prompt、tools、session_id 以及可选 output_format 的对话链。
        output_format: 可选，字符串，指定输出格式说明，会拼接到 system_prompt 后面。
        """
        # 1. output_parser 默认用 StrOutputParser，如果没传入的话
        if not output_parser:
            output_parser = StrOutputParser()
        # 2. 构建 prompt，包含 system prompt、历史消息和用户输入
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input_prompt}"),
        ])
        # 3. 返回带有历史消息管理的 RunnableWithMessageHistory
        return RunnableWithMessageHistory(
            (prompt | client | output_parser).with_config(tool=tools),
            self.get_message_history,
            input_messages_key="input_prompt",
            history_messages_key="chat_history"
        )

    async def async_build_chain(
        self,
        client: AzureChatOpenAI,
        system_prompt: str = "",
        output_parser=None,
        tools=None,
    ):
        """
        异步构建对话链（实际调用同步 build_chain）。
        """
        return self.build_chain(
            client=client,
            system_prompt=system_prompt,
            output_parser=output_parser,
            tools=tools,
        )

    def get_chat_model(self, **kwargs):
        """
        获取当前 provider 的 chat model。
        """
        if self.provider not in self.providers:
            logger.error(f"Unknown provider: {self.provider}")
            raise ValueError(f"Unknown provider: {self.provider}")
        return self.providers[self.provider].get_chat_model(**kwargs)

    def get_embedding_model(self, **kwargs):
        """
        获取当前 provider 的 embedding model。
        """
        if self.provider not in self.providers:
            logger.error(f"Unknown provider: {self.provider}")
            raise ValueError(f"Unknown provider: {self.provider}")
        return self.providers[self.provider].get_embedding_model(**kwargs)

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
            try:
                system_prompt = system_prompt.format(**system_format_dict)
            except KeyError as e:
                logger.error(f"系统提示词格式化失败，缺少参数: {e}")
        chain = self.build_chain(
            client=self.chat_model,
            system_prompt=system_prompt,
            output_parser=output_parser
        )
        try:
            response = chain.invoke(
                {"input_prompt": input_prompt},
                config={"configurable": {"session_id": session_id}}
            )
            logger.info(f"LLM调用完成: role={role}, session_id={session_id}")
        except Exception as e:
            logger.error(f"{role} invoke llm 报错，错误如下：{e}")
            # TODO: 可根据业务需求决定是否抛出异常或返回特定错误对象
            response = ""
        return response

