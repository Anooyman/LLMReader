import logging

import tiktoken
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from pydantic import Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

from abc import ABC, abstractmethod

from config import (
    LLM_CONFIG,
    LLM_EMBEDDING_CONFIG,
)
logging.basicConfig(
    level=logging.INFO,  # 可根据需要改为 DEBUG
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

from typing import Optional

class LimitedChatMessageHistory(InMemoryChatMessageHistory):
    max_messages: int = Field(default=20)
    max_tokens: int = Field(default=32768)
    encoding_name: Optional[str] = Field(default="cl100k_base")  # 你可以根据模型换成合适的encoding

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_tokens = kwargs.get("max_tokens", 32768)
        self.encoding_name = kwargs.get("encoding_name", "cl100k_base")
        # 你可以在这里做额外的初始化

    def _count_tokens(self, message):
        # 这里用 tiktoken 统计 token 数
        try:
            encoding = tiktoken.get_encoding(self.encoding_name)
            if hasattr(message, "content"):
                return len(encoding.encode(message.content))
            else:
                return 0
        except ImportError:
            # 没装 tiktoken 就简单估算
            if hasattr(message, "content"):
                return len(message.content) // 4
            else:
                return 0

    def _total_tokens(self):
        return sum(self._count_tokens(m) for m in self.messages)

    def add_message(self, message):
        super().add_message(message)
        # 限制消息条数
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
        # 限制 token 总数
        while self._total_tokens() > self.max_tokens and len(self.messages) > 1:
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
    """
    def __init__(self, provider: str) -> None:
        self.message_histories = {}
        self.provider = provider.lower()
        self.providers = {
            "azure": AzureLLMProvider(),
            "openai": OpenAILLMProvider(),
            "ollama": OllamaLLMProvider(),
        }
        # 兼容原有 azure 默认
        self.chat_model = self.get_chat_model()
        self.embbeding_model = self.get_embedding_model()

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
            raise ValueError(f"Unknown provider: {self.provider}")
        return self.providers[self.provider].get_chat_model(**kwargs)

    def get_message_history(self, session_id=None):
        # 根据 session_id 获取对应的对话历史
        if session_id not in self.message_histories:
            if session_id in ["chat"]:
                self.message_histories[session_id] = LimitedChatMessageHistory()
            else:
                self.message_histories[session_id] = LimitedChatMessageHistory(max_messages=3)
        return self.message_histories[session_id]

    def add_message_to_history(self, session_id=None, message=None):
        if message is None:
            message = HumanMessage("")  # 或 SystemMessage("")，根据你的业务场景
        if session_id not in self.message_histories:
            logger.warning(f"Can't find {session_id}, in current history. Create a new history.")
            if session_id in ["chat"]:
                self.message_histories[session_id] = LimitedChatMessageHistory()
            else:
                self.message_histories[session_id] = LimitedChatMessageHistory(max_messages=3)
        self.message_histories[session_id].add_message(message)

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
        return self.build_chain(
            client=client,
            system_prompt=system_prompt,
            output_parser=output_parser,
            tools=tools,
        )

    def get_chat_model(self, **kwargs):
        if self.provider not in self.providers:
            raise ValueError(f"Unknown provider: {self.provider}")
        return self.providers[self.provider].get_chat_model(**kwargs)

    def get_embedding_model(self, **kwargs):
        if self.provider not in self.providers:
            raise ValueError(f"Unknown provider: {self.provider}")
        return self.providers[self.provider].get_embedding_model(**kwargs)