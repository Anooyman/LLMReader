import asyncio
import logging

import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession

from llm import LLMBase
from src.common.utility import *

message_histories = {}


logging.basicConfig(
    level=logging.INFO,  # 可根据需要改为 DEBUG
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class MCPClient(LLMBase):

    def __init__(self, provider: str = "azure"):
        super().__init__(provider)
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.server_config = None
        self.session_dict = {}

    async def async_init(self, server_config: dict):
        self.server_config = server_config
        await self.connect_to_server()
        await self._process_tool()

    async def display_tools(self):
        """Display available tools with human-readable names"""
        tools_response = await self.session.list_tools()

        for tool in tools_response.tools:
            # get_display_name() returns the title if available, otherwise the name
            display_name = get_display_name(tool)
            print(f"Tool: {display_name}")
            if tool.description:
                print(f"   {tool.description}")

    async def display_resources(self):
        """Display available resources with human-readable names"""
        resources_response = await self.session.list_resources()

        for resource in resources_response.resources:
            display_name = get_display_name(resource)
            print(f"Resource: {display_name} ({resource.uri})")

    async def cleanup(self):
        try:
            await self.exit_stack.aclose()
            logger.info("Resources cleaned up successfully.")
        except Exception as e:
            logger.exception("Error during cleanup")

    async def connect_to_server(self):
        try:
            for name, config in self.server_config.items():
                connection_type = config.get("type", "")
                logger.info(f"Connecting to server with type: {connection_type}")

                try:
                    server_params = StdioServerParameters(
                        command=config.get("command"),
                        args=config.get("args"),
                        env=config.get("env"),
                    )
                    stdio_transport = await self.exit_stack.enter_async_context(
                        stdio_client(server_params))
                    stdio, write = stdio_transport
                    session = await self.exit_stack.enter_async_context(
                        ClientSession(stdio, write))

                    await session.initialize()
                    logger.info(f"Session {name} initialized successfully.")

                except Exception as e:

                    if connection_type == "sse":
                        streams = await self.exit_stack.enter_async_context(
                            sse_client(config.get("url")))
                        session = await self.exit_stack.enter_async_context(
                            ClientSession(*streams))
                        await session.initialize()
                        logger.info(f"Session {name} initialized successfully.")

                    elif connection_type == 'streamable-http':
                        read_stream, write_stream, _= await self.exit_stack.enter_async_context(
                            streamablehttp_client(config.get("url")))
                        session = await self.exit_stack.enter_async_context(
                            ClientSession(read_stream, write_stream))
                        await session.initialize()
                        logger.info(f"Session {name} initialized successfully.")

                    else:
                        logger.error(f"Error: undefined connection type {connection_type}")
                        return
                finally:
                    self.session_dict[name] = session

        except Exception as e:
            logger.exception("Failed to connect to server.")
            return 

    async def _process_tool(self):
        self.tools_name = []
        self.tool_descs = []
        self.tool_name_to_session = {}
        for name, session in self.session_dict.items():
            response = await session.list_tools()
            TOOL_DESC = "{name_for_model}: What is the {name_for_model} API useful for? {description_for_model}. Parameters: {parameters} Format the arguments as a JSON object."
            for tool in response.tools:
                self.tool_name_to_session[tool.name] = (name, session)
                self.tools_name.append(tool.name)
                self.tool_descs.append(
                    TOOL_DESC.format(
                    name_for_model = tool.name,
                    description_for_model = tool.description,
                    parameters = tool.inputSchema
                    )
                )

if __name__ == "__main__":
    ...
