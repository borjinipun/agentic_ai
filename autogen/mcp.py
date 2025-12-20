from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
import os
from dotenv import load_dotenv
import asyncio
# Load the env variables
load_dotenv()

# Get the fetch tool from mcp-server-fetch.
fetch_mcp_server = StdioServerParams(command="uvx", args=["mcp-server-fetch"])

async def main():
    # Create an MCP workbench which provides a session to the mcp server.
    async with McpWorkbench(fetch_mcp_server) as workbench:  # type: ignore
        # Create an agent that can use the fetch tool.
        model_client = OpenAIChatCompletionClient(
        model='llama-3.3-70b-versatile',
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
        # Define model capabilities (Groq models support function calling and JSON output)
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "structured_output": True,
            "family": 'llama-3.3-70b',
        }
    )
        fetch_agent = AssistantAgent(
            name="fetcher", model_client=model_client, workbench=workbench, reflect_on_tool_use=True
        )

        # Let the agent fetch the content of a URL and summarize it.
        result = await fetch_agent.run(task="Summarize the content of https://en.wikipedia.org/wiki/Seattle")
        assert isinstance(result.messages[-1], TextMessage)
        print(result.messages[-1].content)

        # Close the connection to the model client.
        await model_client.close()

asyncio.run(main())