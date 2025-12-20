from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from dotenv import load_dotenv

import asyncio
# Load the env variables
load_dotenv()

# Define a tool that searches the web for information.
# For simplicity, we will use a mock function here that returns a static string.
async def web_search(query: str) -> str:
    """Find information on the web"""
    return "AutoGen is a programming framework for building multi-agent applications."


# Create an agent that uses the OpenAI GPT-4o model.
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
agent = AssistantAgent(
    name="assistant",
    model_client=model_client,
    tools=[web_search],
    system_message="Use tools to solve tasks.",
    reflect_on_tool_use=True,
    model_client_stream=True,  
)

# Use asyncio.run(agent.run(...)) when running in a script.
async def main():
    #await Console(agent.run_stream(task="Find information on AutoGen"))
    result = await agent.run(task="Find information on AutoGen")
    print(result)
    # Close the connection to the model client.
    await model_client.close()
if __name__ == "__main__":
    asyncio.run(main())