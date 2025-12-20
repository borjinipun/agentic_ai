from agno.agent import Agent 
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools 
from dotenv import load_dotenv
# Load the env variables
load_dotenv()


llm = Groq(id="LLama-3.3-70b-versatile", temperature=0)

# Initialize the agent with Groq provided LLM
agent = Agent(
    model=llm,
    description="You are a diligent research assistant skilled at gathering, analyzing and "
    "summarizing information form multiple sources",
    tools=[DuckDuckGoTools()],
    markdown=True
    )
# prompt the agent to perform a research task
agent.print_response ("Research the impact of Mexico 50 percent tariff on India in 2025. Provide a summary with references to your sources.")