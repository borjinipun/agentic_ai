from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

from dotenv import load_dotenv
# Load the env variables
load_dotenv()

web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGoTools()],
    instructions="Always include sources",
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools()],
    instructions="Use tables to display data",
    markdown=True,
)

agent_team = Agent(
    name="Team Lead Agent",
    role="Coordinate web and finance research",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[web_agent, finance_agent],   # 👈 agents are tools
    instructions=[
        "Delegate research to the Web Agent",
        "Delegate financial data retrieval to the Finance Agent",
        "Always include sources",
        "Use tables where applicable"
    ],
    markdown=True,
)

# Give the team a task
agent_team.print_response("What's the market outlook and financial performance of SAIL stock in next 3 months?", stream=True)
