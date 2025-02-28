"""
phi_data_finance.py - agentic example of analyzing stocks using
    Yahoo Finance! API

Author: Manish Bhobe
My experiments with Python, AI and Gen AI
Code shared for learning purposes only!
"""

from phi.agent import Agent
from phi.model.groq import Groq
from phi.model.openai import OpenAIChat
from phi.model.google import Gemini
from phi.model.anthropic import Claude
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv

load_dotenv()


def get_symbol_from_name(name: str) -> str:
    """get the Yahoo Finance! stock symbol given the company name"""
    name_symbol_map = {
        # some popular stocks on NSE
        "Reliance Industries Limited": "RELIANCE.NS",
        "Pidilite Industries Limited": "PIDILITIND.NS",
        "Bajaj Auto": "BAJAJ-AUTO.NS",
        "Dixon": "DIXON.NS",
    }

    return name_symbol_map.get(name, "Unknown")


# define my agent that can search the web
"""
OpenAI works best (as expected)
Anthropic Claude (works well, almost as good as OpenAI)
Google Gemini (works well, not as good as OpenAI or Claude)
Llama3/Groq 
Deepseek-R1/Groq (does not work!)
"""
finance_agent = Agent(
    name="Finance Agent",
    # model=Groq(model="llama-3.3-70b-versatile", temperature=0.0, max_tokens=2043),
    # model=Groq(id="deepseek-r1-distill-llama-70b", temperature=0.0, max_tokens=2043),
    # model=OpenAIChat(id="gpt-4o",temperature=0.0, max_tokens=2043),
    # model=Gemini(id="gemini-1.5-flash", show_tool_calls=True),
    model=Claude(id="claude-3-5-sonnet-20240620", temperature=0.0, max_tokens=2043),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            company_info=True,
            company_news=True,
        ),
    ],
    instructions=[
        "Use tables to display data",
        "Format all numbers using India locale formatting",
        "If you do not know the company symbol use get_symbol_from_name tool, even if it is not a public company",
    ],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
)

# ask the agent something, will be returned from DuckDuckGo
finance_agent.print_response(
    "Summarize and compare fundamentals and analyst recommendations for RELIANCE.NS and PIDILITIND.NS",
    stream=True,
)
