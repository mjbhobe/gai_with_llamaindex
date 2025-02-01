"""
phi_data_hello.py - simple program using PhiData, that searches
the web and responds to a question.

Author: Manish Bhobe
My experiments with Python, AI and Gen AI
Code shared for learning purposes only!
"""
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv

load_dotenv()

# define my agent that can search the web
web_agent = Agent (
    name = "Web Agent",
    model = Groq(id="deepseek-r1-distill-llama-70b", temperature=0.7, max_tokens=2043),
    tools = [DuckDuckGo()],
    instructions = ["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

# ask the agent something, will be returned from DuckDuckGo
web_agent.print_response("Tell me about Deepseek-R1", stream=True)
