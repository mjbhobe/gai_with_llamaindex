"""
starter_agents.py - building a simple agent workflow using Llamaindex

NOTE: this example uses Open AI as it's LLM. But you can replace it
with an LLM of your choice, such as Gemini or Anthropic or Groq

Author: Manish Bhobe
My experiments with Python, AI and Generative AI
Code is meant for learning purposes ONLY!
"""

import asyncio
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

# we will use OpenAI in this example
# You'll need to get an OpenAI API key from https://platform.openai.com/account/api-keys
# and save it to local .env file as OPENAI_API_KEY=sk-...
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec

# load all API keys from .env file
load_dotenv()
console = Console()


# here we define 2 simple functions that will serve as tools that
# the LLM can use.
# NOTE: the LLM will use the docstring, so be as expressive as possible
# when writing your docstring
def multiply(a: float, b: float) -> float:
    """multiply 2 numbers and return the product"""
    console.print(f"[red]The multiply tool was called with a: {a} and b: {b}[/red]")
    return a * b


def add(a: float, b: float) -> float:
    """add 2 numbers and return the sum"""
    console.print(f"[red]The add tool was called with a: {a} and b: {b}[/red]")
    return a + b


# instantiate the LLM & ask a question
llm = OpenAI(model="gpt-3.5-turbo")

# my list of tools
finance_tools = YahooFinanceToolSpec().to_tool_list()
finance_tools.extend([multiply, add])

# setup a workflow that will call agents as needed
workflow = AgentWorkflow.from_tools_or_functions(
    tools_or_functions=finance_tools,
    llm=llm,
    system_prompt=(
        """
        You are an ISDA certified financial analyst who can perform basic mathematical operations 
        using tools that are provided to you.
        You should ALWAYS use the tools provided to you even if you are capable of peforming the task
        on your own.
        You will return structured data, where available. All numbers must be formatted
        using the India locale
        """
    ),
)


async def main():
    # user_prompt = """
    #     Sachin Tendulkar is a fantastic cricketer from India. Can you tell me about his stats.
    #     Suppose he scores 65 each in 3 innings, what is his total score? Then he adds anther
    #     75 in the next innings, what is the grand total of his score?
    # """
    user_prompt = """
        What is the current price of Pidilite Industries stock. 
        If I add 500 to it and multiply that by 2.5 what will I get?
        \n
        What are the analysts saying about the stock's performance
        \n
        Fetch me the latest 10 headlines about Pidilite and always show your sources.
    """
    response = await workflow.run(user_msg=user_prompt)
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
