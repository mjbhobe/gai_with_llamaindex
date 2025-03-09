"""
starter_agents_stream_and_events.py - agent workflows with user feedback

NOTE: this example uses Open AI as it's LLM. But you can replace it
with an LLM of your choice, such as Gemini or Anthropic or Groq

Author: Manish Bhobe
My experiments with Python, AI and Generative AI
Code is meant for learning purposes ONLY!
"""

import os
import asyncio
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

# we will use OpenAI in this example
# You'll need to get an OpenAI API key from https://platform.openai.com/account/api-keys
# and save it to local .env file as OPENAI_API_KEY=sk-...
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    AgentInput,
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStream,
)
from llama_index.tools.tavily_research import TavilyToolSpec

# load all API keys from .env file
load_dotenv()
console = Console()

# my list of tools
# instantiate Tavily search tools
tavily_tools = TavilyToolSpec(api_key=os.getenv("TAVILY_API_KEY"))


# instantiate the LLM & ask a question
llm = OpenAI(model="gpt-3.5-turbo")


# setup a workflow that will call agents as needed
workflow = AgentWorkflow.from_tools_or_functions(
    tools_or_functions=tavily_tools.to_tool_list(),
    llm=llm,
    system_prompt=(
        """
        You are a helpful assistant that can search the web for information.
        """
    ),
)


async def main():
    user_prompt = """
        What's the weather like today and for the next 5 days in Panjim, Goa, India.
    """
    handler = workflow.run(user_msg=user_prompt)
    # handle events as they come
    async for event in handler.stream_events():
        if isinstance(event, AgentStream):
            # print each agent stream as we receive it
            print(event.delta, end="", flush=True)
        elif isinstance(event, AgentInput):
            print("Agent input: ", event.input)  # the current input messages
            print("Agent name:", event.current_agent_name)  # the current agent name
        elif isinstance(event, AgentOutput):
            print("Agent output: ", event.response)  # the current full response
            print(
                "Tool calls made: ", event.tool_calls
            )  # the selected tool calls, if any
            print("Raw LLM response: ", event.raw)  # the raw llm api response
        elif isinstance(event, ToolCallResult):
            print("Tool called: ", event.tool_name)  # the tool name
            print("Arguments to the tool: ", event.tool_kwargs)  # the tool kwargs
            print("Tool output: ", event.tool_output)  # the tool output

    # print final output
    print(str(await handler))


if __name__ == "__main__":
    asyncio.run(main())
