"""
starter_llm.py - shows you how to use an LLM for plain Q&A
using the Llamaindex framework

NOTE: this example uses Open AI as it's LLM. For an example
with an opensource LLM, such as Groq, see starter_llm_groq.py

Author: Manish Bhobe
My experiments with Python, AI and Generative AI
Code is meant for learning purposes ONLY!
"""

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

# we will use OpenAI in this example
# You'll need to get an OpenAI API key from https://platform.openai.com/account/api-keys
# and save it to local .env file as OPENAI_API_KEY=sk-...
from llama_index.llms.openai import OpenAI

# load all API keys from .env file
load_dotenv()
console = Console()

# instantiate the LLM & ask a question
response = OpenAI(model="gpt-3.5-turbo").complete("The meaning of life is")
console.print("[green]The meaning of life is [/green]")
print(str(response))

# and with streaming
handle = OpenAI().stream_complete("Chattrapati Shivaji Maharaj is")
console.print("[blue]Chattrapati Shivaji Maharaj is [/blue]", end="")
for token in handle:
    print(token.delta, end="", flush=True)
print("\n")

# use with chat interface
from llama_index.core.base.llms.types import ChatMessage

user_message: str = "Tell me some hard facts about the US economy."
messages = [
    ChatMessage(role="system", content="You are a helpful assistant."),
    ChatMessage(role="user", content=user_message),
]
response = OpenAI("gpt-4o-mini").chat(messages)
console.print(f"[yellow]Chat: {user_message} [/yellow]")
console.print(Markdown(response.message.content))
