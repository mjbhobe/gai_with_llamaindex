""" 
basic_qa_with_llm.py - basic Q&A with an LLama3 LLM using 
Groq and llama3-70b-8192 model

Author: Manish Bhobe
My experiments with Python, AI and Generative AI
Code is meant for learning purposes ONLY!
"""

from dotenv import load_dotenv
from rich import print
from rich.console import Console
from rich.markdown import Markdown

# from llama_index.core.base.llms.types import ChatMessage
from llama_index.llms.groq import Groq

# load your API keys from .env file
# It should have entry for GROQ_API_KEY
load_dotenv()
console = Console()

# create a remote LLama3 LLM using Groq inference API
llm = Groq(
    model="llama3-70b-8192",
    temperature=0.0,
    max_tokens=2048,
)


def get_completion(part: str) -> str:
    response = llm.complete(part)
    return response.text


# now ask the LLM to complete the text

# text = ""
# while True:
#     # continuous loop, broken when user enters any one of exit|bye|quit
#     text = input("Enter completion (type exit or quit or bye to quit): ")
#     if text.strip().lower() in ["exit", "quit", "bye"]:
#         break
#     print(f"[{text}] {get_completion(text)}")

user_prompt = None
while True:
    # infinite loop
    console.print("[cyan]Ask me anything [/cyan][yellow](type exit to quit)[/yellow]: ")
    user_prompt = console.input("[green]").strip()
    if user_prompt.strip().lower() in ["bye", "quit", "exit"]:
        break
    console.print(Markdown(get_completion(user_prompt)))
