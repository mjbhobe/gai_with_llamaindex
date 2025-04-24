""" 
basic_qa_with_llm.py - basic Q&A with an LLama3 LLM using 
Groq and llama3-70b-8192 model

Author: Manish Bhobe
My experiments with Python, AI and Generative AI
Code is meant for learning purposes ONLY!
"""

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from llama_index.llms.groq import Groq

# load your API keys from .env file
# It should have entry for GROQ_API_KEY (create a new one at https://console.groq.com/keys)
load_dotenv()
console = Console()

# create a remote LLama3 LLM using Groq inference API
llm = Groq(
    model="llama3-70b-8192",
    temperature=0.0,
    max_tokens=2048,
)


def get_completion(prompt: str) -> str:
    """utility function to pass a prompt to LLM & get a response
    Args:
        prompt (str) - the query to your LLM
    Returns:
        response (str) - text response from LLM (response.content)
    """
    response = llm.complete(prompt)
    return response.text


# now ask the LLM to complete the text

user_prompt = None
while True:
    # infinite loop, broken when user types any of ["bye", "quit", "exit"]
    console.print("[cyan]Ask me anything [/cyan][yellow](type exit to quit)[/yellow]: ")
    user_prompt = console.input("[green]").strip()
    if user_prompt.strip().lower() in ["bye", "quit", "exit"]:
        break
    response = llm.complete(user_prompt)
    console.print(Markdown(response.text))
    # console.print(Markdown(get_completion(user_prompt)))