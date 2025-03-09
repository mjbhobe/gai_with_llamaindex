"""
starter_llm_gemini.py - shows you how to use an LLM for plain Q&A
using the Llamaindex framework

NOTE: this example uses Google Gemini it's LLM. 
For OpenAI, see starter_llm.py
For Groq (with Llama), see starter_llm_groq.py

Author: Manish Bhobe
My experiments with Python, AI and Generative AI
Code is meant for learning purposes ONLY!
"""

import os
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

# You'll need to get a Gemini API key from Google AI Studio (aistudio.google.com/app/apikey)
# and save it to local .env file as GOOGLE_API_KEY=XXXXX....
from llama_index.llms.gemini import Gemini
import google.generativeai as genai

# load all API keys from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
console = Console()

# instantiate the LLM & ask a question
llm = Gemini(model="models/gemini-1.5-flash")
response = llm.complete("The meaning of life is")
console.print("[green]The meaning of life is [/green]")
print(str(response))

# and with streaming - NOTE: for Gemmini, we have to instantiate
# the LLM with streaming=True otherwise streaming DOES NOT WORK!!
llm2 = Gemini(model="models/gemini-pro", api_key=GOOGLE_API_KEY, streaming=True)
handle = llm2.stream_complete("Chattrapati Shivaji Maharaj is")
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
response = llm.chat(messages)
console.print(f"[yellow]Chat: {user_message} [/yellow]")
console.print(Markdown(response.message.content))
