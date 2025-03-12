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

import os, sys
import time
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

console.print("[cyan]List of Google Gemini available models[/cyan]")
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        console.print(Markdown(f"* {m.name}"))

# instantiate the LLM & ask a question
GEMINI_MODEL_NAME = "models/gemini-1.5-pro"
console.print(f"[red]Using Gemini model {GEMINI_MODEL_NAME}[/red]")
llm = Gemini(model=GEMINI_MODEL_NAME)
response = llm.complete("The meaning of life is")
console.print("[green]The meaning of life is [/green]")
console.print(Markdown(str(response)))
print("=" * 80)

# and with streaming - NOTE: for Gemmini, we have to instantiate
# the LLM with streaming=True otherwise streaming DOES NOT WORK!!
llm2 = Gemini(model=GEMINI_MODEL_NAME, api_key=GOOGLE_API_KEY, streaming=True)
handle = llm2.stream_complete("Chattrapati Shivaji Maharaj is")
console.print("[blue]Chattrapati Shivaji Maharaj is [/blue]", end="")
markdown_text = ""
for token in handle:
    # print(token.delta, end="", flush=True)
    # NOTE: while the above print(...) call works with OpenAI, it fails with
    # Gemini. For Gemini, we must use token.text instead of token.delta
    # print(token.text, end="", flush=True)
    markdown_text += token.text
    if len(markdown_text) % 100 == 0 or "\n" in token.text:
        console.clear()
        console.print(Markdown(markdown_text))
        time.sleep(0.1)  # Optional: Add a small delay for better visual effect

print("=" * 80)
sys.exit(-1)

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
