"""
chat_with_llm.py: basic chatbot with an LLM

This examples shows how you can implement a basic chatbot 
with an LLM. The LLM will respond based on data on which 
it is pretrained.
This is basically like a mini-chatgpt (with chat history!)
Built with llama_index and Groq (LLama3 model)

Author: Manish Bhobe
My experiments with Python, ML, and Generative AI.
Code is provided as-is and meant for learning only!
"""

import os
from dotenv import load_dotenv
from textwrap import dedent
from rich.console import Console
from rich.markdown import Markdown

from llama_index.core.base.llms.types import ChatMessage
from llama_index.llms.groq import Groq

load_dotenv()

llm = Groq(
    # model="llama3-70b-8192",
    model="deepseek-r1-distill-llama-70b",
    temperature=0.0,
    max_tokens=2048,
)

console = Console()

# let's give this chatbot some Mumbai flair - the chatbot will respond
# like a Mumbai local, with all local slang & lingo.

system_prompt = dedent(
    """
    - Think of yourself as an enthusiastic assistant, ready to help you with any questions you have.
    - You have deep knowledge about the world, and about Mumbai in particular.
                       
    Instructions:
    - You are a local from Mumbai, India, who is proficient in English as well as local slang.
    - Don't limit your responses to questions about Mumbai as your knowledge is NOT limited to Mumbai 
      alone. Answer any question from the user.
    - Use casual English in your response, but throw in Mumbai slang words (such as "fundu", "jugaad",
      "bawa", "bole to", "gyaan", "aapunki" etc.) - it will make you more relatable. Add a meaning of the Mumbai slang in brackets the first time you use it in a conversation, so non-Mumbai folks can
      understand aapunki bhaasha (slang for "our lingo").
    - Don't start all your responses with "Ayy" - use some variety, your responses need not always sound
      like a local "tapori" (slang for a "street thug").
    - Use any tools provided to you only if you cannot answer the question directly. Don't use tools for 
      every question.
    - If you cannot answer a question, say so, and don't try to fake it. Apologize in classic Mumbai 
      style.
"""
)


def get_chat_response(user_message: str, chat_client=llm):
    messages = [
        ChatMessage(
            role="system",
            # content="You are a helpful assistant who responds in a calm and friendly tone",
            content=system_prompt,
        ),
        ChatMessage(role="user", content=user_message),
    ]
    response = chat_client.chat(messages)
    return response


# and ask a question
# user_input = ""

# while True:
#     # infinite loop until user enters quit or exit or bye
#     user_input = input("User input (or type quit or exit or bye to quit): ")
#     user_input = user_input.strip()
#     if user_input.lower() in ["quit", "exit", "bye"]:
#         break

#     response = get_chat_response(user_input)
#     print(response)
#     print("-" * 80 + "\n")

user_prompt = None
while True:
    # infinite loop
    console.print("[cyan]Ask me anything [/cyan][yellow](type exit to quit)[/yellow]: ")
    user_prompt = console.input("[green]").strip()
    if user_prompt.strip().lower() in ["bye", "quit", "exit"]:
        break
    console.print(Markdown(str(get_chat_response(user_prompt))))
