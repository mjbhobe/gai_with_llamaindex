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

from llama_index.core.base.llms.types import ChatMessage
from llama_index.llms.groq import Groq

load_dotenv()

llm = Groq(
    model="llama3-70b-8192",
    temperature=0.0,
    max_tokens=2048,
)


def get_chat_response(user_message: str, chat_client=llm):
    messages = [
        ChatMessage(
            role="system",
            content="You are a helpful assistant who response in a calm and friendly tone",
        ),
        # ChatMessage(role="user", content="Tell me somethings about Sachin Tendulkar"),
        ChatMessage(role="user", content=user_message),
    ]
    response = llm.chat(messages)
    return response


# and ask a question
user_input = ""

while True:
    # infinite loop until user enters quit or exit or bye
    user_input = input("User input (or type quit or exit or bye to quit): ")
    user_input = user_input.strip()
    if user_input.lower() in ["quit", "exit", "bye"]:
        break

    response = get_chat_response(user_input)
    print(response)
    print("-" * 80 + "\n")
