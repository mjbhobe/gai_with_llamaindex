"""
qna_with_llm.py: basic Q&A with an LLM

This examples shows how you can instantiate an LLM
and ask questions to it. The LLM will respond based on
data on which it is pretrained.
This is basically like a mini-chatgpt (with no chat history!)
Built with llama_index

Author: Manish Bhobe
My experiments with Python, ML, and Generative AI.
Code is provided as-is and meant for learning only!

"""

from dotenv import load_dotenv
from llama_index.llms.groq import Groq

# load all keys from .env file
# your .env file must have a GROQ_API_KEY=<<groq_api_key_value>> entry
# to create one visit: https://console.groq.com/keys
load_dotenv()

# instantiate remote LLM on Groq Cloud
# for list of available LLMs visit https://console.groq.com/playground and look
# for list of models in the dropdown to select model
llm = Groq(
    # model="llama3-70b-8192",
    model="deepseek-r1-distill-llama-70b",
    temperature=0.0,
    max_tokens=2048,
)

# and ask a question
user_input = ""

while True:
    # infinite loop until user enters quit or exit or bye
    user_input = input("User input (or type quit or exit or bye to quit): ")
    user_input = user_input.strip()
    if user_input.lower() in ["quit", "exit", "bye"]:
        break

    response = llm.complete(user_input)
    print(response.text)
    print("-" * 80 + "\n")
