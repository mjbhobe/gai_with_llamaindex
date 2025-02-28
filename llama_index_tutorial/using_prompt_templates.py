"""
using_prompt_templates.py - using prompt templates with LlamaIndex

This example shows you how to use LlamaIndex's PromptTemplates to answer
questions from a grounded context. The grounded context here is a text
file (President Donald Trump's inaugural speech 2025), from which we
will ask LLM to answer questions. We'll continue to use Groq hosted Llama3 LLM

Author: Manish Bhobe
My experiments with Python, ML, and Generative AI.
Code is provided as-is and meant for learning only!
"""

import pathlib
from dotenv import load_dotenv
from textwrap import fill  # to wrap text response to certain width

from llama_index.core import PromptTemplate
from llama_index.llms.groq import Groq


TEXT_WIDTH = 80  # wrap to these many columns

# load all keys from .env file
# your .env file must have a GROQ_API_KEY=<<groq_api_key_value>> entry
# to create one visit: https://console.groq.com/keys
load_dotenv()

# instantiate remote LLM on Groq Cloud
# for list of available LLMs visit https://console.groq.com/playground and look
# for list of models in the dropdown to select model
llm = Groq(
    model="llama3-70b-8192",
    temperature=0.0,
    max_tokens=2048,
)

# my prompt template
template = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
)
qa_template = PromptTemplate(template)


# load the presidential address from local folder - this will provide the
# context {context_str} in the above prompt
speech_path = pathlib.Path(__file__).parent / "docs" / "trump_inaugural_speech.txt"
assert speech_path.exists(), f"FATAL: {speech_path} - file does not exist!"
with open(str(speech_path), "r") as f:
    speech = f.read()


def get_prompt_response(user_prompt: str, context: str = speech) -> str:
    """utility function to get response from LLM using a prompt template"""
    # you can create text prompt (for completion API)
    prompt = qa_template.format(context_str=context, query_str=user_prompt)
    # ask llm to respond
    response = llm.complete(prompt)
    # wrapping text response to TEXT_WIDTH width for convenience
    # the wrapping is not actually required
    return fill(response.text, width=TEXT_WIDTH)


# and ask a question from the speech
user_input = ""

while True:
    # infinite loop until user enters quit or exit or bye
    user_input = input("User input (or type quit or exit or bye to quit): ")
    user_input = user_input.strip()
    if user_input.lower() in ["quit", "exit", "bye"]:
        break

    response = get_prompt_response(user_input)
    print(response)
    print("-" * TEXT_WIDTH + "\n")
