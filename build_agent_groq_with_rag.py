"""
build_agent_groq_with_rag.py: example of building simple agents that use LLama2 LLM 
    accessed via Groq API with RAG.
    For RAG, we used Wikipedia's page on the 2023 Canadian Federal Budget, saved as PDF
    in the ./docs folder

Author: Manish Bhobe
My experiments with Python, ML and Generative AI with llamaindex.
Code is meant for illustration purposes ONLY. Use at your own risk!
Author is not liable for any damages arising from direct/indirect use of this code.
"""

import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

# load all API keys from .env file
# your .env file must have a GROQ_API_KEY entry
load_dotenv()

# NOTE: by default llamaindex uses OpenAI. In this program we'll use Groq API with llama3-70b-8192 instead
# @see: https://docs.llamaindex.ai/en/stable/examples/cookbooks/llama3_cookbook_groq/ for more examples
from llama_index.llms.groq import Groq

# create the LLM & embeddings
llm = Groq(model="llama3-70b-8192", temperature=0.0)
# llm = Groq(model="deepseek-r1-distill-llama-70b", temperature=0.0)

# and we'll use Jina embeddings instead of OpenAI embeddings, which may not be compatible with
# our Llama model
# @see: https://docs.llamaindex.ai/en/stable/examples/embeddings/jinaai_embeddings/
from llama_index.embeddings.jinaai import JinaEmbedding

# NOTE: your .env file must have a JINAAI_API_KEY entry
embed_model = JinaEmbedding(
    api_key=os.environ["JINAAI_API_KEY"],
    model="jina-embeddings-v3",
)

# and we MUST add the LLM to our globals, so that llamaindex uses them
# instead of the default OpenAI ones.
from llama_index.core import Settings

Settings.llm = llm
Settings.embed_model = embed_model

# now llamaindex will use our LLM instead of default OpenAI

# vectorize all documents in the docs folder
documents = SimpleDirectoryReader("./docs").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# quick sanity check
response = query_engine.query(
    "What was the total amount of the 2023 Canadian federal budget?"
)
print(response)

# # create the basic tools - we'll use 2 tools that call simple Python functions
# def multiply(a: float, b: float) -> float:
#     """multiply two numbers and returns the product"""
#     return a * b


# multiply_tool = FunctionTool.from_defaults(fn=multiply)


# def add(a: float, b: float) -> float:
#     """Add two numbers and return the sum"""
#     return a + b


# add_tool = FunctionTool.from_defaults(fn=add)

# # create our agent that uses the 2 tools we just created
# agent = ReActAgent.from_tools([multiply_tool, add_tool], llm=llm, verbose=True)


# # now ask a question
# print("-" * 80)
# response = agent.chat("What is 20+(2*4)? Use a tool to calculate every step.")
# print("-" * 80)
# print(response)
# # and another
# print("-" * 80)
# response = agent.chat("What is 25*(13+87)? Use a tool to calculate every step.")
# print("-" * 80)
# print(response)
