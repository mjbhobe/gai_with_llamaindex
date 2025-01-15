# starter_groq.py - starter llamaindex example
# we use the Groq() API with llama instead of OpenAI

from dotenv import find_dotenv, load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# load all API keys from .env file
# your .env file must have a GROQ_API_KEY entry
load_dotenv()

# NOTE: by default llamaindex uses OpenAI
# here we specify that we want to use Groq instead
# @see: https://docs.llamaindex.ai/en/stable/examples/cookbooks/llama3_cookbook_groq/ for more examples
from llama_index.llms.groq import Groq

# create


# read all documents in the essay directory
documents = SimpleDirectoryReader("essay").load_data()

# create in-memory vectors for the documents
index = VectorStoreIndex.from_documents(documents)

# build a query engine off the index
query_index = index.as_query_engine()

# now ask away
user_query = ""
while True:
    user_query = input("Enter query (quit to exit): ")
    user_query = user_query.strip().lower()
    if user_query == "quit":
        break

    response = query_index.query(user_query)
    print(response)
