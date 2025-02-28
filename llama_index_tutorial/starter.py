# starter.py - starter llamaindex example
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# load all API keys from .env file
load_dotenv()
console = Console()

# NOTE: by default llamaindex uses OpenAI
# so long as OPENAP_API_KEY is loaded into environment, llamaindex
# auto creates the OpenAI() llm & vector database

# read all documents in the essay directory
documents = SimpleDirectoryReader("../essay").load_data()

# create in-memory vectors for the documents
index = VectorStoreIndex.from_documents(documents)

# build a query engine off the index
query_index = index.as_query_engine()

# now ask away
user_query = ""
while True:
    # user_query = input("Enter query (OR type quit to exit): ")
    user_query = console.input("[green]Enter query (OR type quit to exit): [/green]")
    user_query = user_query.strip().lower()
    if user_query == "quit":
        break

    response = query_index.query(user_query)
    print(response)
