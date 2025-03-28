# starter.py - starter llamaindex example
# In this version we will persist our vector store that we create from the essay

import os
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

# load all API keys from .env file
load_dotenv()
console = Console()

# NOTE: by default llamaindex uses OpenAI & it's embeddings
# so long as OPENAP_API_KEY is loaded into environment, llamaindex
# auto creates the OpenAI() llm & vector database

# read all documents in the essay directory
PERSIST_DIR = "../storage"  # persist vector store in this directory

if not os.path.exists(PERSIST_DIR):
    console.print(
        f"[yellow]Creating local context in {PERSIST_DIR}...![/yellow]", end=""
    )
    # load documents & create index
    documents = SimpleDirectoryReader("../essay").load_data()
    # create in-memory vectors for the documents
    index = VectorStoreIndex.from_documents(documents)
    # and store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    # print("Local storage context created!")
    console.print(f"[yellow]Done![/yellow]")
else:
    # print(f"Loading storage context from {PERSIST_DIR}")
    console.print(f"[cyan]Loading storage context from {PERSIST_DIR}[/cyan]")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# build a query engine off the index
query_index = index.as_query_engine()

# now ask away
user_query = ""
while True:
    # user_query = input("Enter query (quit to exit): ")
    user_query = console.input("[green]Enter query (OR type quit to exit): [/green]")
    user_query = user_query.strip().lower()
    if user_query == "quit":
        break

    response = query_index.query(user_query)
    print(response)
