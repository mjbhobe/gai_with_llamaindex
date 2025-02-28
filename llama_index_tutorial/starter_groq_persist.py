"""
starter_groq.py - starter llamaindex example with open source LLM & embeddings

In this program we will use the Groq API with the llama3-70b-8192 LLM
and the Jina Embeddings (jina-embeddings-v3) instead of the default OpenAI() model.

To use Groq API, you'll need an API key - to get one visit: https://console.groq.com/keys
Create an entry GROQ_API_KEY=gsk_XXXXX in your .env file, where gsk_XXXXX is the key you'll
create at the link above.

To use Jina embeddings, you'll need another API key - to get one visit: https://jina.ai/api-dashboard/key-manager
Create an entry JINA_API_KEY=jina_YYYYYY in your .env file, where jina_YYYYY is the key you'll
create at the link above.

Here we'll persist our embeddings, so we don't have to create them each time, which saves cost!

Author: Manish Bhobe
My experiments with Python, ML and Generative AI with llamaindex.
Code is meant for illustration purposes ONLY. Use at your own risk!
Author is not liable for any damages arising from direct/indirect use of this code.
"""

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
# your .env file must have a GROQ_API_KEY entry
load_dotenv()
console = Console()

# ------ [Start] code to enable Groq LLM & related Jina Embeddings ------------------------------------------

# NOTE: by default llamaindex uses OpenAI. In this program we'll use Groq API with llama3-70b-8192 instead
# @see: https://docs.llamaindex.ai/en/stable/examples/cookbooks/llama3_cookbook_groq/ for more examples
from llama_index.llms.groq import Groq

# and we'll use Jina embeddings instead of OpenAI embeddings, which may not be compatible with
# our Llama model
# @see: https://docs.llamaindex.ai/en/stable/examples/embeddings/jinaai_embeddings/
from llama_index.embeddings.jinaai import JinaEmbedding

# create the LLM & embeddings
# llm = Groq(model="llama3-70b-8192")
llm = Groq(
    model="llama3-70b-8192",
    # model="deepseek-r1-distill-llama-70b",
    # temperature=0.0,
    # max_tokens=2048,
)
# NOTE: your .env file must have a JINAAI_API_KEY entry
embed_model = JinaEmbedding(
    api_key=os.environ["JINAAI_API_KEY"],
    model="jina-embeddings-v3",
)

# and we MUST add the LLM & embeddings to our globals, so that llamaindex uses them
# instead of the default OpenAI ones.
from llama_index.core import Settings

Settings.llm = llm
Settings.embed_model = embed_model

# now llamaindex will use our LLM & embedding model

# ------ [End] code to enable Groq LLM & related Jina Embeddings ------------------------------------------

PERSIST_DIR = "../storage2"

if not os.path.exists(PERSIST_DIR):
    console.print(
        f"[yellow]Creating local context in {PERSIST_DIR}...[/yellow]", end=""
    )
    # load documents & create index
    documents = SimpleDirectoryReader("../essay").load_data()
    # create in-memory vectors for the documents
    index = VectorStoreIndex.from_documents(documents)
    # and store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    console.print("[yellow]Done![/yellow]")
else:
    console.print(f"[cyan]Loading storage context from {PERSIST_DIR}[/cyan]")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# build a query engine off the index
query_index = index.as_query_engine()

# now ask away
user_query = ""
while True:
    user_query = console.input("[green]Enter query (OR type quit to exit): [/green]")
    user_query = user_query.strip().lower()
    if user_query == "quit":
        break

    response = query_index.query(user_query)
    print(response)
