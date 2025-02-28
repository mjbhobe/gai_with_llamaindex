from llama_index.readers.google import GoogleMapsTextSearchReader
from llama_index.core import VectorStoreIndex

loader = GoogleMapsTextSearchReader()
documents = loader.load_data(
    text="I want to eat quality South Indian, Chinese and Maharashtrian food in Mumbai near Dadar",
    number_of_results=160,
)


index = VectorStoreIndex.from_documents(documents)
# index.query("Which Turkish restaurant has the best reviews?")
query_index = index.as_query_engine()
query_index.query("List the top 10 Maharashtrian food eateries")
