import os
from dotenv import load_dotenv
import nest_asyncio


# load all keys from .env file
load_dotenv()

nest_asyncio.apply()

print("Hello World!")
