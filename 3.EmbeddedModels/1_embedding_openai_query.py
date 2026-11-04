# Learning and inference demo for Embedding Models using LangChain and OpenAI API

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding=OpenAIEmbeddings(model='text-embedding-3-large',dimensions=32)

embedding.embed_query("Delhi is the capital of India")

print(str(result))

