# Learning and inference demo for Embedding Models using LangChain and HuggingFace Local Model
from langchain_huggingface import HuggingFaceEmbeddings

embedding=HuggingFaceEmbeddings(model_name='sentenece-transformers/all-MiniLM-L6-v2')

text="Delhi is the capital of India."

vector=embedding.embed_query(text)

print(str(vector))

