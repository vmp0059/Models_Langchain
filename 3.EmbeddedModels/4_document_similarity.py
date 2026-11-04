from langchain_openai import OPENAIEmbeddings

from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

load_dotenv()

embedding=OpenAIEmbeddings(model='text-embedding-3-large',dimensions=300)

documents=[
    "Delhi is the capital of India.",
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Washington is the capital of USA."
]

query="What is the capital of India?"

doc_embeddings=embedding.embed_documents(documents)
query_embedding=embedding.embed_query(query)

print(cosine_similarity([query_embedding], doc_embeddings))
