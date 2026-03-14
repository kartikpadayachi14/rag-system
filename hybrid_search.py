import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import numpy as np


documents = [
    "Azure Machine Learning helps deploy AI models in the cloud",
    "Transformers use attention mechanisms for natural language processing",
    "FAISS enables fast similarity search for vector databases",
    "Vector databases store embeddings for AI applications",
    "Cloud infrastructure enables scalable AI systems",
    "FAISS indexes vectors to enable efficient similarity search",
    "Large language models use transformers architecture",
]

print("\nDocuments loaded:", len(documents))

#Keyword Search(BM25)
tokenized_docs = [doc.lower().split() for doc in documents]

bm25 = BM25Okapi(tokenized_docs)

print("BM25 keyword search ready")

#Embedding
print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(documents)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

print("Vector database  created")

# Search Function

def hybrid_search(query):
    print("\nUser Query:", query)

    tokenized_query = query.lower().split()

    keyword_results = bm25.get_top_n(tokenized_query, documents, n=3)

    print("\nKeyword Results:")
    for r in keyword_results:
        print("-", r)

    query_embedding = model.encode([query])
    k=3

    distances, indices = index.search(np.array(query_embedding),k)

    vector_results = []

    for i in indices[0]:
        vector_results.append(documents[i])

    print("\nVector Resuts:")
    for r in vector_results:
        print("-", r)

    #combine results
    combined_results = list(set(keyword_results + vector_results))

    print("\nHybrid Results:")
    for r in combined_results:
        print("-", r)

# User Query
while True:
    query = input("\nAsk a question (type 'exit' to stop): ")

    if query =="exit":
        break
    hybrid_search(query)