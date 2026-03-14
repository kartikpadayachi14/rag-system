import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
import mlflow

# Documents (Knowledge Base)

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


# Keyword Search (BM25)

tokenized_docs = [doc.lower().split() for doc in documents]

bm25 = BM25Okapi(tokenized_docs)

print("BM25 keyword search ready")


# Vector Search (Embeddings)

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(documents)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

print("Vector database created")


# Prompt Builder

def build_prompt(context, question):

    prompt = f"""
You are an AI assistant helping answer questions using provided documents.

Rules:
- Use ONLY the information from the context.
- If the answer is not present, respond: "I don't know."
- Keep answers concise and factual.

Return the answer in this format:

Answer: <answer>
Source: <document source>
Confidence: <low/medium/high>

Context:
{context}

Question:
{question}

Answer:
"""
    return prompt


# Hybrid Search

def hybrid_search(query):

    print("\nUser Query:", query)

    with mlflow.start_run():

        #Log Parameters

        mlflow.log_param("embedding_model","all-MiniLM-L6-v2")
        mlflow.log_param("retriever", "hybrid")
        mlflow.log_param("top_k",3)
        mlflow.log_param("query", query)


    # -------- Keyword Search --------

    tokenized_query = query.lower().split()

    keyword_results = bm25.get_top_n(tokenized_query, documents, n=3)

    print("\nKeyword Results:")
    for r in keyword_results:
        print("-", r)


    # -------- Vector Search --------

    query_embedding = model.encode([query])

    k = 3

    distances, indices = index.search(np.array(query_embedding), k)

    vector_results = []

    for i in indices[0]:
        vector_results.append(documents[i])

    print("\nVector Results:")
    for r in vector_results:
        print("-", r)


    # -------- Combine Results --------

    combined_results = list(dict.fromkeys(keyword_results + vector_results))

    print("\nHybrid Results:")
    for r in combined_results:
        print("-", r)


    # -------- Context Compression --------

    top_k = combined_results[:3]

    context = " ".join(top_k)

    # -------- Guardrail --------

    if len(context.strip()) == 0:
        return "I don't know"


    # -------- Build Prompt --------

    prompt = build_prompt(context, query)

    print("\nGenerated Prompt Sent To LLM:")
    print(prompt)

    # (Simulated LLM answer for now)
    answer = "Answer: FAISS is used for efficient vector similarity search.\nSource: FAISS enables fast similarity search for vector databases\nConfidence: medium"

    print("\nLLM Output:")
    print(answer)

    #Log metrics

    docs_retrieved = len(combined_results)
    mlflow.log_metric("docs_retrieved", docs_retrieved)

    uncertain = 1 if "I don't know" in answer else 0
    mlflow.log_metric("uncertain_answer", uncertain)

    # Guardrail example
    if "I don't know" in answer:
        print("\nLogged uncertain case.")

mlflow.set_experiment("Hybrid-RAG-Experiments")

# User Query Loop

while True:

    query = input("\nAsk a question (type 'exit' to stop): ")

    if query == "exit":
        break

    hybrid_search(query)