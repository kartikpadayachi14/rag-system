import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import faiss  
import numpy as np

# -------- Step 1: Example Document --------

text = """
Transformers are neural networks used in natural language processing.
They were introduced in the paper Attention Is All You Need.
Transformers allow models to process words in parallel.
They are the foundation of modern AI systems like ChatGPT.
They are widely used in translation, summarization, and question answering.
"""

# -------- Step 2: Sentence Chunking --------

def smart_chunk(text, max_sentences=2):
    sentences = sent_tokenize(text)
    chunks = []

    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i + max_sentences])
        chunks.append(chunk)

    return chunks

chunks = smart_chunk(text)

print("\nChunks:")
for c in chunks:
    print("-", c)

# -------- Step 3: Add Metadata --------

documents = []

for chunk in chunks:
    documents.append({
        "text": chunk,
        "source": "ai_article"
    })

# -------- Step 4: Create Embeddings --------

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [doc["text"] for doc in documents]

embeddings = model.encode(texts)

# -------- Step 5: Store in Vector DB --------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

print("\nVector database created.")

# -------- Step 6: Query --------

query = input("\nAsk a question: ")

query_embedding = model.encode([query])

k = 2

distances, indices = index.search(np.array(query_embedding), k)

# -------- Step 7: Retrieve Results --------

print("\nTop Results:")

for i in indices[0]:
    print("\nSource:", documents[i]["source"])
    print("Text:", documents[i]["text"])