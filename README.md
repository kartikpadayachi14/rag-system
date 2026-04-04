# Hybrid RAG System (Retrieval-Augmented Generation)

This project implements a **Hybrid Retrieval-Augmented Generation (RAG) system** that combines **keyword search (BM25)** and **vector search (FAISS + Sentence Transformers)** to retrieve relevant documents and generate answers using an LLM.

The system also includes **MLflow tracking**, **guardrails**, and **Docker support** for experimentation and deployment.


## Project Overview

Traditional search uses keyword matching, while modern AI systems use vector embeddings.
This project combines both approaches using **Hybrid Search** to improve retrieval accuracy.

Pipeline:

User Query → Hybrid Search (BM25 + Vector Search) → Context Compression → Prompt Builder → LLM → Answer


## Features

* Hybrid search (BM25 + Vector Search)
* Vector database using FAISS
* Embeddings using Sentence Transformers
* Context compression
* Prompt builder for LLM
* Guardrails for unknown answers
* MLflow experiment tracking
* Docker support
* CI/CD pipeline (GitHub Actions)

---

## Project Structure

```
rag-system/
│
├── rag_pipeline.py      # Main RAG pipeline
├── rag_chunk.py         # Document chunking
├── hybrid_search.py     # Hybrid retrieval (BM25 + FAISS)
├── guardrails.py        # Safety & fallback logic
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker container
├── mlflow.db            # MLflow experiment database
│
└── .github/workflows/
    └── ci.yml           # CI/CD pipeline
```

---

## Technologies Used

| Component           | Technology                                     |
| ------------------- | ---------------------------------------------- |
| Embeddings          | Sentence Transformers (all-MiniLM-L6-v2)       |
| Vector Search       | FAISS                                          |
| Keyword Search      | BM25                                           |
| Experiment Tracking | MLflow                                         |
| Language Model      | (Simulated / Can connect OpenAI / HuggingFace) |
| Deployment          | Docker                                         |
| CI/CD               | GitHub Actions                                 |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/rag-system.git
cd rag-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the system

```bash
python rag_pipeline.py
```

---

## Example Query

```
Ask a question: What is FAISS?
```

Example Output:

```
Answer: FAISS is used for efficient vector similarity search.
Source: FAISS enables fast similarity search for vector databases
Confidence: medium
```

---

## MLflow Tracking

This project logs:

* Embedding model used
* Retriever type (Hybrid)
* Top-K documents retrieved
* Query
* Number of documents retrieved
* Uncertain answers

To run MLflow UI:

```bash
mlflow ui
```

Then open:

```
http://127.0.0.1:5000
```

---

## Docker

Build Docker image:

```bash
docker build -t rag-system .
```

Run container:

```bash
docker run -it rag-system
```

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is an AI architecture that:

1. Retrieves relevant documents
2. Sends them as context to an LLM
3. LLM generates an answer using that context

This reduces hallucinations and improves factual accuracy.
