# 🤖 RAG Interview Assistant — Powered by Endee Vector Database

> A production-grade **Retrieval Augmented Generation (RAG)** system built using
> **Endee** as the vector database, **Sentence Transformers** for embeddings, and
> **FLAN-T5** for answer generation. No paid APIs. Fully open-source.

---

## 📌 Project Overview

This project is an AI-powered Interview Assistant that answers software engineering
questions using RAG (Retrieval Augmented Generation).

- A knowledge base of **10 real software engineering interview Q&As** is embedded
  and stored as vectors in **Endee vector database**
- When a user asks a question, it is converted to a vector and used to **search
  Endee** for the most semantically relevant documents
- The retrieved documents are passed as context to **FLAN-T5**, which generates
  a grounded, accurate answer
- Every answer is **100% grounded** in the knowledge base — no hallucination

---

## 🧠 What is RAG?

**RAG (Retrieval Augmented Generation)** is an AI technique that combines:

1. **Retrieval** — finding relevant documents from a knowledge base using
   vector similarity search in Endee
2. **Augmented** — using those retrieved documents as context for the LLM
3. **Generation** — the LLM generates a grounded answer from that context

| Plain LLM | RAG with Endee |
|---|---|
| May hallucinate | Grounded in real documents |
| Fixed training data | Knowledge base can be updated anytime |
| No source tracing | Shows exactly which docs were retrieved |
| Black box answers | Explainable and transparent |

---

## 🏗️ System Design
```
User Question
     │
     ▼
┌─────────────────────────────┐
│  Sentence Transformer       │  ← Converts question to 384-dim vector
│  (all-MiniLM-L6-v2)         │
└──────────┬──────────────────┘
           │  Query Vector
           ▼
┌─────────────────────────────┐
│   Endee Vector Database     │  ← Cosine similarity search
│   Running at localhost:8080 │     Returns Top-3 matching documents
│   Index: interview_qa       │
│   Dimension: 384            │
│   Precision: INT8           │
└──────────┬──────────────────┘
           │  Retrieved Documents + Similarity Scores
           ▼
┌─────────────────────────────┐
│   Context Builder           │  ← Formats retrieved docs as prompt
└──────────┬──────────────────┘
           │  Prompt = Question + Retrieved Context
           ▼
┌─────────────────────────────┐
│   FLAN-T5 (base)            │  ← Generates final answer
│   (google/flan-t5-base)     │     from retrieved context only
└──────────┬──────────────────┘
           │
           ▼
      Final Answer
```

---

## 🔍 How Endee Is Used

Endee is the **core component** of this project. It acts as the long-term
vector memory of the AI system.

| Operation | Code | Description |
|---|---|---|
| **Start Server** | `docker compose up -d` | Starts Endee at localhost:8080 |
| **Connect** | `client = Endee()` | Connects Python to Endee |
| **Create Index** | `client.create_index(name, dimension=384, space_type="cosine")` | Creates vector index |
| **Store Vectors** | `index.upsert([{id, vector, meta, filter}])` | Stores embeddings permanently |
| **Search Vectors** | `index.query(vector, top_k=3)` | Finds most similar documents |
| **Get Index** | `client.get_index(name)` | Reconnects to existing index |

### Why Endee?
- High-performance vector search using **HNSW algorithm**
- Stores vectors **persistently** in Docker volume — survives restarts
- Supports **cosine similarity** — perfect for semantic search
- Simple Python SDK — easy to integrate
- Runs **100% locally** — no cloud costs, no API keys

---

## 📁 Project Structure
```
endee/
├── src/
|   └── rag_app/
│        ├── app.py              ← Main interactive Q&A application
│        ├── rag_pipeline.py     ← Core RAG logic (embed, store, retrieve, generate)
│        └── data_loader.py      ← Loads knowledge base into Endee (run once)
├── docker-compose.yml           ← Starts Endee vector database server
├── requirements.txt             ← Python dependencies
└── README.md
```

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.9+
- Docker Desktop (must be running)

### Step 1 — Clone the repository
```bash
git clone https://github.com/<your-username>/endee.git
cd endee
```

### Step 2 — Start Endee vector database
```bash
docker compose up -d
```
Verify Endee is running:
```bash
curl http://localhost:8080/api/v1/index/list
```
Expected: `{"indexes":[]}`

### Step 3 — Install Python dependencies
```bash
pip install -r requirements.txt
```
> ⚠️ First run downloads ~500MB of AI models. One-time only.

### Step 4 — Load knowledge base into Endee
```bash
python src/rag_app/data_loader.py
```
Expected output:
```
Creating Endee index 'interview_qa'...
Index created.
Storing 10 documents in Endee...
  Stored: q1
  Stored: q2
  ...
  Stored: q10
All 10 documents stored in Endee.
```

### Step 5 — Run the RAG assistant
```bash
python src/rag_app/app.py
```

---

## 💬 Example Input / Output
```
╔══════════════════════════════════════════════╗
║   RAG Interview Assistant                    ║
║   Vector DB: Endee  |  LLM: FLAN-T5          ║
╚══════════════════════════════════════════════╝

Ask a question (or 'quit'): what is docker?

📚 Retrieved Context from Endee:
  [score=0.6976] What is Docker?
           Docker packages apps into containers. Containers run
           consistently across environments...
  [score=0.2972] What is Git?
           Git is a version control system...
  [score=0.1445] What is a REST API?
           A REST API uses HTTP methods...

🤖 Final Answer:
Docker packages apps into containers. Containers run consistently
across environments. docker-compose manages multi-container apps.
```

---

## 🔄 Complete Workflow
```
Phase 1 — Setup (run once):
docker compose up -d
→ Endee vector database starts at localhost:8080

Phase 2 — Data Loading (run once):
python src/data_loader.py
→ 10 Q&As converted to 384-dim vectors
→ Stored permanently in Endee Docker volume

Phase 3 — Question Answering (every run):
python src/app.py
→ User types a question
→ Question converted to vector using Sentence Transformer
→ Endee performs cosine similarity search
→ Top 3 matching documents returned with scores
→ FLAN-T5 generates answer using retrieved context
→ Answer displayed with source documents
```

---

## 🛠️ Tech Stack

| Component | Tool | Purpose |
|---|---|---|
| **Vector Database** | Endee | Store and search vector embeddings |
| **Embeddings** | all-MiniLM-L6-v2 | Convert text to 384-dim vectors |
| **LLM** | FLAN-T5 base | Generate answers from context |
| **Language** | Python 3.10 | Project orchestration |
| **Containerization** | Docker | Run Endee locally |

---

## 📦 Requirements
```
endee
sentence-transformers
transformers
torch
```

---

## 🚀 Future Improvements

- Expand knowledge base with more Q&A pairs
- Add Streamlit web UI for better experience
- Upgrade to larger LLM (FLAN-T5-large or Mistral)
- Add hybrid search using Endee's sparse vector support
- Deploy to cloud with persistent Endee instance

---

## 👨‍💻 Author

**Avinash**
Built as part of Endee.io assignment — demonstrating real-world RAG
using Endee as the vector database.

---

*Built with ❤️ using [Endee Vector Database](https://github.com/endee-io/endee)*
