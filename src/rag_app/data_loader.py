# src/data_loader.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_app.rag_pipeline import setup_index, store_data

# Inline dataset (no fragile .txt file needed)
KNOWLEDGE_BASE = [
    {"id": "q1",  "question": "What is a REST API?",
     "answer": "A REST API uses HTTP methods like GET, POST, PUT, DELETE. It is stateless and returns JSON data."},
    {"id": "q2",  "question": "What is SQL vs NoSQL?",
     "answer": "SQL is relational with fixed schemas. NoSQL is flexible, scales horizontally. SQL uses tables; NoSQL uses documents, key-value, or graphs."},
    {"id": "q3",  "question": "What is a vector database?",
     "answer": "A vector database stores embeddings and enables similarity search using algorithms like HNSW. Used in AI for semantic search and RAG."},
    {"id": "q4",  "question": "What is RAG?",
     "answer": "RAG is Retrieval Augmented Generation. It retrieves relevant documents using vector search and passes them as context to an LLM to generate accurate answers."},
    {"id": "q5",  "question": "What is Docker?",
     "answer": "Docker packages apps into containers. Containers run consistently across environments. docker-compose manages multi-container apps."},
    {"id": "q6",  "question": "What is OOP?",
     "answer": "OOP has four pillars: Encapsulation, Inheritance, Polymorphism, Abstraction. It makes code modular and reusable."},
    {"id": "q7",  "question": "What is machine learning?",
     "answer": "ML is a subset of AI where algorithms learn from data. Types: supervised, unsupervised, reinforcement learning."},
    {"id": "q8",  "question": "What are embeddings?",
     "answer": "Embeddings are dense vector representations of text capturing semantic meaning. Similar texts have similar vectors."},
    {"id": "q9",  "question": "What is Git?",
     "answer": "Git is a version control system. Key commands: git clone, add, commit, push, pull, branch, merge."},
    {"id": "q10", "question": "What is the CAP theorem?",
     "answer": "CAP theorem: a distributed system can guarantee only 2 of 3 — Consistency, Availability, Partition Tolerance."},
]

if __name__ == "__main__":
    print("=" * 50)
    print("Loading knowledge base into Endee...")
    print("=" * 50)
    setup_index(force_recreate=True)
    store_data(KNOWLEDGE_BASE)
    print(f"Done! {len(KNOWLEDGE_BASE)} documents stored.")
    print("Now run: python src/app.py")