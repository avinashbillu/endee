# src/rag_pipeline.py
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from endee import Endee, Precision
import torch

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GENERATION_MODEL = "google/flan-t5-base"
INDEX_NAME = "interview_qa"
DIMENSION = 384
TOP_K = 3

# Load models once at module level
print("Loading embedding model...")
embed_model = SentenceTransformer(EMBEDDING_MODEL)

print("Loading FLAN-T5 generation model...")
tokenizer = T5Tokenizer.from_pretrained(GENERATION_MODEL)
gen_model = T5ForConditionalGeneration.from_pretrained(GENERATION_MODEL)
gen_model.eval()

# Connect to Endee
print("Connecting to Endee...")
client = Endee()
index = None


def setup_index(force_recreate=False):
    global index

    if force_recreate:
        try:
            print(f"Deleting old index '{INDEX_NAME}'...")
            client.delete_index(INDEX_NAME)
            print("Deleted.")
        except Exception:
            pass  # Ignore if it didn't exist

    # Always try to just connect first
    try:
        index = client.get_index(INDEX_NAME)
        print(f"✅ Connected to existing index '{INDEX_NAME}'.")
        return
    except Exception:
        pass  # Index doesn't exist, create it below

    # Only reaches here if index truly doesn't exist
    print(f"Creating Endee index '{INDEX_NAME}'...")
    client.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        space_type="cosine",
        precision=Precision.INT8
    )
    index = client.get_index(INDEX_NAME)
    print("Index created.\n")


def store_data(documents):
    print(f"Storing {len(documents)} documents in Endee...")

    for doc in documents:
        text = f"Question: {doc['question']} Answer: {doc['answer']}"
        embedding = embed_model.encode(text).tolist()

        # Must include 'filter' key as per Endee SDK
        vector = {
            "id": doc["id"],
            "vector": embedding,
            "meta": {
                "question": doc["question"],
                "answer": doc["answer"]
            },
            "filter": {}
        }

        index.upsert([vector])
        print(f"  Stored: {doc['id']}")

    print(f"\nAll {len(documents)} documents stored in Endee.\n")


def retrieve(query, top_k=TOP_K):
    query_vec = embed_model.encode(query).tolist()

    results = index.query(
        vector=query_vec,
        top_k=top_k,
        ef=128,
        include_vectors=False
    )

    docs = []
    for item in results:
        # Results are dicts: item['id'], item['similarity'], item['meta']
        score = round(item["similarity"], 4)
        if score < 0.3:
            continue
        docs.append({
            "id": item["id"],
            "score": round(item["similarity"], 4),
            "question": item.get("meta", {}).get("question", ""),
            "answer": item.get("meta", {}).get("answer", "")
        })
    return docs


def generate_answer(query, contexts):
    # Use top retrieved answer as the base
    if not contexts:
        return "❌ No relevant information found in the database."
    best_answer = contexts[0]['answer']
    
    prompt = f"Question: {query} Answer: {best_answer} Summarize:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    with torch.no_grad():
        outputs = gen_model.generate(
            inputs["input_ids"],
            max_new_tokens=150,
            num_beams=3,
            early_stopping=True,
            no_repeat_ngram_size=3,
            min_length=20
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # If generated answer is too short or low quality, use retrieved answer directly
    if len(generated.strip()) < 20 or generated.strip() in prompt:
        return best_answer

    return generated