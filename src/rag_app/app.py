# src/app.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_app.rag_pipeline import setup_index, retrieve, generate_answer

print("""
╔══════════════════════════════════════════════╗
║   RAG Interview Assistant                    ║
║   Vector DB: Endee  |  LLM: FLAN-T5          ║
╚══════════════════════════════════════════════╝
""")

# IMPORTANT: force_recreate=False means just CONNECT, don't create
setup_index(force_recreate=False)
print("✅ Connected to Endee. Ready!\n")

while True:
    query = input("\nAsk a question (or 'quit'): ").strip()
    if not query:
        continue
    if query.lower() in ("quit", "exit"):
        print("Goodbye!")
        break

    # Step 1: Retrieve from Endee
    contexts = retrieve(query)

    print("\n📚 Retrieved Context from Endee:")
    for c in contexts:
        print(f"  [score={c['score']}] {c['question']}")
        print(f"           {c['answer'][:80]}...")

    # Step 2: Generate with FLAN-T5
    answer = generate_answer(query, contexts)

    print(f"\n🤖 Final Answer:\n{answer}")
    print("\n" + "=" * 50)