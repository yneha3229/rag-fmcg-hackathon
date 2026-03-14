from app import build_vectorstore, build_rag_chain, strategy1_chunks, strategy2_chunks, load_documents, embeddings
import csv
import os

# 10 Test queries with expected answers
test_queries = [
    {
        "question": "What is demand forecasting?",
        "expected": "Predicting future customer demand using historical data"
    },
    {
        "question": "What are examples of FMCG products?",
        "expected": "Packaged foods, beverages, toiletries, cosmetics"
    },
    {
        "question": "What is supply chain management?",
        "expected": "Managing flow of goods from raw materials to end customer"
    },
    {
        "question": "How is AI used in supply chains?",
        "expected": "AI analyzes data to identify demand patterns and optimize operations"
    },
    {
        "question": "What is inventory management?",
        "expected": "Managing stock levels, ordering, storing and using inventory"
    },
    {
        "question": "What are quantitative methods in demand forecasting?",
        "expected": "Using historical data and analytical tools to produce predictions"
    },
    {
        "question": "What is artificial intelligence?",
        "expected": "Capability of systems to perform tasks like learning and reasoning"
    },
    {
        "question": "What are applications of AI?",
        "expected": "Search engines, recommendation systems, virtual assistants, autonomous vehicles"
    },
    {
        "question": "What is the purpose of supply chain management?",
        "expected": "Creating net value, optimizing flow of materials and information"
    },
    {
        "question": "What methods are used in demand forecasting?",
        "expected": "Qualitative methods based on expert opinion and quantitative methods using data"
    }
]

def score_answer(answer, expected):
    """Score answer: 3=correct, 2=partial, 1=wrong"""
    answer_lower = answer.lower()
    expected_words = expected.lower().split()

    # Count how many key words from expected appear in answer
    matches = sum(1 for word in expected_words if word in answer_lower)
    match_ratio = matches / len(expected_words)

    if match_ratio >= 0.5:
        return 3  # Correct
    elif match_ratio >= 0.2:
        return 2  # Partial
    else:
        return 1  # Wrong

def run_evaluation():
    print("Loading documents and building vector stores...")
    text = load_documents()
    
    docs1 = strategy1_chunks(text)
    docs2 = strategy2_chunks(text)
    
    vs1 = build_vectorstore(docs1, "strategy1")
    vs2 = build_vectorstore(docs2, "strategy2")
    
    chain1 = build_rag_chain(vs1)
    chain2 = build_rag_chain(vs2)
    
    print("Running evaluation on 10 queries...\n")
    
    results = []
    total_s1 = 0
    total_s2 = 0

    for i, item in enumerate(test_queries):
        question = item["question"]
        expected = item["expected"]

        print(f"Query {i+1}: {question}")

        # Get answers from both strategies
        answer1, _ = chain1(question)
        answer2, _ = chain2(question)

        # Score both answers
        score1 = score_answer(answer1, expected)
        score2 = score_answer(answer2, expected)

        total_s1 += score1
        total_s2 += score2

        results.append({
            "query": question,
            "expected": expected,
            "strategy1_answer": answer1[:100] + "..." if len(answer1) > 100 else answer1,
            "strategy1_score": score1,
            "strategy2_answer": answer2[:100] + "..." if len(answer2) > 100 else answer2,
            "strategy2_score": score2
        })

        print(f"  S1 Score: {score1}/3 | S2 Score: {score2}/3")

    # Print summary table
    print("\n" + "="*60)
    print("EVALUATION RESULTS SUMMARY")
    print("="*60)
    print(f"{'Query':<45} {'S1':>4} {'S2':>4}")
    print("-"*60)
    for r in results:
        print(f"{r['query'][:44]:<45} {r['strategy1_score']:>4} {r['strategy2_score']:>4}")
    print("-"*60)
    print(f"{'TOTAL SCORE':<45} {total_s1:>4} {total_s2:>4}")
    print(f"{'MAX POSSIBLE':<45} {'30':>4} {'30':>4}")
    print(f"{'PERCENTAGE':<45} {total_s1/30*100:.0f}% {total_s2/30*100:.0f}%")
    print("="*60)

    winner = "Strategy 1 (Large Chunks)" if total_s1 > total_s2 else "Strategy 2 (Small Chunks)"
    print(f"\n🏆 Winner: {winner}")

    # Save to CSV
    os.makedirs("results", exist_ok=True)
    with open("results/evaluation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("\n✅ Results saved to results/evaluation.csv")

if __name__ == "__main__":
    run_evaluation()