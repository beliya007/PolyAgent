"""E2E: full central router pipeline (assumes the server is running)."""
import httpx
import time

BASE = "http://127.0.0.1:8000"

print("=" * 70)
print("Hello-Agents — full central router pipeline")
print("=" * 70)

test_cases = [
    {
        "name": "calculation",
        "query": "What is 12 * 8 + 5?",
    },
    {
        "name": "tool_use",
        "query": "Generate 2 UUIDs for me.",
    },
    {
        "name": "knowledge_query",
        "query": "What is an agent framework?",
    },
    {
        "name": "chitchat",
        "query": "Hi!",
    },
]

for case in test_cases:
    print(f"\n{'-' * 70}")
    print(f">>> [{case['name']}] {case['query']}")
    t0 = time.time()
    r = httpx.post(
        f"{BASE}/api/agent/chat",
        json={"query": case["query"], "mode": "auto"},
        timeout=180,
    )
    elapsed = time.time() - t0
    data = r.json()

    print(f"status: {r.status_code}, elapsed: {elapsed:.2f}s")

    intent = data.get("intent", {})
    print("\n[intent]")
    print(f"  intent: {intent.get('intent')} (confidence {intent.get('confidence', 0):.2f})")
    print(f"  entities: {intent.get('entities')}")
    print(f"  mode: {intent.get('mode')}")

    plan = data.get("plan", {})
    steps = plan.get("steps", [])
    print(f"\n[plan] {len(steps)} steps")
    for s in steps[:4]:
        print(f"  [{s.get('step')}] {s.get('action')} - {str(s.get('purpose', ''))[:50]}")

    exec_data = data.get("execution", {})
    print("\n[execution]")
    print(f"  paradigm: {exec_data.get('paradigm', 'N/A')}")
    print(f"  steps: {len(exec_data.get('steps', []))}")

    reflection = data.get("reflection", {})
    print("\n[reflection]")
    print(f"  score: {reflection.get('score', 0):.2f}")
    print(f"  satisfactory: {reflection.get('is_satisfactory')}")

    print("\n[final answer]")
    answer = data.get("final_answer", "")
    print(answer[:300] + ("..." if len(answer) > 300 else ""))
    print(f"  ({len(answer)} chars)")

print(f"\n{'=' * 70}")
print("Pipeline test complete.")
print("=" * 70)
