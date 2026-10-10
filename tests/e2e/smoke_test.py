"""End-to-end API smoke test (assumes the server is running)."""
import httpx
import time

BASE = "http://127.0.0.1:8000"

print("=" * 60)
print("Hello-Agents — e2e API smoke test")
print("=" * 60)


def section(title):
    print(f"\n[{title}]")


section("1. Health")
r = httpx.get(f"{BASE}/api/agent/health", timeout=10)
print(f"  {r.status_code} {r.json()}")

section("2. Tools")
r = httpx.get(f"{BASE}/api/agent/tools", timeout=10)
tools = r.json()["tools"]
print(f"  {len(tools)} tools registered: {tools}")

section("3. Agent paradigms")
r = httpx.get(f"{BASE}/api/agent/agents", timeout=10)
for p in r.json()["paradigms"]:
    print(f"  {p['name']}: {p['desc']}")

section("4. Intent recognition")
r = httpx.post(
    f"{BASE}/api/agent/intent",
    json={"query": "Calculate 15 * 8 + 32 for me."},
    timeout=30,
)
data = r.json()
print(f"  intent: {data['intent']} (confidence {data['confidence']:.2f})")
print(f"  recommended mode: {data['recommended_mode']}")

section("5. Memory system")
r = httpx.post(
    f"{BASE}/api/agent/memory/add",
    json={"content": "user prefers concise answers", "memory_type": "semantic", "importance": 0.85},
    timeout=10,
)
print(f"  add: {r.json()}")

r = httpx.post(
    f"{BASE}/api/agent/memory/search",
    json={"query": "conciseness", "limit": 3},
    timeout=10,
)
print(f"  search: {len(r.json()['results'])} hits")

section("6. RAG")
r = httpx.post(
    f"{BASE}/api/agent/rag/ingest",
    json={
        "content": (
            "# Deployment\n"
            "Scale resources with workload. Hot-spot regions double the baseline.\n"
            "Peak hours deserve 1.5x capacity."
        ),
        "doc_id": "deployment",
    },
    timeout=30,
)
print(f"  ingest: {r.json()}")

r = httpx.post(
    f"{BASE}/api/agent/rag/search",
    json={"query": "scaling rules", "top_k": 2},
    timeout=15,
)
print(f"  search: {len(r.json()['results'])} hits")

section("7. Central router (greeting)")
t0 = time.time()
r = httpx.post(
    f"{BASE}/api/agent/chat",
    json={"query": "Hello! What can you do?", "mode": "auto"},
    timeout=90,
)
elapsed = time.time() - t0
data = r.json()
print(f"  status: {r.status_code}, elapsed: {elapsed:.1f}s")
print(f"  intent: {data['intent']['intent']} | paradigm: {data['used_paradigm']} | score: {data['reflection']['score']:.2f}")
print(f"  answer: {data['final_answer'][:300]}")

print("\n" + "=" * 60)
print("Smoke test complete.")
print("=" * 60)
