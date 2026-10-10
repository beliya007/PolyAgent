"""E2E: v4-style test (assumes the server is running)."""
import httpx
import time

BASE = "http://127.0.0.1:8000"

print("=" * 60)
print("PolyAgent — v4-style e2e test")
print("=" * 60)


def section(title):
    print(f"\n[{title}]")


section("1. Health")
r = httpx.get(f"{BASE}/api/agent/health", timeout=5)
print(f"  {r.json()}")

section("2. Tools")
r = httpx.get(f"{BASE}/api/agent/tools", timeout=5)
tools = r.json()["tools"]
print(f"  {len(tools)} tools registered")
for t in tools:
    print(f"    - {t}")

section("3. Agent paradigms")
r = httpx.get(f"{BASE}/api/agent/agents", timeout=5)
for p in r.json()["paradigms"]:
    print(f"  [{p['name']}] {p['desc']}")

section("4. Intent recognition")
r = httpx.post(
    f"{BASE}/api/agent/intent",
    json={"query": "Generate 3 UUIDs"},
    timeout=30,
)
data = r.json()
print(f"  intent: {data['intent']} (confidence {data['confidence']:.2f})")
print(f"  recommended mode: {data['recommended_mode']}")

section("5. Plan")
r = httpx.post(
    f"{BASE}/api/agent/plan",
    json={"query": "Generate 3 UUIDs and format them as JSON"},
    timeout=60,
)
data = r.json()
print(f"  intent: {data['intent']}, mode: {data['mode']}")
print(f"  plan steps: {len(data['plan']['steps'])}")
for s in data["plan"]["steps"][:4]:
    print(f"    [{s['step']}] {s['action']} - {s['purpose']}")

section("6. Memory")
r = httpx.post(
    f"{BASE}/api/agent/memory/add",
    json={"content": "preference: concise answers", "memory_type": "semantic", "importance": 0.9},
    timeout=5,
)
print(f"  add: {r.json()}")

r = httpx.post(
    f"{BASE}/api/agent/memory/search",
    json={"query": "preferences", "limit": 3},
    timeout=5,
)
print(f"  search: {len(r.json()['results'])} hits")

r = httpx.get(f"{BASE}/api/agent/memory/stats", timeout=5)
print(f"  stats: {r.json()}")

section("7. RAG")
r = httpx.post(
    f"{BASE}/api/agent/rag/ingest",
    json={
        "content": "# Scaling\nScale resources with workload. Hot-spot regions double baseline.",
        "doc_id": "scaling",
    },
    timeout=30,
)
print(f"  ingest: {r.json()}")

r = httpx.post(
    f"{BASE}/api/agent/rag/search",
    json={"query": "scaling rules", "top_k": 2},
    timeout=15,
)
results = r.json()["results"]
print(f"  search: {len(results)} hits")
for item in results:
    print(f"    [{item['score']:.2f}] {item['content'][:60]}...")

section("8. LLM stats")
r = httpx.get(f"{BASE}/api/agent/llm/stats", timeout=5)
print(f"  {r.json()}")

section("9. System stats")
r = httpx.get(f"{BASE}/api/agent/stats", timeout=5)
data = r.json()
print(f"  LLM: {data.get('llm', {}).get('model')}")
print(f"  Tools: {data.get('tools', [])}")

print("\n" + "=" * 60)
print("v4-style e2e test complete.")
print("=" * 60)
