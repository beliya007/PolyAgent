"""E2E API test (staged) — assumes the server is running."""
import httpx
import time

BASE = "http://127.0.0.1:8000"

print("=" * 70)
print("PolyAgent — staged API test")
print("=" * 70)


def section(title):
    print(f"\n[{title}]")


section("1. Health")
r = httpx.get(f"{BASE}/api/agent/health", timeout=5)
print(f"  {r.status_code} {r.json()}")

section("2. Tools")
r = httpx.get(f"{BASE}/api/agent/tools", timeout=5)
print(f"  {len(r.json()['tools'])} tools registered")

section("3. Intent recognition")
r = httpx.post(
    f"{BASE}/api/agent/intent",
    json={"query": "Generate 3 UUIDs"},
    timeout=15,
)
print(f"  intent: {r.json()['intent']} (confidence {r.json()['confidence']:.2f})")

section("4. Staged tests")

section("4.1 /plan")
t0 = time.time()
r = httpx.post(
    f"{BASE}/api/agent/plan",
    json={"query": "Generate 3 UUIDs and count their characters", "mode": "auto"},
    timeout=90,
)
print(f"  status: {r.status_code}, elapsed: {time.time()-t0:.2f}s")
if r.status_code == 200:
    data = r.json()
    print(f"  intent: {data['intent']}, mode: {data['mode']}")
    print(f"  plan steps: {len(data['plan']['steps'])}")
    for s in data['plan']['steps'][:3]:
        print(f"    [{s['step']}] {s['action']} - {s['purpose']}")

section("4.2 /chat/simple")
t0 = time.time()
r = httpx.post(
    f"{BASE}/api/agent/chat/simple",
    json={"query": "Hi! What can you do?"},
    timeout=60,
)
print(f"  status: {r.status_code}, elapsed: {time.time()-t0:.2f}s")
if r.status_code == 200:
    print(f"  answer: {r.json()['answer'][:150]}...")

section("5. Memory")
r = httpx.post(
    f"{BASE}/api/agent/memory/add",
    json={"content": "preference: short answers", "memory_type": "semantic", "importance": 0.8},
    timeout=5,
)
print(f"  add: {r.json()}")

r = httpx.post(
    f"{BASE}/api/agent/memory/search",
    json={"query": "preferences", "limit": 3},
    timeout=5,
)
data = r.json()
print(f"  search: {len(data['results'])} hits")

section("6. RAG")
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
    json={"query": "peak hour capacity", "top_k": 2},
    timeout=15,
)
data = r.json()
print(f"  search: {len(data['results'])} hits")

print("\n" + "=" * 70)
print("Staged test complete.")
print("=" * 70)
