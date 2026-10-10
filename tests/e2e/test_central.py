"""E2E test for the central router (assumes the server is running)."""
import httpx
import time

BASE = "http://127.0.0.1:8000"

print("=" * 60)
print("Hello-Agents — central router test")
print("=" * 60)


def section(title):
    print(f"\n[{title}]")


section("Simple dialogue")
t0 = time.time()
r = httpx.post(
    f"{BASE}/api/agent/chat",
    json={"query": "Hello", "mode": "simple"},
    timeout=60,
)
print(f"  status: {r.status_code}, elapsed: {time.time()-t0:.1f}s")
if r.status_code == 200:
    data = r.json()
    print(f"  intent: {data.get('intent', {}).get('intent')}")
    print(f"  mode: {data.get('used_paradigm')}")
    print(f"  answer: {data.get('final_answer', '')[:100]}")

section("Tool use")
t0 = time.time()
r = httpx.post(
    f"{BASE}/api/agent/chat",
    json={"query": "Generate 3 UUIDs for me.", "mode": "auto"},
    timeout=240,
)
print(f"  status: {r.status_code}, elapsed: {time.time()-t0:.1f}s")
if r.status_code == 200:
    data = r.json()
    print(f"  intent: {data.get('intent', {}).get('intent')}")
    print(f"  mode: {data.get('used_paradigm')}")
    print(f"  score: {data.get('reflection', {}).get('score', 0):.2f}")
    print(f"  answer: {data.get('final_answer', '')[:200]}")
    exec_data = data.get("execution", {})
    print(f"  steps: {len(exec_data.get('steps', []))}")
    for s in exec_data.get("steps", [])[:3]:
        print(f"    [{s['step']}] {s['action']} - {str(s.get('output', ''))[:80]}")

print()
print("=" * 60)
print("Central router test complete.")
print("=" * 60)
