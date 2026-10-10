"""E2E: full central router test (assumes the server is running)."""
import httpx
import time

BASE = "http://127.0.0.1:8000"

print("=" * 70)
print("Hello-Agents — full pipeline test")
print("=" * 70)

query = "Generate 2 UUIDs and tell me their total character count."
print(f"\nquery: {query}")

t0 = time.time()
r = httpx.post(
    f"{BASE}/api/agent/chat",
    json={"query": query, "mode": "auto"},
    timeout=300,
)
elapsed = time.time() - t0
print(f"status: {r.status_code}, elapsed: {elapsed:.2f}s")

data = r.json()
print(f"\n[intent]")
print(f"  intent: {data['intent']['intent']} (confidence {data['intent']['confidence']:.2f})")
print(f"  entities: {data['intent']['entities']}")
print(f"  reasoning: {data['intent']['reasoning']}")

print(f"\n[plan]")
plan_steps = data['plan']['steps'] if isinstance(data['plan'], dict) else []
print(f"  {len(plan_steps)} steps:")
for s in plan_steps:
    print(f"    [{s['step']}] {s['action']} - {s['purpose'][:60]}")

print(f"\n[execution]")
exec_steps = data['execution'].get('steps', [])
print(f"  {len(exec_steps)} steps:")
for s in exec_steps:
    status = 'OK' if s.get('success', True) else 'FAIL'
    out = (s.get('output') or '')[:120].replace('\n', ' ')
    print(f"    {status} [{s.get('step')}] {s.get('action')} - {out}")

print(f"\n[reflection]")
print(f"  score: {data['reflection']['score']:.2f}")
print(f"  satisfactory: {data['reflection']['is_satisfactory']}")
if data['reflection'].get('issues'):
    print(f"  issues: {data['reflection']['issues']}")

print(f"\n[final answer]")
print(data['final_answer'])

print("\n" + "=" * 70)
print(f"Full pipeline test complete (elapsed {elapsed:.2f}s)")
print("=" * 70)
