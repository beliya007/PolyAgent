"""
Generic multi-agent A2A coordination example.

This script demonstrates how to use the A2A protocol to let several
specialised agents collaborate through point-to-point direct calls. The
agents here are intentionally generic (calculator, weather, knowledge,
planner) so the example can be reused as a starting point for any domain.

Run with:

    cd agent
    python -m protocols.demo_multi_agent_a2a
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

from protocols.a2a import (
    A2AClient,
    A2AServer,
    A2ATool,
    AuthManager,
    MessageSigner,
    SecureA2AServer,
    StreamingA2AClient,
)
from protocols.anp import (
    ANPDiscovery,
    ANPIdentity,
    ANPMesh,
    ANPRouter,
    HealthMonitor,
    SecureMessageBus,
)


# ============================================================
#  1. Define specialised agents (A2A servers)
# ============================================================
def create_calculator_agent() -> A2AServer:
    server = A2AServer(
        name="CalculatorAgent",
        description="Performs arithmetic operations.",
        capabilities={"math": ["add", "multiply", "statistics"]},
    )

    @server.skill("add")
    def add(text: str) -> str:
        nums = [float(x) for x in text.replace(",", " ").split() if x.strip()]
        return json.dumps({"sum": sum(nums)}, ensure_ascii=False)

    @server.skill("multiply")
    def multiply(text: str) -> str:
        nums = [float(x) for x in text.replace(",", " ").split() if x.strip()]
        result = 1.0
        for n in nums:
            result *= n
        return json.dumps({"product": result}, ensure_ascii=False)

    @server.skill("statistics")
    def statistics(text: str) -> str:
        nums = [float(x) for x in text.replace(",", " ").split() if x.strip()]
        if not nums:
            return json.dumps({"error": "no numbers"}, ensure_ascii=False)
        return json.dumps(
            {
                "count": len(nums),
                "sum": sum(nums),
                "mean": sum(nums) / len(nums),
                "min": min(nums),
                "max": max(nums),
            },
            ensure_ascii=False,
        )

    return server


def create_knowledge_agent() -> A2AServer:
    server = A2AServer(
        name="KnowledgeAgent",
        description="Answers general knowledge questions from a small built-in corpus.",
        capabilities={"knowledge": ["ask", "list_topics"]},
    )

    CORPUS = {
        "agents": "An agent perceives its environment and takes actions to achieve goals.",
        "rag": "Retrieval-augmented generation grounds LLM answers in external documents.",
        "memory": "Memory lets agents persist facts across turns and sessions.",
        "mcp": "The Model Context Protocol standardises how agents call tools.",
        "a2a": "The Agent-to-Agent protocol lets independent agents collaborate.",
    }

    @server.skill("ask")
    def ask(text: str) -> str:
        text_l = text.lower()
        for key, value in CORPUS.items():
            if key in text_l:
                return json.dumps({"topic": key, "answer": value}, ensure_ascii=False)
        return json.dumps(
            {"topic": None, "answer": "I don't have that topic in my corpus."},
            ensure_ascii=False,
        )

    @server.skill("list_topics")
    def list_topics(text: str) -> str:  # noqa: ARG001
        return json.dumps({"topics": list(CORPUS.keys())}, ensure_ascii=False)

    return server


def create_planner_agent() -> A2AServer:
    server = A2AServer(
        name="PlannerAgent",
        description="Decomposes a task into ordered steps.",
        capabilities={"planning": ["decompose"]},
    )

    @server.skill("decompose")
    def decompose(text: str) -> str:
        parts = [p.strip() for p in re.split(r"[\.,;\n]", text) if p.strip()]
        if not parts:
            parts = [text.strip()]
        return json.dumps(
            {"task": text, "steps": [{"idx": i + 1, "step": p} for i, p in enumerate(parts)]},
            ensure_ascii=False,
        )

    return server


# ============================================================
#  2. Coordinator
# ============================================================
import re  # noqa: E402  — placed after the agents so the example reads top-down


class GenericA2ACoordinator:
    """Coordinator that uses the A2A protocol to delegate subtasks."""

    def __init__(self) -> None:
        self.calculator = create_calculator_agent()
        self.knowledge = create_knowledge_agent()
        self.planner = create_planner_agent()

        self.clients = {
            "calculator": A2AClient(
                "http://localhost:9101", use_http=False, local_server=self.calculator
            ),
            "knowledge": A2AClient(
                "http://localhost:9102", use_http=False, local_server=self.knowledge
            ),
            "planner": A2AClient(
                "http://localhost:9103", use_http=False, local_server=self.planner
            ),
        }

        self.tools = {
            "calculator": A2ATool.from_server(self.calculator, name="calculator"),
            "knowledge": A2ATool.from_server(self.knowledge, name="knowledge"),
            "planner": A2ATool.from_server(self.planner, name="planner"),
        }

        self.auth = AuthManager(secret_key="polyagent-a2a-secret")
        self.signer = MessageSigner(secret_key="polyagent-a2a-signing")
        self.secure_servers = {
            "calculator": SecureA2AServer(self.calculator, self.auth, self.signer),
            "knowledge": SecureA2AServer(self.knowledge, self.auth, self.signer),
        }

        self.streaming_clients = {
            "knowledge": StreamingA2AClient("http://localhost:9102"),
        }

    def solve(self, task: str) -> Dict[str, Any]:
        """Run the full coordination flow on a generic task."""
        result: Dict[str, Any] = {"task": task, "steps": []}

        plan_resp = self.clients["planner"].execute_skill("decompose", task)
        result["steps"].append(
            {"agent": "PlannerAgent", "skill": "decompose", "output": plan_resp.get("result")}
        )

        knowledge_resp = self.clients["knowledge"].execute_skill("ask", task)
        result["steps"].append(
            {"agent": "KnowledgeAgent", "skill": "ask", "output": knowledge_resp.get("result")}
        )

        calc_resp = self.clients["calculator"].execute_skill(
            "statistics", "10 20 30 40 50"
        )
        result["steps"].append(
            {"agent": "CalculatorAgent", "skill": "statistics", "output": calc_resp.get("result")}
        )

        return result


# ============================================================
#  3. ANP discovery
# ============================================================
def setup_anp_network() -> ANPDiscovery:
    discovery = ANPDiscovery()
    services = [
        {
            "service_id": "calculator_1",
            "service_name": "Calculator Service",
            "service_type": "math",
            "capabilities": ["add", "multiply", "statistics"],
            "endpoint": "http://localhost:9101",
            "metadata": {"load": 0.2, "price": 0.01, "version": "1.0.0"},
        },
        {
            "service_id": "knowledge_1",
            "service_name": "Knowledge Service",
            "service_type": "knowledge",
            "capabilities": ["ask", "list_topics"],
            "endpoint": "http://localhost:9102",
            "metadata": {"load": 0.4, "price": 0.02, "version": "1.0.0"},
        },
        {
            "service_id": "planner_1",
            "service_name": "Planner Service",
            "service_type": "planning",
            "capabilities": ["decompose"],
            "endpoint": "http://localhost:9103",
            "metadata": {"load": 0.1, "price": 0.005, "version": "1.0.0"},
        },
    ]
    for svc in services:
        discovery.register(**svc)
    return discovery


# ============================================================
#  4. Demonstrations
# ============================================================
def demo_coordination() -> None:
    print("\n" + "=" * 70)
    print("Generic multi-agent A2A coordination demo")
    print("=" * 70)
    coordinator = GenericA2ACoordinator()
    result = coordinator.solve("Explain how an agent uses memory and RAG")
    print(f"\nTask: {result['task']}")
    print(f"\nSteps ({len(result['steps'])}):")
    for i, step in enumerate(result["steps"], 1):
        print(f"  [{i}] {step['agent']}.{step['skill']}")
        print(f"      -> {step['output']}")


def demo_secure() -> None:
    print("\n" + "=" * 70)
    print("Secure A2A (authentication + signing) demo")
    print("=" * 70)
    coordinator = GenericA2ACoordinator()
    token = coordinator.auth.generate_token("coordinator_agent", scopes=["execute_skill"], ttl=600)
    signed = coordinator.signer.sign_message({"skill": "add", "input": "2 3"})
    result = coordinator.secure_servers["calculator"].handle_secure_request(
        signed_msg=signed, token=token
    )
    print(f"  success: {result.get('success')}")
    print(f"  authenticated_agent: {result.get('authenticated_agent')}")
    print(f"  verified_signature: {result.get('verified_signature')}")


async def demo_streaming() -> None:
    print("\n" + "=" * 70)
    print("Streaming A2A demo")
    print("=" * 70)
    coordinator = GenericA2ACoordinator()
    chunks = []
    async for chunk in coordinator.streaming_clients["knowledge"].stream_skill(
        "ask", "agents"
    ):
        chunks.append(f"[{chunk.chunk_id}] {chunk.content}")
        if chunk.is_final:
            break
    print(f"\nReceived {len(chunks)} chunks")
    for c in chunks[:3]:
        print(f"  {c}")


def demo_anp_discovery() -> None:
    print("\n" + "=" * 70)
    print("ANP discovery + A2A demo")
    print("=" * 70)
    discovery = setup_anp_network()
    router = ANPRouter(discovery)
    best = router.route_by_capability("ask", strategy="least_loaded")
    print(f"  Selected: {best.service_name} (load={best.load:.1%})")
    client = A2AClient(best.endpoint, use_http=False)
    card = client.get_agent_card()
    print(f"  Card: {card.get('name')}")


def demo_anp_identity_mesh() -> None:
    print("\n" + "=" * 70)
    print("ANP identity + mesh demo")
    print("=" * 70)
    identities = [
        ANPIdentity(agent_id="agent_a", service_type="math"),
        ANPIdentity(agent_id="agent_b", service_type="knowledge"),
        ANPIdentity(agent_id="agent_c", service_type="planning"),
    ]
    mesh = ANPMesh()
    for ident in identities:
        mesh.add_agent(ident)
    mesh.connect("agent_a", "agent_b")
    mesh.connect("agent_b", "agent_c")
    path = mesh.find_path("agent_a", "agent_c")
    print(f"  agent_a -> agent_c path: {' -> '.join(path) if path else 'unreachable'}")

    bus = SecureMessageBus(secret_key="polyagent-anp-bus")
    for ident in identities:
        bus.register_identity(ident)
    msg_id = bus.send(
        from_agent="agent_a",
        to_agent="agent_b",
        payload={"action": "compute", "data": [1, 2, 3]},
    )
    print(f"  sent message {msg_id}")
    msg = bus.receive("agent_b")
    if msg:
        print(f"  agent_b received action={msg.payload.get('action')}")


# ============================================================
#  5. Entry point
# ============================================================
def main() -> None:
    print("=" * 70)
    print("PolyAgent — generic multi-agent A2A / ANP demo")
    print("=" * 70)
    print(
        """
This script demonstrates:

  1. A2A direct communication between independent agents.
  2. Secure A2A (authentication + signing).
  3. Streaming A2A for long-running tasks.
  4. ANP service discovery + A2A wiring.
  5. ANP identity + service mesh + secure message bus.

Use it as a starting template for your own domain-specific agents.
"""
    )
    demo_coordination()
    demo_secure()
    asyncio.run(demo_streaming())
    demo_anp_discovery()
    demo_anp_identity_mesh()
    print("\n" + "=" * 70)
    print("All demos complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
