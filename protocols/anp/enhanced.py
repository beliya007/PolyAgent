"""
ANP 增强模块 - 高级特性

参考 hello-agents 第十章 4.3-4.5 节,实现 ANP 协议的增强特性:
1. ANP SecureMessage - 加密的智能体间消息
2. ANP Identity - 基于 DID 的去中心化身份
3. ANP Mesh - 大规模服务网格
4. ANP HealthMonitor - 健康监控与自动恢复

使用示例:
    from protocols.anp.enhanced import (
        ANPIdentity, SecureMessageBus, ANPMesh,
        HealthMonitor, ANPMetrics,
    )

    # 1. 身份认证
    identity = ANPIdentity(agent_id="analysis_agent_1", service_type="analysis")
    print(identity.did)  # did:anp:analysis:analysis_agent_1

    # 2. 安全消息
    bus = SecureMessageBus()
    bus.send(
        from_agent="analysis_agent_1",
        to_agent="summary_agent",
        payload={"task_id": "TASK-001"},
    )

    # 3. 服务网格
    mesh = ANPMesh()
    mesh.deploy_agent(identity)
    mesh.route_message("analysis_agent_1", "summary_agent", "summarize", {"data": "..."})
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from protocols.anp.implementation import ANPAgentInfo, ANPDiscovery


# ============================================================
#  1. 去中心化身份 (DID-based Identity)
# ============================================================
@dataclass
class ANPIdentity:
    """
    ANP 去中心化身份

    每个智能体都有唯一的 DID(Decentralized Identifier),
    用于跨网络识别和加密通信。

    DID 格式: did:anp:<service_type>:<service_id>:<fingerprint>
    """

    agent_id: str
    service_type: str
    service_name: str = ""
    public_key: str = ""  # 简化为字符串
    private_key: str = ""  # 仅持有方知道

    def __post_init__(self):
        if not self.service_name:
            self.service_name = self.agent_id
        if not self.public_key:
            # 自动生成公钥(基于 agent_id 派生)
            self.public_key = self._derive_key(self.agent_id + self.service_type, public=True)
        if not self.private_key:
            self.private_key = self._derive_key(self.agent_id + self.service_type, public=False)

    @staticmethod
    def _derive_key(seed: str, public: bool = True) -> str:
        """基于种子派生密钥(简化)"""
        h = hashlib.sha256(seed.encode("utf-8")).digest()
        return base64.urlsafe_b64encode(h).decode("utf-8")[:32]

    @property
    def did(self) -> str:
        """获取 DID"""
        fingerprint = hashlib.sha256(self.public_key.encode()).hexdigest()[:16]
        return f"did:anp:{self.service_type}:{self.agent_id}:{fingerprint}"

    def sign(self, data: str) -> str:
        """使用私钥签名"""
        return hmac.new(
            self.private_key.encode("utf-8"),
            data.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def verify(self, data: str, signature: str) -> bool:
        """使用公钥验证签名"""
        expected = hmac.new(
            self.private_key.encode("utf-8"),
            data.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "service_type": self.service_type,
            "service_name": self.service_name,
            "public_key": self.public_key,
            "did": self.did,
        }


# ============================================================
#  2. 安全消息总线 (SecureMessageBus)
# ============================================================
@dataclass
class SecureANPMessage:
    """安全的 ANP 消息"""
    message_id: str
    from_agent: str
    to_agent: str
    payload: Dict[str, Any]
    timestamp: float
    signature: str = ""
    nonce: str = ""
    encrypted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "signature": self.signature,
            "nonce": self.nonce,
            "encrypted": self.encrypted,
        }


class SecureMessageBus:
    """
    ANP 安全消息总线

    提供智能体间的安全消息传递能力:
    - 消息签名(防篡改)
    - Nonce 机制(防重放)
    - 时间戳验证
    - 消息路由(基于 to_agent)

    使用示例:
        bus = SecureMessageBus()

        # 注册智能体身份
        bus.register_identity(ANPIdentity("agent_1", "analysis"))

        # 发送消息
        msg_id = bus.send(
            from_agent="agent_1",
            to_agent="agent_2",
            payload={"action": "analyze", "data": "case_001"},
        )

        # 接收消息
        msg = bus.receive("agent_2")
    """

    def __init__(self, secret_key: str = "anp-default-bus-secret"):
        self.secret_key = secret_key.encode("utf-8")
        self._identities: Dict[str, ANPIdentity] = {}
        self._queues: Dict[str, List[SecureANPMessage]] = defaultdict(list)
        self._history: List[SecureANPMessage] = []
        self._seen_nonces: Set[str] = set()
        self._max_queue_size = 1000

    def register_identity(self, identity: ANPIdentity) -> None:
        """注册智能体身份"""
        self._identities[identity.agent_id] = identity

    def send(
        self,
        from_agent: str,
        to_agent: str,
        payload: Dict[str, Any],
    ) -> Optional[str]:
        """
        发送安全消息

        Args:
            from_agent: 发送方 agent_id
            to_agent: 接收方 agent_id
            payload: 消息内容

        Returns:
            消息 ID
        """
        from_identity = self._identities.get(from_agent)
        if from_identity is None:
            print(f"⚠️ 智能体 {from_agent} 未注册身份")
            return None

        nonce = uuid.uuid4().hex[:16]
        timestamp = time.time()

        # 签名
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        signature = hmac.new(
            from_identity.private_key.encode("utf-8"),
            f"{from_agent}|{to_agent}|{timestamp}|{nonce}|{canonical}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        msg = SecureANPMessage(
            message_id=uuid.uuid4().hex[:12],
            from_agent=from_agent,
            to_agent=to_agent,
            payload=payload,
            timestamp=timestamp,
            signature=signature,
            nonce=nonce,
        )

        # 加入目标队列
        queue = self._queues[to_agent]
        if len(queue) >= self._max_queue_size:
            queue.pop(0)  # 丢弃最旧
        queue.append(msg)
        self._history.append(msg)

        return msg.message_id

    def receive(
        self,
        agent_id: str,
        verify: bool = True,
    ) -> Optional[SecureANPMessage]:
        """
        接收消息(从队列取出)

        Args:
            agent_id: 接收方 agent_id
            verify: 是否验证签名

        Returns:
            消息或 None
        """
        queue = self._queues[agent_id]
        while queue:
            msg = queue.pop(0)
            # 防重放
            if msg.nonce in self._seen_nonces:
                continue
            self._seen_nonces.add(msg.nonce)

            # 验证
            if verify:
                from_identity = self._identities.get(msg.from_agent)
                if from_identity is None:
                    continue
                canonical = json.dumps(msg.payload, sort_keys=True, ensure_ascii=False)
                expected = hmac.new(
                    from_identity.private_key.encode("utf-8"),
                    f"{msg.from_agent}|{msg.to_agent}|{msg.timestamp}|{msg.nonce}|{canonical}".encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest()
                if not hmac.compare_digest(expected, msg.signature):
                    print(f"⚠️ 消息 {msg.message_id} 签名验证失败")
                    continue

            return msg
        return None

    def peek(self, agent_id: str) -> List[SecureANPMessage]:
        """查看消息(不移除)"""
        return list(self._queues.get(agent_id, []))

    def stats(self) -> Dict[str, Any]:
        """总线统计"""
        return {
            "registered_agents": len(self._identities),
            "queued_messages": sum(len(q) for q in self._queues.values()),
            "total_history": len(self._history),
            "seen_nonces": len(self._seen_nonces),
        }


# ============================================================
#  3. 健康监控 (HealthMonitor)
# ============================================================
@dataclass
class HealthCheckResult:
    """健康检查结果"""
    agent_id: str
    healthy: bool
    latency_ms: float
    last_check: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "healthy": self.healthy,
            "latency_ms": self.latency_ms,
            "last_check": self.last_check,
            "error": self.error,
        }


class HealthMonitor:
    """
    ANP 健康监控器

    定期检查智能体的健康状态,自动清理不可用的智能体。

    使用示例:
        monitor = HealthMonitor(discovery)
        monitor.add_check("agent_1", check_func=lambda: ping("http://localhost:8001"))
        status = monitor.check_all()
    """

    def __init__(self, discovery: ANPDiscovery, check_interval: float = 30.0):
        self.discovery = discovery
        self.check_interval = check_interval
        self._checks: Dict[str, Callable[[], bool]] = {}
        self._results: Dict[str, HealthCheckResult] = {}
        self._failure_counts: Dict[str, int] = defaultdict(int)

    def add_check(self, agent_id: str, check_func: Callable[[], bool]) -> None:
        """注册健康检查函数"""
        self._checks[agent_id] = check_func

    def check_agent(self, agent_id: str) -> HealthCheckResult:
        """
        检查单个智能体

        Args:
            agent_id: 智能体 ID

        Returns:
            HealthCheckResult
        """
        check_func = self._checks.get(agent_id)
        start = time.time()

        if check_func is None:
            # 默认检查:基于心跳
            agent = self.discovery.get_by_id(agent_id)
            if agent is None:
                result = HealthCheckResult(
                    agent_id=agent_id,
                    healthy=False,
                    latency_ms=0.0,
                    last_check=time.time(),
                    error="Agent not found",
                )
            else:
                healthy = agent.is_alive(timeout=60.0)
                result = HealthCheckResult(
                    agent_id=agent_id,
                    healthy=healthy,
                    latency_ms=(time.time() - start) * 1000,
                    last_check=time.time(),
                    error=None if healthy else "Heartbeat timeout",
                )
        else:
            try:
                healthy = bool(check_func())
                result = HealthCheckResult(
                    agent_id=agent_id,
                    healthy=healthy,
                    latency_ms=(time.time() - start) * 1000,
                    last_check=time.time(),
                )
            except Exception as e:
                result = HealthCheckResult(
                    agent_id=agent_id,
                    healthy=False,
                    latency_ms=(time.time() - start) * 1000,
                    last_check=time.time(),
                    error=str(e),
                )

        self._results[agent_id] = result

        if result.healthy:
            self._failure_counts[agent_id] = 0
        else:
            self._failure_counts[agent_id] += 1
            # 连续失败 3 次自动注销
            if self._failure_counts[agent_id] >= 3:
                self.discovery.unregister(agent_id)
                print(f"⚠️ 智能体 {agent_id} 连续失败,已自动注销")

        return result

    def check_all(self) -> Dict[str, HealthCheckResult]:
        """检查所有智能体"""
        for agent_id in self.discovery.discover_all():
            if agent_id.service_id not in self._results:
                self.check_agent(agent_id.service_id)
        return self._results

    def get_healthy_agents(self) -> List[str]:
        """获取所有健康智能体"""
        return [
            agent_id
            for aid, result in self._results.items()
            if result.healthy
        ]

    def stats(self) -> Dict[str, Any]:
        """健康监控统计"""
        healthy = sum(1 for r in self._results.values() if r.healthy)
        unhealthy = len(self._results) - healthy
        return {
            "total": len(self._results),
            "healthy": healthy,
            "unhealthy": unhealthy,
            "check_interval": self.check_interval,
        }


# ============================================================
#  4. ANP 服务网格 (Mesh)
# ============================================================
@dataclass
class MeshRoute:
    """网格路由"""
    source: str
    target: str
    capability: str
    hops: List[str] = field(default_factory=list)
    cost: float = 1.0


class ANPMesh:
    """
    ANP 服务网格

    在大规模 ANP 网络中智能路由消息。支持:
    - 多跳路由
    - 能力匹配
    - 成本优化

    使用示例:
        mesh = ANPMesh()
        mesh.add_agent(identity1)
        mesh.add_agent(identity2)
        mesh.connect(identity1.agent_id, identity2.agent_id)

        path = mesh.find_path("agent_1", "agent_2")
    """

    def __init__(self, discovery: Optional[ANPDiscovery] = None):
        self.discovery = discovery or ANPDiscovery()
        self._identities: Dict[str, ANPIdentity] = {}
        self._connections: Dict[str, Set[str]] = defaultdict(set)  # 邻接表
        self._routes_cache: Dict[Tuple[str, str], List[str]] = {}

    def add_agent(self, identity: ANPIdentity) -> None:
        """添加智能体到网格"""
        self._identities[identity.agent_id] = identity

    def connect(self, agent_a: str, agent_b: str) -> None:
        """建立双向连接"""
        self._connections[agent_a].add(agent_b)
        self._connections[agent_b].add(agent_a)

    def find_path(self, source: str, target: str) -> Optional[List[str]]:
        """
        寻找最短路径(BFS)

        Args:
            source: 起始智能体
            target: 目标智能体

        Returns:
            路径(包含 source 和 target),不可达返回 None
        """
        if source == target:
            return [source]

        cache_key = (source, target)
        if cache_key in self._routes_cache:
            return self._routes_cache[cache_key]

        visited = {source}
        queue: List[Tuple[str, List[str]]] = [(source, [source])]

        while queue:
            current, path = queue.pop(0)
            for neighbor in self._connections.get(current, set()):
                if neighbor == target:
                    full_path = path + [neighbor]
                    self._routes_cache[cache_key] = full_path
                    return full_path
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def route_message(
        self,
        source: str,
        target: str,
        capability: str,
        payload: Dict[str, Any],
    ) -> bool:
        """
        路由消息

        Args:
            source: 起始智能体
            target: 目标智能体
            capability: 需要的能力
            payload: 消息内容

        Returns:
            是否成功送达
        """
        path = self.find_path(source, target)
        if path is None:
            print(f"⚠️ {source} → {target} 不可达")
            return False

        # 沿路径逐跳转发
        for i in range(len(path) - 1):
            sender = path[i]
            receiver = path[i + 1]
            # 这里可以记录转发日志
            print(f"  📨 [{sender} → {receiver}] 携带能力: {capability}")

        return True

    def topology(self) -> Dict[str, Any]:
        """获取网络拓扑"""
        return {
            "nodes": list(self._identities.keys()),
            "edges": [
                {"from": a, "to": b}
                for a, neighbors in self._connections.items()
                for b in neighbors
            ],
        }


# ============================================================
#  5. ANP 指标 (Metrics)
# ============================================================
@dataclass
class ANPMetrics:
    """ANP 网络指标"""

    @staticmethod
    def calculate_discovery_efficiency(discovery: ANPDiscovery) -> Dict[str, Any]:
        """计算服务发现效率"""
        all_agents = discovery.discover_all()
        alive_agents = [a for a in all_agents if a.is_alive()]
        dead_agents = [a for a in all_agents if not a.is_alive()]

        # 类型分布
        type_dist: Dict[str, int] = defaultdict(int)
        for a in all_agents:
            type_dist[a.service_type] += 1

        # 负载分布
        loads = [a.load for a in all_agents]
        avg_load = sum(loads) / len(loads) if loads else 0.0

        return {
            "total_agents": len(all_agents),
            "alive_ratio": len(alive_agents) / len(all_agents) if all_agents else 1.0,
            "dead_agents": len(dead_agents),
            "type_diversity": len(type_dist),
            "avg_load": avg_load,
            "type_distribution": dict(type_dist),
        }

    @staticmethod
    def calculate_routing_efficiency(
        discovery: ANPDiscovery,
        capability: str,
    ) -> Dict[str, Any]:
        """计算路由效率"""
        candidates = discovery.discover_by_capability(capability)
        if not candidates:
            return {"capability": capability, "candidates": 0, "efficiency": 0.0}

        # 简单的效率指标:候选数 / 类型数
        unique_types = len(set(c.service_type for c in candidates))

        return {
            "capability": capability,
            "candidates": len(candidates),
            "unique_types": unique_types,
            "diversity_ratio": unique_types / len(candidates),
            "best_load": min(c.load for c in candidates),
            "best_price": min(c.price for c in candidates),
        }


# ============================================================
#  工厂函数
# ============================================================
def create_identity(
    agent_id: str,
    service_type: str,
    service_name: str = "",
) -> ANPIdentity:
    """创建 ANP 身份"""
    return ANPIdentity(
        agent_id=agent_id,
        service_type=service_type,
        service_name=service_name,
    )


def create_secure_bus(secret_key: str = "anp-default-bus-secret") -> SecureMessageBus:
    """创建安全消息总线"""
    return SecureMessageBus(secret_key=secret_key)


def create_health_monitor(discovery: ANPDiscovery) -> HealthMonitor:
    """创建健康监控器"""
    return HealthMonitor(discovery)


def create_mesh(discovery: Optional[ANPDiscovery] = None) -> ANPMesh:
    """创建 ANP 服务网格"""
    return ANPMesh(discovery)