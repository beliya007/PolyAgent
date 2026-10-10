"""
ANP (Agent Network Protocol) 核心实现

参考 polyagent 第十章 4 节设计，提供：
- ANPDiscovery: 服务发现中心
- ANPRouter: 智能路由器
- ANPAgentInfo: 智能体信息
- 服务注册与发现
- 任务分发与负载均衡

使用示例：
    from protocols.anp import ANPDiscovery, register_service

    # 创建服务发现中心
    discovery = ANPDiscovery()

    # 注册智能体服务
    register_service(
        discovery=discovery,
        service_id="nlp_agent_1",
        service_name="NLP处理专家A",
        service_type="nlp",
        capabilities=["text_analysis", "sentiment_analysis"],
        endpoint="http://localhost:8001",
        metadata={"load": 0.3, "price": 0.01}
    )

    # 发现服务
    services = discovery.discover_by_type("nlp")
    print(f"发现 {len(services)} 个 NLP 服务")

    # 智能路由（选择最合适的服务）
    from protocols.anp import ANPRouter
    router = ANPRouter(discovery)
    best = router.route_by_capability("text_analysis")
    print(f"最佳服务: {best.service_name}")
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class LoadBalanceStrategy(str, Enum):
    """负载均衡策略"""
    ROUND_ROBIN = "round_robin"           # 轮询
    LEAST_LOADED = "least_loaded"         # 最少负载
    LOWEST_PRICE = "lowest_price"         # 最低价格
    RANDOM = "random"                     # 随机
    CAPABILITY_MATCH = "capability_match"  # 能力匹配


@dataclass
class ANPAgentInfo:
    """ANP 智能体信息"""
    service_id: str
    service_name: str
    service_type: str
    capabilities: List[str]
    endpoint: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    did: str = ""                          # 去中心化标识符（简化）
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    load: float = 0.0                      # 当前负载 0.0-1.0
    price: float = 0.0                     # 调用价格
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_id": self.service_id,
            "service_name": self.service_name,
            "service_type": self.service_type,
            "capabilities": self.capabilities,
            "endpoint": self.endpoint,
            "metadata": self.metadata,
            "did": self.did,
            "registered_at": self.registered_at,
            "last_heartbeat": self.last_heartbeat,
            "load": self.load,
            "price": self.price,
            "version": self.version,
        }

    def update_heartbeat(self, load: Optional[float] = None) -> None:
        """更新心跳"""
        self.last_heartbeat = time.time()
        if load is not None:
            self.load = max(0.0, min(1.0, load))

    def is_alive(self, timeout: float = 60.0) -> bool:
        """检查是否存活（基于心跳超时）"""
        return (time.time() - self.last_heartbeat) < timeout


class ANPDiscovery:
    """
    ANP 服务发现中心

    提供去中心化的智能体服务注册与发现功能。

    使用示例：
        discovery = ANPDiscovery()

        # 注册服务
        discovery.register(
            service_id="agent_1",
            service_name="Text Analysis Expert",
            service_type="nlp",
            capabilities=["text_analysis", "sentiment"],
            endpoint="http://localhost:8001"
        )

        # 发现服务
        agents = discovery.discover_by_type("nlp")
        agents = discovery.discover_by_capability("sentiment")
        agent = discovery.get_by_id("agent_1")
    """

    def __init__(self):
        """初始化服务发现中心"""
        self._agents: Dict[str, ANPAgentInfo] = {}
        print("🌐 ANPDiscovery 服务发现中心已创建")

    # ============================================================
    #  服务注册
    # ============================================================
    def register(
        self,
        service_id: str,
        service_name: str,
        service_type: str,
        capabilities: List[str],
        endpoint: str,
        metadata: Optional[Dict[str, Any]] = None,
        version: str = "1.0.0",
    ) -> ANPAgentInfo:
        """
        注册智能体服务

        Args:
            service_id: 服务唯一ID
            service_name: 服务名称
            service_type: 服务类型（如 nlp, vision, math）
            capabilities: 能力列表
            endpoint: 服务端点
            metadata: 元数据（如 load, price, version）
            version: 服务版本

        Returns:
            ANPAgentInfo
        """
        # 生成 DID（去中心化标识符）
        did = f"did:anp:{service_type}:{service_id}"

        metadata = metadata or {}

        agent_info = ANPAgentInfo(
            service_id=service_id,
            service_name=service_name,
            service_type=service_type,
            capabilities=capabilities,
            endpoint=endpoint,
            metadata=metadata,
            did=did,
            version=version,
            load=metadata.get("load", 0.0),
            price=metadata.get("price", 0.0),
        )

        self._agents[service_id] = agent_info
        print(f"  ✅ 服务注册: {service_name} ({service_type}) @ {endpoint}")
        return agent_info

    def unregister(self, service_id: str) -> bool:
        """注销服务"""
        if service_id in self._agents:
            agent = self._agents.pop(service_id)
            print(f"  ❌ 服务注销: {agent.service_name}")
            return True
        return False

    # ============================================================
    #  服务发现
    # ============================================================
    def get_by_id(self, service_id: str) -> Optional[ANPAgentInfo]:
        """通过 ID 获取服务"""
        return self._agents.get(service_id)

    def discover_all(self) -> List[ANPAgentInfo]:
        """发现所有服务"""
        return list(self._agents.values())

    def discover_by_type(self, service_type: str) -> List[ANPAgentInfo]:
        """通过服务类型发现"""
        return [a for a in self._agents.values() if a.service_type == service_type]

    def discover_by_capability(self, capability: str) -> List[ANPAgentInfo]:
        """通过能力发现"""
        return [a for a in self._agents.values() if capability in a.capabilities]

    def discover_by_endpoint(self, endpoint: str) -> List[ANPAgentInfo]:
        """通过端点发现"""
        return [a for a in self._agents.values() if a.endpoint == endpoint]

    # ============================================================
    #  心跳与健康检查
    # ============================================================
    def heartbeat(
        self,
        service_id: str,
        load: Optional[float] = None,
    ) -> bool:
        """更新心跳"""
        agent = self._agents.get(service_id)
        if agent is None:
            return False
        agent.update_heartbeat(load=load)
        return True

    def cleanup_dead(self, timeout: float = 60.0) -> int:
        """清理死亡的服务"""
        dead_ids = [
            sid for sid, a in self._agents.items()
            if not a.is_alive(timeout)
        ]
        for sid in dead_ids:
            self.unregister(sid)
        return len(dead_ids)

    # ============================================================
    #  状态
    # ============================================================
    def count(self) -> int:
        """服务总数"""
        return len(self._agents)

    def stats(self) -> Dict[str, Any]:
        """统计信息"""
        type_counts = {}
        for a in self._agents.values():
            type_counts[a.service_type] = type_counts.get(a.service_type, 0) + 1

        return {
            "total_agents": len(self._agents),
            "alive_agents": sum(1 for a in self._agents.values() if a.is_alive()),
            "type_distribution": type_counts,
            "total_capabilities": sum(len(a.capabilities) for a in self._agents.values()),
        }

    def list_services(self) -> str:
        """列出所有服务（人类可读）"""
        if not self._agents:
            return "暂无注册服务"

        lines = [f"已注册服务 ({len(self._agents)} 个):"]
        for a in self._agents.values():
            lines.append(f"\n- {a.service_name} ({a.service_id})")
            lines.append(f"  类型: {a.service_type}")
            lines.append(f"  端点: {a.endpoint}")
            lines.append(f"  能力: {', '.join(a.capabilities)}")
            lines.append(f"  负载: {a.load:.1%} | 价格: {a.price:.4f}")
            lines.append(f"  DID: {a.did}")

        return "\n".join(lines)


class ANPRouter:
    """
    ANP 智能路由器

    在多个满足条件的智能体中选择最优的一个。
    支持多种负载均衡策略。

    使用示例：
        router = ANPRouter(discovery)

        # 根据能力选择最优服务
        best = router.route_by_capability("text_analysis")

        # 根据策略选择
        best = router.route_by_capability("sentiment", strategy="least_loaded")
    """

    def __init__(self, discovery: ANPDiscovery):
        self.discovery = discovery
        self._round_robin_index: Dict[str, int] = {}

    def route_by_type(
        self,
        service_type: str,
        strategy: str = "least_loaded",
    ) -> Optional[ANPAgentInfo]:
        """
        根据服务类型路由

        Args:
            service_type: 服务类型
            strategy: 负载均衡策略

        Returns:
            最优服务
        """
        candidates = self.discovery.discover_by_type(service_type)
        if not candidates:
            return None
        return self._select_by_strategy(candidates, strategy)

    def route_by_capability(
        self,
        capability: str,
        strategy: str = "least_loaded",
    ) -> Optional[ANPAgentInfo]:
        """
        根据能力路由

        Args:
            capability: 能力名称
            strategy: 负载均衡策略

        Returns:
            最优服务
        """
        candidates = self.discovery.discover_by_capability(capability)
        if not candidates:
            return None
        return self._select_by_strategy(candidates, strategy)

    def route_by_id(self, service_id: str) -> Optional[ANPAgentInfo]:
        """通过 ID 路由"""
        return self.discovery.get_by_id(service_id)

    def _select_by_strategy(
        self,
        candidates: List[ANPAgentInfo],
        strategy: str,
    ) -> ANPAgentInfo:
        """
        根据策略选择最优服务

        Args:
            candidates: 候选服务列表
            strategy: 策略

        Returns:
            最优服务
        """
        if not candidates:
            raise ValueError("候选服务为空")

        if strategy == "round_robin":
            return self._round_robin(candidates)
        elif strategy == "least_loaded":
            return min(candidates, key=lambda a: a.load)
        elif strategy == "lowest_price":
            return min(candidates, key=lambda a: a.price)
        elif strategy == "random":
            import random
            return random.choice(candidates)
        else:
            # 默认最少负载
            return min(candidates, key=lambda a: a.load)

    def _round_robin(self, candidates: List[ANPAgentInfo]) -> ANPAgentInfo:
        """轮询策略"""
        key = ",".join(sorted(a.service_id for a in candidates))
        idx = self._round_robin_index.get(key, 0)
        selected = candidates[idx % len(candidates)]
        self._round_robin_index[key] = idx + 1
        return selected

    def update_load(self, service_id: str, load: float) -> None:
        """更新服务负载"""
        self.discovery.heartbeat(service_id, load=load)

    def stats(self) -> Dict[str, Any]:
        """统计信息"""
        return self.discovery.stats()


# ============================================================
#  便捷函数
# ============================================================
def register_service(
    discovery: ANPDiscovery,
    service_id: str,
    service_name: str,
    service_type: str,
    capabilities: List[str],
    endpoint: str,
    metadata: Optional[Dict[str, Any]] = None,
    version: str = "1.0.0",
) -> ANPAgentInfo:
    """注册服务的便捷函数"""
    return discovery.register(
        service_id=service_id,
        service_name=service_name,
        service_type=service_type,
        capabilities=capabilities,
        endpoint=endpoint,
        metadata=metadata,
        version=version,
    )


def discover_services_by_type(
    discovery: ANPDiscovery,
    service_type: str,
) -> List[ANPAgentInfo]:
    """按类型发现服务的便捷函数"""
    return discovery.discover_by_type(service_type)


def create_anp_discovery() -> ANPDiscovery:
    """工厂函数：创建 ANPDiscovery"""
    return ANPDiscovery()


def create_anp_router(discovery: ANPDiscovery) -> ANPRouter:
    """工厂函数：创建 ANPRouter"""
    return ANPRouter(discovery)