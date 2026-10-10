"""
ANP (Agent Network Protocol) 子模块

参考 polyagent 第十章 4 节设计,实现大规模智能体网络协议。

ANP 协议设计理念:
- 去中心化服务发现
- 大规模网络中的智能体路由
- 动态扩展(新智能体自动加入)

ANP 核心概念:
- 服务注册:智能体向网络注册自己的能力
- 服务发现:根据能力描述查找合适的智能体
- 智能路由:根据负载、价格等指标选择最合适的智能体
- 任务分发:将任务分配给选定的智能体

主要组件:
- ANPDiscovery: 服务发现中心(基于 DID 的服务注册/发现)
- ANPRouter: 智能路由器(负载均衡、任务分发)
- ANPTool: 工具包装器,让 Agent 通过 ANP 调用其他服务
- enhanced: 增强模块(身份认证/安全消息/健康监控/服务网格)
"""

from protocols.anp.implementation import (
    ANPDiscovery,
    ANPRouter,
    ANPAgentInfo,
    register_service,
    discover_services_by_type,
)

# 增强模块(可选)
try:
    from protocols.anp.enhanced import (
        ANPIdentity,
        SecureMessageBus,
        SecureANPMessage,
        HealthMonitor,
        HealthCheckResult,
        ANPMesh,
        MeshRoute,
        ANPMetrics,
        create_identity,
        create_secure_bus,
        create_health_monitor,
        create_mesh,
    )
    _ENHANCED_AVAILABLE = True
except ImportError:
    _ENHANCED_AVAILABLE = False

__all__ = [
    "ANPDiscovery",
    "ANPRouter",
    "ANPAgentInfo",
    "register_service",
    "discover_services_by_type",
    # 增强
    "ANPIdentity",
    "SecureMessageBus",
    "SecureANPMessage",
    "HealthMonitor",
    "HealthCheckResult",
    "ANPMesh",
    "MeshRoute",
    "ANPMetrics",
    "create_identity",
    "create_secure_bus",
    "create_health_monitor",
    "create_mesh",
]