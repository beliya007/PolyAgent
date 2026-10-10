"""
ANPTool - ANP 工具包装器

参考 hello-agents 第十章 4.2 节设计，将 ANP 服务发现封装成 BaseTool，
让 Agent 可以通过 ANP 发现和调用网络中的其他服务。

核心功能：
1. 服务注册：将自己的能力注册到 ANP 网络
2. 服务发现：通过能力描述查找服务
3. 智能路由：根据负载、价格选择最优服务
4. 任务分发：向选定的服务发送请求

使用示例：
    from protocols.anp import ANPDiscovery, register_service
    from protocols.anp.tool import ANPTool

    # 创建 ANPTool（会自动管理 ANPDiscovery）
    anp_tool = ANPTool(name="anp")

    # 添加到 Agent
    agent.add_tool(anp_tool)

    # Agent 可以使用以下 action：
    # - register_service: 注册服务
    # - discover_services: 发现服务
    # - route_task: 智能路由任务
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from tools.base import BaseTool, ToolParameter

from protocols.anp.implementation import (
    ANPDiscovery,
    ANPRouter,
    ANPAgentInfo,
)


class ANPTool(BaseTool):
    """
    ANP 工具包装器

    将 ANP 服务发现与路由功能封装成 BaseTool。

    特性：
    - 内置 ANPDiscovery 服务发现
    - 支持多种负载均衡策略
    - 完整的网络管理功能

    支持的 action：
    - register_service: 注册服务
    - discover_services: 发现服务
    - route_task: 智能路由任务
    - list_services: 列出所有服务
    - heartbeat: 更新心跳
    """

    def __init__(
        self,
        name: str = "anp",
        description: str = "ANP 网络工具 - 智能体网络协议",
        discovery: Optional[ANPDiscovery] = None,
    ):
        """
        初始化 ANPTool

        Args:
            name: 工具名称
            description: 工具描述
            discovery: 可选的服务发现实例（默认创建新的）
        """
        self.name = name
        self.description = description

        self.discovery = discovery or ANPDiscovery()
        self.router = ANPRouter(self.discovery)

        # BaseTool 属性
        self.parameters = []

        super().__init__()

    # ============================================================
    #  BaseTool 接口
    # ============================================================
    def execute(self, **kwargs) -> str:
        """
        执行 ANP 操作

        支持的 actions：
        - register_service: 注册服务
        - discover_services: 发现服务（按类型或能力）
        - route_task: 智能路由任务
        - list_services: 列出所有服务
        - heartbeat: 更新心跳
        - unregister: 注销服务
        - stats: 统计信息
        """
        action = kwargs.get("action", "")

        if action == "register_service":
            return self._register_service(kwargs)
        elif action == "discover_services":
            return self._discover_services(kwargs)
        elif action == "route_task":
            return self._route_task(kwargs)
        elif action == "list_services":
            return self.discovery.list_services()
        elif action == "heartbeat":
            return self._heartbeat(kwargs)
        elif action == "unregister":
            return self._unregister(kwargs)
        elif action == "stats":
            return json.dumps(self.discovery.stats(), ensure_ascii=False, indent=2)
        else:
            return self._help_message()

    def _help_message(self) -> str:
        """帮助信息"""
        return """
ANP 网络工具支持的 action:

1. register_service - 注册服务
   参数: service_id, service_name, service_type, capabilities, endpoint, [metadata]

2. discover_services - 发现服务
   参数: service_type (或 capability)

3. route_task - 智能路由任务
   参数: capability, [strategy]

5. list_services - 列出所有服务

6. heartbeat - 更新心跳
   参数: service_id, [load]

7. unregister - 注销服务
   参数: service_id

8. stats - 统计信息
"""

    # ============================================================
    #  注册服务
    # ============================================================
    def _register_service(self, kwargs: Dict[str, Any]) -> str:
        """注册服务"""
        service_id = kwargs.get("service_id", "")
        if not service_id:
            return "❌ 缺少 service_id 参数"

        try:
            agent_info = self.discovery.register(
                service_id=service_id,
                service_name=kwargs.get("service_name", service_id),
                service_type=kwargs.get("service_type", "general"),
                capabilities=kwargs.get("capabilities", []),
                endpoint=kwargs.get("endpoint", ""),
                metadata=kwargs.get("metadata", {}),
                version=kwargs.get("version", "1.0.0"),
            )
            return f"✅ 服务注册成功: {agent_info.service_name} ({agent_info.service_id})\nDID: {agent_info.did}"
        except Exception as e:
            return f"❌ 注册失败: {e}"

    # ============================================================
    #  发现服务
    # ============================================================
    def _discover_services(self, kwargs: Dict[str, Any]) -> str:
        """发现服务"""
        service_type = kwargs.get("service_type", "")
        capability = kwargs.get("capability", "")

        if service_type:
            agents = self.discovery.discover_by_type(service_type)
        elif capability:
            agents = self.discovery.discover_by_capability(capability)
        else:
            agents = self.discovery.discover_all()

        if not agents:
            return f"未发现匹配的服务（service_type={service_type}, capability={capability}）"

        lines = [f"发现 {len(agents)} 个服务:"]
        for a in agents:
            lines.append(f"\n- {a.service_name} ({a.service_id})")
            lines.append(f"  类型: {a.service_type}")
            lines.append(f"  端点: {a.endpoint}")
            lines.append(f"  能力: {', '.join(a.capabilities)}")
            lines.append(f"  负载: {a.load:.1%}")

        return "\n".join(lines)

    # ============================================================
    #  智能路由
    # ============================================================
    def _route_task(self, kwargs: Dict[str, Any]) -> str:
        """智能路由任务"""
        capability = kwargs.get("capability", "")
        service_type = kwargs.get("service_type", "")
        strategy = kwargs.get("strategy", "least_loaded")

        if not capability and not service_type:
            return "❌ 缺少 capability 或 service_type 参数"

        try:
            if capability:
                agent = self.router.route_by_capability(capability, strategy=strategy)
            else:
                agent = self.router.route_by_type(service_type, strategy=strategy)

            if agent is None:
                return "未找到匹配的服务"

            return json.dumps({
                "selected_agent": agent.to_dict(),
                "strategy": strategy,
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"❌ 路由失败: {e}"

    # ============================================================
    #  心跳
    # ============================================================
    def _heartbeat(self, kwargs: Dict[str, Any]) -> str:
        """更新心跳"""
        service_id = kwargs.get("service_id", "")
        load = kwargs.get("load")

        if not service_id:
            return "❌ 缺少 service_id 参数"

        success = self.discovery.heartbeat(service_id, load=load)
        if success:
            return f"✅ 心跳更新成功: {service_id}"
        return f"❌ 服务不存在: {service_id}"

    # ============================================================
    #  注销
    # ============================================================
    def _unregister(self, kwargs: Dict[str, Any]) -> str:
        """注销服务"""
        service_id = kwargs.get("service_id", "")
        if not service_id:
            return "❌ 缺少 service_id 参数"

        success = self.discovery.unregister(service_id)
        if success:
            return f"✅ 服务已注销: {service_id}"
        return f"❌ 服务不存在: {service_id}"

    # ============================================================
    #  便捷方法（直接调用）
    # ============================================================
    def register_service(
        self,
        service_id: str,
        service_name: str,
        service_type: str,
        capabilities: List[str],
        endpoint: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ANPAgentInfo:
        """注册服务的便捷方法"""
        return self.discovery.register(
            service_id=service_id,
            service_name=service_name,
            service_type=service_type,
            capabilities=capabilities,
            endpoint=endpoint,
            metadata=metadata,
        )

    def discover_services(self, capability: str = "", service_type: str = "") -> List[ANPAgentInfo]:
        """发现服务的便捷方法"""
        if service_type:
            return self.discovery.discover_by_type(service_type)
        elif capability:
            return self.discovery.discover_by_capability(capability)
        return self.discovery.discover_all()


def create_anp_tool(
    name: str = "anp",
    description: str = "ANP 网络工具",
) -> ANPTool:
    """工厂函数：创建 ANPTool"""
    return ANPTool(name=name, description=description)