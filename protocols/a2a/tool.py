"""
A2ATool - A2A 工具包装器

参考 polyagent 第十章 3.4 节设计，将 A2A 服务包装成 BaseTool，
让 Agent 可以像调用普通工具一样调用其他 A2A 智能体的服务。

核心特性：
1. 自动展开 A2A 服务器的技能到独立工具
2. 支持同步和异步调用
3. 名称前缀避免冲突

使用示例：
    from protocols.a2a import A2ATool

    # 方式1: 连接到远程 A2A 服务
    tech_expert_tool = A2ATool(
        agent_url="http://localhost:6000",
        name="tech_expert",
        description="技术专家Agent"
    )

    # 添加到 Agent
    agent.add_tool(tech_expert_tool)

    # 方式2: 包装本地 A2A Server 实例
    from protocols.a2a import A2AServer
    server = A2AServer(name="local_agent")
    @server.skill("greet")
    def greet(text: str) -> str:
        return f"Hello, {text}!"

    local_tool = A2ATool.from_server(server)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from tools.base import BaseTool, ToolParameter


@dataclass
class ExpandedA2ASkill:
    """展开后的 A2A 技能"""
    original_name: str
    prefixed_name: str
    description: str
    parameters: Dict[str, Any]
    a2a_tool: "A2ATool"


class A2ATool(BaseTool):
    """
    A2A 工具包装器

    将 A2A 智能体的技能包装成 BaseTool，供 Agent 调用。

    特性：
    - 自动展开远程智能体的技能
    - 自动添加前缀避免冲突
    - 支持同步和异步调用
    - 可直接连接到远程 A2A Server
    """

    def __init__(
        self,
        agent_url: str,
        name: str = "a2a_agent",
        description: str = "A2A 智能体工具",
        timeout: float = 30.0,
        auto_discover: bool = True,
        local_server: Optional["A2AServer"] = None,
    ):
        """
        初始化 A2ATool

        Args:
            agent_url: 远程智能体 URL
            name: 工具名称前缀
            description: 工具描述
            timeout: 请求超时
            auto_discover: 是否自动发现技能
            local_server: 本地 A2A Server 实例（用于离线测试）
        """
        self.agent_url = agent_url
        self.name = name
        self.description = description
        self.timeout = timeout
        self.local_server = local_server

        # 客户端
        if local_server is None:
            from protocols.a2a.client import A2AClient
            self._client = A2AClient(agent_url, timeout=timeout)
        else:
            self._client = None

        # 展开的技能
        self._expanded_skills: List[ExpandedA2ASkill] = []
        self._skill_wrappers: Dict[str, "A2ASkillWrapper"] = {}

        # 初始化参数（BaseTool 要求）
        self.parameters = []

        super().__init__()

        # 自动发现技能
        if auto_discover:
            self._discover_skills()

    def _discover_skills(self) -> None:
        """发现并展开 A2A 智能体的技能"""
        if self.local_server is not None:
            # 本地模式
            skills = self.local_server.list_skills()
        elif self._client is not None:
            # 远程模式
            skills = self._client.list_skills()
        else:
            skills = []

        self._expanded_skills = []
        self._skill_wrappers = {}

        for skill in skills:
            original_name = skill.get("name", "")
            prefixed_name = f"{self.name}_{original_name}" if self.name else original_name

            expanded = ExpandedA2ASkill(
                original_name=original_name,
                prefixed_name=prefixed_name,
                description=skill.get("description", ""),
                parameters=skill.get("parameters", {}),
                a2a_tool=self,
            )

            wrapper = A2ASkillWrapper(
                original_name=original_name,
                prefixed_name=prefixed_name,
                description=expanded.description,
                parameters=expanded.parameters,
                parent=self,
            )

            self._expanded_skills.append(expanded)
            self._skill_wrappers[prefixed_name] = wrapper

    def get_expanded_skills(self) -> List[ExpandedA2ASkill]:
        """获取展开的技能列表"""
        return self._expanded_skills

    def get_skill_names(self) -> List[str]:
        """获取所有展开后的技能名称"""
        return list(self._skill_wrappers.keys())

    def call_expanded_skill(
        self,
        prefixed_name: str,
        input_data: Any = "",
    ) -> str:
        """
        调用展开的技能

        Args:
            prefixed_name: 带前缀的技能名
            input_data: 输入数据

        Returns:
            执行结果
        """
        if prefixed_name not in self._skill_wrappers:
            raise ValueError(f"未知技能: {prefixed_name}")

        wrapper = self._skill_wrappers[prefixed_name]
        return wrapper.run(input_data)

    def execute(self, **kwargs) -> str:
        """
        执行技能（BaseTool 接口）

        支持 action 接口：
        - {"action": "list_skills"} - 列出技能
        - {"action": "call_skill", "skill_name": "...", "input": "..."} - 调用技能
        - {"action": "get_agent_card"} - 获取智能体卡片
        """
        action = kwargs.get("action", "")

        if action == "list_skills":
            return self._list_skills()
        elif action == "call_skill":
            skill_name = kwargs.get("skill_name", "")
            input_data = kwargs.get("input", "")
            return self._call_skill(skill_name, input_data)
        elif action == "get_agent_card":
            return self._get_agent_card()
        else:
            return self._list_skills()

    def _list_skills(self) -> str:
        """列出技能"""
        lines = [f"A2A 智能体技能 ({self.name}):"]
        lines.append(f"URL: {self.agent_url}")

        for skill in self._expanded_skills:
            lines.append(f"\n- {skill.prefixed_name}: {skill.description}")
            params = skill.parameters.get("properties", {})
            if params:
                lines.append("  参数:")
                for param_name, param_info in params.items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    lines.append(f"    - {param_name} ({param_type}): {param_desc}")

        return "\n".join(lines)

    def _call_skill(self, skill_name: str, input_data: Any) -> str:
        """调用技能"""
        # 查找技能
        wrapper = self._skill_wrappers.get(skill_name)
        if wrapper is None:
            return f"❌ 未知技能: {skill_name}"

        try:
            return wrapper.run(input_data)
        except Exception as e:
            return f"❌ 调用失败: {e}"

    def _get_agent_card(self) -> str:
        """获取智能体卡片"""
        if self.local_server is not None:
            card = self.local_server.get_agent_card()
        elif self._client is not None:
            card = self._client.get_agent_card()
        else:
            card = {}

        return json.dumps(card, ensure_ascii=False, indent=2)

    def to_openai_schema(self) -> List[Dict[str, Any]]:
        """导出所有展开技能的 OpenAI schema"""
        schemas = []
        for wrapper in self._skill_wrappers.values():
            schemas.append(wrapper.to_openai_schema())
        return schemas

    @classmethod
    def from_server(
        cls,
        server: "A2AServer",
        name: Optional[str] = None,
        description: str = "",
    ) -> "A2ATool":
        """
        从本地 A2A Server 创建 A2ATool

        Args:
            server: 本地 A2A Server 实例
            name: 工具名称前缀（默认使用 server.name）
            description: 工具描述

        Returns:
            A2ATool 实例
        """
        return cls(
            agent_url=f"local://{server.name}",
            name=name or server.name,
            description=description or server.description,
            local_server=server,
        )


class A2ASkillWrapper(BaseTool):
    """
    单个 A2A 技能的包装器

    将 A2A 智能体上的单个技能包装成 BaseTool。
    """

    def __init__(
        self,
        original_name: str,
        prefixed_name: str,
        description: str,
        parameters: Dict[str, Any],
        parent: A2ATool,
    ):
        self.original_name = original_name
        self.prefixed_name = prefixed_name
        self.parameters_schema = parameters
        self.a2a_tool = parent

        # BaseTool 属性
        self.name = prefixed_name
        self.description = description
        self.parameters = self._build_parameters()

        super().__init__()

    def _build_parameters(self) -> List[ToolParameter]:
        """从 parameters schema 构建参数列表"""
        params = []
        properties = self.parameters_schema.get("properties", {})

        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "string")
            if param_type == "number":
                param_type = "number"
            elif param_type == "integer":
                param_type = "integer"
            elif param_type == "boolean":
                param_type = "boolean"
            elif param_type == "array":
                param_type = "array"
            elif param_type == "object":
                param_type = "object"
            else:
                param_type = "string"

            params.append(ToolParameter(
                name=param_name,
                type=param_type,
                description=param_info.get("description", ""),
                required=param_name in self.parameters_schema.get("required", []),
            ))

        return params

    def execute(self, **kwargs) -> str:
        """执行 A2A 技能"""
        # 合并参数
        if isinstance(kwargs.get("input"), dict):
            input_data = kwargs["input"]
        elif len(kwargs) == 1 and "input" in kwargs:
            input_data = kwargs["input"]
        else:
            input_data = kwargs if kwargs else ""

        # 调用 A2A 技能
        if self.a2a_tool.local_server is not None:
            # 本地模式
            result = self.a2a_tool.local_server.execute_skill(
                self.original_name,
                input_data,
            )
        elif self.a2a_tool._client is not None:
            # 远程模式
            result = self.a2a_tool._client.execute_skill(
                self.original_name,
                input_data,
            )
        else:
            return "❌ A2ATool 未配置客户端"

        # 提取结果
        if result.get("success"):
            return self._format_result(result.get("result", ""))
        return f"❌ 调用失败: {result.get('error', '未知错误')}"

    def _format_result(self, result: Any) -> str:
        """格式化结果"""
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)