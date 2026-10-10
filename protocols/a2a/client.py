"""
A2A Client - 智能体客户端

参考 polyagent 第十章 3.3 节设计，实现：
- 连接到其他 A2A Server
- 调用其他智能体的技能
- 任务管理（创建/查询/取消）
- 智能体发现（获取 agent card）

使用示例：
    from protocols.a2a import A2AClient

    # 创建客户端连接到研究员智能体
    client = A2AClient("http://localhost:5000")

    # 获取智能体信息
    card = client.get_agent_card()
    print(card['name'], card['skills'])

    # 调用技能
    result = client.execute_skill("research", "research AI在医疗领域的应用")
    print(result)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import asyncio

# HTTP 客户端可用性
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


@dataclass
class A2AResponse:
    """A2A 响应"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    status_code: int = 200
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "status_code": self.status_code,
            "metadata": self.metadata or {},
        }


class A2AClient:
    """
    A2A 智能体客户端

    连接到远程 A2A Server 并调用其技能。

    使用示例：
        client = A2AClient("http://localhost:5000")

        # 获取智能体卡片
        card = client.get_agent_card()

        # 执行技能
        response = client.execute_skill("research", "research AI")
    """

    def __init__(
        self,
        agent_url: str,
        timeout: float = 30.0,
        use_http: bool = True,
        local_server: Optional["A2AServer"] = None,
    ):
        """
        初始化 A2A Client

        Args:
            agent_url: 远程智能体 URL，如 "http://localhost:5000"
            timeout: 请求超时（秒）
            use_http: 是否使用 HTTP（False 则使用内存模拟）
            local_server: 本地 A2A Server 实例（用于进程内直接调用）
        """
        self.agent_url = agent_url.rstrip("/")
        self.timeout = timeout
        self.use_http = use_http
        self.local_server = local_server

        # 缓存智能体卡片
        self._agent_card: Optional[Dict[str, Any]] = None

        # 检查 HTTP 客户端可用性
        if self.use_http and not HTTPX_AVAILABLE and local_server is None:
            print(f"⚠️ httpx 未安装,A2AClient 将使用内存模拟模式")
            self.use_http = False

    # ============================================================
    #  智能体发现
    # ============================================================
    def get_agent_card(self) -> Dict[str, Any]:
        """
        获取智能体卡片

        Returns:
            智能体元信息字典
        """
        if self._agent_card is not None:
            return self._agent_card

        if self.local_server is not None:
            # 本地内存模式:直接从 server 获取
            self._agent_card = self.local_server.get_agent_card()
        elif self.use_http:
            self._agent_card = self._http_get_agent_card()
        else:
            # 内存模拟模式返回基础信息
            self._agent_card = {
                "name": f"RemoteAgent@{self.agent_url}",
                "description": "Remote A2A Agent",
                "version": "1.0.0",
                "capabilities": {},
                "skills": [],
                "endpoints": {},
            }

        return self._agent_card

    def _http_get_agent_card(self) -> Dict[str, Any]:
        """HTTP 获取智能体卡片"""
        try:
            import httpx
            url = f"{self.agent_url}/.well-known/agent.json"
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {
                "name": f"UnknownAgent@{self.agent_url}",
                "description": f"Failed to fetch agent card: {e}",
                "version": "0.0.0",
                "capabilities": {},
                "skills": [],
                "endpoints": {},
                "error": str(e),
            }

    def list_skills(self) -> List[Dict[str, Any]]:
        """列出远程智能体的所有技能"""
        card = self.get_agent_card()
        return card.get("skills", [])

    def has_skill(self, skill_name: str) -> bool:
        """检查远程智能体是否有指定技能"""
        skills = self.list_skills()
        return any(s.get("name") == skill_name for s in skills)

    # ============================================================
    #  技能调用
    # ============================================================
    def execute_skill(
        self,
        skill_name: str,
        input_data: Any = "",
    ) -> Dict[str, Any]:
        """
        同步调用远程技能

        Args:
            skill_name: 技能名
            input_data: 输入数据

        Returns:
            执行结果
        """
        if self.local_server is not None:
            return self.local_server.execute_skill(skill_name, input_data)
        if self.use_http:
            return self._http_execute_skill(skill_name, input_data)
        else:
            return self._memory_execute_skill(skill_name, input_data)

    def _http_execute_skill(
        self,
        skill_name: str,
        input_data: Any,
    ) -> Dict[str, Any]:
        """HTTP 调用技能"""
        try:
            import httpx
            url = f"{self.agent_url}/a2a/execute"
            payload = {
                "skill": skill_name,
                "input": input_data if isinstance(input_data, str) else json.dumps(input_data),
            }
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
                return {
                    "success": True,
                    "result": result.get("result"),
                    "status": result.get("status"),
                    "task_id": result.get("task_id"),
                    "artifacts": result.get("artifacts", []),
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def _memory_execute_skill(
        self,
        skill_name: str,
        input_data: Any,
    ) -> Dict[str, Any]:
        """
        内存模式模拟执行（用于本地测试）

        注意：实际的远程执行需要 HTTP 模式
        """
        # 简单模拟返回
        return {
            "success": True,
            "result": f"[Mock] Skill '{skill_name}' executed with input: {input_data}",
            "status": "completed",
            "task_id": "mock_task_id",
            "artifacts": [],
            "warning": "HTTP mode not available, using mock execution",
        }

    async def aexecute_skill(
        self,
        skill_name: str,
        input_data: Any = "",
    ) -> Dict[str, Any]:
        """
        异步调用远程技能

        Args:
            skill_name: 技能名
            input_data: 输入数据

        Returns:
            执行结果
        """
        if not self.use_http:
            return self._memory_execute_skill(skill_name, input_data)

        try:
            import httpx
            url = f"{self.agent_url}/a2a/execute"
            payload = {
                "skill": skill_name,
                "input": input_data if isinstance(input_data, str) else json.dumps(input_data),
            }
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
                return {
                    "success": True,
                    "result": result.get("result"),
                    "status": result.get("status"),
                    "task_id": result.get("task_id"),
                    "artifacts": result.get("artifacts", []),
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    # ============================================================
    #  任务管理
    # ============================================================
    def create_task(
        self,
        skill_name: str,
        input_data: Any = "",
    ) -> Dict[str, Any]:
        """
        创建任务（异步执行）

        Args:
            skill_name: 技能名
            input_data: 输入数据

        Returns:
            任务信息
        """
        if not self.use_http:
            return self._memory_execute_skill(skill_name, input_data)

        try:
            import httpx
            url = f"{self.agent_url}/a2a/task"
            payload = {
                "skill": skill_name,
                "input": input_data,
            }
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def get_task(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态"""
        if not self.use_http:
            return {"error": "HTTP mode not available"}

        try:
            import httpx
            url = f"{self.agent_url}/a2a/task/{task_id}"
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    # ============================================================
    #  信息
    # ============================================================
    def __repr__(self) -> str:
        return f"A2AClient(agent_url='{self.agent_url}', use_http={self.use_http})"