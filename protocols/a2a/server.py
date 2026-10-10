"""
A2A Server - 智能体服务端

参考 polyagent 第十章 3.2 节、3.3 节设计，实现：
- 智能体注册（技能、能力）
- 接收并处理来自其他智能体的请求
- 任务生命周期管理
- 基于 FastAPI 的 HTTP 服务（可选）

使用示例：
    from protocols.a2a import A2AServer

    server = A2AServer(
        name="researcher",
        description="负责搜索和分析资料的Agent",
        capabilities={"research": ["search", "analyze"]}
    )

    @server.skill("research")
    def do_research(text: str) -> str:
        import re
        match = re.search(r'research\s+(.+)', text, re.IGNORECASE)
        topic = match.group(1).strip() if match else text
        return str({"topic": topic, "findings": f"研究结果: {topic}"})

    # 启动 HTTP 服务
    server.run(host="localhost", port=5000)

    # 或仅在内存中注册技能（用于单元测试）
    result = server.execute_skill("research", "research AI在医疗领域的应用")
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from functools import wraps
import inspect


class TaskStatus(str, Enum):
    """A2A 任务状态"""
    CREATED = "created"           # 任务已创建
    NEGOTIATING = "negotiating"   # 协商中
    ACCEPTED = "accepted"         # 已接受
    WORKING = "working"           # 执行中
    COMPLETED = "completed"       # 已完成
    FAILED = "failed"             # 失败
    CANCELED = "canceled"         # 已取消


@dataclass
class SkillDefinition:
    """技能定义"""
    name: str
    description: str
    func: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Artifact:
    """A2A 工件 - 任务执行的产物"""
    name: str
    description: str = ""
    data: Any = None
    mime_type: str = "text/plain"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "data": self.data,
            "mimeType": self.mime_type,
        }


@dataclass
class Task:
    """A2A 任务"""
    id: str
    skill: str
    input: Any
    status: TaskStatus = TaskStatus.CREATED
    result: Any = None
    error: Optional[str] = None
    artifacts: List[Artifact] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "skill": self.skill,
            "input": self.input,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


class A2AServer:
    """
    A2A 智能体服务端

    提供智能体间的点对点通信能力。支持：
    - 技能注册（@server.skill 装饰器）
    - 任务创建与执行
    - HTTP 服务（基于 FastAPI，可选）
    - 同步技能执行（execute_skill）

    使用示例：
        server = A2AServer(name="my_agent", description="My Agent")

        @server.skill("greet")
        def greet(text: str) -> str:
            return f"Hello, {text}!"

        # 同步执行
        result = server.execute_skill("greet", "World")
        # 异步任务
        task = server.create_task("greet", "World")
        result = server.execute_task(task)
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        version: str = "1.0.0",
        capabilities: Optional[Dict[str, List[str]]] = None,
    ):
        """
        初始化 A2A Server

        Args:
            name: 智能体名称（唯一标识）
            description: 智能体描述
            version: 版本号
            capabilities: 能力字典，如 {"math": ["add", "subtract"]}
        """
        self.name = name
        self.description = description
        self.version = version
        self.capabilities = capabilities or {}

        self._skills: Dict[str, SkillDefinition] = {}
        self._tasks: Dict[str, Task] = {}

        print(f"🤖 A2A Server 创建: {name} v{version}")

    # ============================================================
    #  技能注册
    # ============================================================
    def skill(
        self,
        name: Optional[str] = None,
        description: str = "",
    ) -> Callable:
        """
        技能装饰器

        使用示例：
            @server.skill("add")
            def add_numbers(text: str) -> str:
                '''加法计算'''
                return ...
        """
        def decorator(func: Callable) -> Callable:
            skill_name = name or func.__name__
            skill_desc = description or func.__doc__ or ""

            # 从函数签名提取参数 schema
            sig = inspect.signature(func)
            properties = {}
            required = []

            for param_name, param in sig.parameters.items():
                if param_name in ("return", "self", "cls"):
                    continue
                param_type = "string"
                if param.annotation == float or param.annotation == int:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == list:
                    param_type = "array"
                elif param.annotation == dict:
                    param_type = "object"

                properties[param_name] = {
                    "type": param_type,
                    "description": param.annotation.__name__ if hasattr(param.annotation, "__name__") else "any",
                }
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)

            self._skills[skill_name] = SkillDefinition(
                name=skill_name,
                description=skill_desc,
                func=func,
                parameters={
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            )

            print(f"  ✅ 技能注册: {skill_name}")

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

        return decorator

    def register_skill(
        self,
        name: str,
        func: Callable,
        description: str = "",
    ) -> None:
        """
        直接注册技能（不通过装饰器）

        Args:
            name: 技能名
            func: 技能函数
            description: 技能描述
        """
        self._skills[name] = SkillDefinition(
            name=name,
            description=description or func.__doc__ or "",
            func=func,
        )
        print(f"  ✅ 技能注册: {name}")

    # ============================================================
    #  技能查询
    # ============================================================
    def list_skills(self) -> List[Dict[str, Any]]:
        """
        列出所有技能

        Returns:
            技能列表
        """
        return [
            {
                "name": s.name,
                "description": s.description,
                "parameters": s.parameters,
            }
            for s in self._skills.values()
        ]

    def get_skill(self, name: str) -> Optional[SkillDefinition]:
        """获取指定技能"""
        return self._skills.get(name)

    def has_skill(self, name: str) -> bool:
        """检查是否有指定技能"""
        return name in self._skills

    def get_agent_card(self) -> Dict[str, Any]:
        """
        获取智能体卡片（用于服务发现）

        Returns:
            智能体元信息
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "capabilities": self.capabilities,
            "skills": self.list_skills(),
            "endpoints": {
                "execute_skill": "/a2a/execute",
                "create_task": "/a2a/task",
                "get_task": "/a2a/task/{task_id}",
                "agent_card": "/.well-known/agent.json",
            },
        }

    # ============================================================
    #  任务管理
    # ============================================================
    def create_task(self, skill_name: str, input_data: Any) -> Task:
        """
        创建任务

        Args:
            skill_name: 技能名
            input_data: 输入数据

        Returns:
            Task 对象
        """
        task = Task(
            id=str(uuid.uuid4())[:8],
            skill=skill_name,
            input=input_data,
            status=TaskStatus.CREATED,
        )
        self._tasks[task.id] = task
        return task

    def execute_task(self, task: Task) -> Task:
        """
        执行任务

        Args:
            task: Task 对象

        Returns:
            更新后的 Task 对象
        """
        if task.skill not in self._skills:
            task.status = TaskStatus.FAILED
            task.error = f"未知技能: {task.skill}"
            task.updated_at = time.time()
            return task

        task.status = TaskStatus.WORKING
        task.updated_at = time.time()

        try:
            skill = self._skills[task.skill]

            # 解析输入
            if isinstance(task.input, str):
                # 简单解析 "skill_name arg1 arg2" 格式
                parts = re.split(r'\s+', task.input.strip(), maxsplit=1)
                if len(parts) == 2 and parts[0].lower() == task.skill.lower():
                    actual_input = parts[1]
                else:
                    actual_input = task.input
            else:
                actual_input = task.input

            # 调用技能
            if isinstance(actual_input, dict):
                result = skill.func(**actual_input)
            else:
                result = skill.func(actual_input)

            # 设置结果
            task.result = result
            task.status = TaskStatus.COMPLETED

            # 创建工件
            if result is not None:
                task.artifacts.append(Artifact(
                    name=f"{task.skill}_result",
                    description=f"Result of skill {task.skill}",
                    data=result,
                ))

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)

        task.updated_at = time.time()
        return task

    def execute_skill(
        self,
        skill_name: str,
        input_data: Any,
    ) -> Dict[str, Any]:
        """
        同步执行技能（便捷方法）

        Args:
            skill_name: 技能名
            input_data: 输入数据

        Returns:
            执行结果字典(包含 success 字段,保持与 A2AClient 一致)
        """
        task = self.create_task(skill_name, input_data)
        task = self.execute_task(task)

        success = task.status == TaskStatus.COMPLETED
        return {
            "success": success,
            "task_id": task.id,
            "status": task.status.value,
            "result": task.result,
            "error": task.error,
            "artifacts": [a.to_dict() for a in task.artifacts],
        }

    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务"""
        return self._tasks.get(task_id)

    # ============================================================
    #  HTTP 服务（基于 FastAPI，可选）
    # ============================================================
    def run(self, host: str = "localhost", port: int = 5000, use_fastapi: bool = False) -> None:
        """
        启动 HTTP 服务

        Args:
            host: 主机地址
            port: 端口
            use_fastapi: 是否使用 FastAPI（需要安装 fastapi 和 uvicorn）
        """
        if use_fastapi:
            self._run_fastapi(host, port)
        else:
            # 仅输出信息，不实际启动服务
            print(f"\n🚀 A2A Server '{self.name}' 配置完成")
            print(f"   端点: http://{host}:{port}")
            print(f"   智能体卡片: http://{host}:{port}/.well-known/agent.json")
            print(f"   技能数量: {len(self._skills)}")
            print(f"\n💡 提示: 使用 A2AClient 连接到其他 A2A Server 进行点对点通信")
            print(f"   或使用 use_fastapi=True 启动 HTTP 服务")

    def _run_fastapi(self, host: str, port: int) -> None:
        """使用 FastAPI 启动 HTTP 服务"""
        try:
            from fastapi import FastAPI, HTTPException
            import uvicorn
        except ImportError:
            print("⚠️ 未安装 fastapi/uvicorn,无法启动 HTTP 服务")
            return

        app = FastAPI(title=f"A2A Server: {self.name}")

        @app.get("/.well-known/agent.json")
        async def agent_card():
            return self.get_agent_card()

        @app.get("/a2a/skills")
        async def list_skills():
            return self.list_skills()

        @app.post("/a2a/execute")
        async def execute_skill(request: Dict[str, str]):
            skill_name = request.get("skill", "")
            input_data = request.get("input", "")
            if not self.has_skill(skill_name):
                raise HTTPException(status_code=404, detail=f"未知技能: {skill_name}")
            return self.execute_skill(skill_name, input_data)

        @app.post("/a2a/task")
        async def create_task(request: Dict[str, Any]):
            task = self.create_task(
                skill_name=request.get("skill", ""),
                input_data=request.get("input"),
            )
            task = self.execute_task(task)
            return task.to_dict()

        @app.get("/a2a/task/{task_id}")
        async def get_task(task_id: str):
            task = self.get_task(task_id)
            if not task:
                raise HTTPException(status_code=404, detail="任务不存在")
            return task.to_dict()

        uvicorn.run(app, host=host, port=port)