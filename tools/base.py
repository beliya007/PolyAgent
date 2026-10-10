"""
工具基类

基于 polyagent 第七章 Tool 设计:
- 每个工具是一段可调用的能力
- 工具元数据(name / description / parameters schema)供 LLM 检索
- run() / arun() 方法支持两种调用方式:
    1. kwargs 风格: tool.run(name='alice')
    2. dict 风格:   tool.run({'name': 'alice'})
- schema() 导出 OpenAI function calling 兼容的 schema
- 同步链式: ToolChain 串接多个工具,每步引用上一步输出(polyagent 7.5.4)
- 异步执行: AsyncToolExecutor + tool.aexecute/arun(polyagent 7.5.4)
- ToolResult 提供结构化结果,便于链式组合与统一错误处理
"""
from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from llm_client import tool_to_openai_schema


@dataclass
class ToolParameter:
    """工具参数定义"""

    name: str
    type: str = "string"           # string / number / integer / boolean / array / object
    description: str = ""
    required: bool = False
    default: Any = None
    enum: Optional[List[Any]] = None


@dataclass
class ToolSchema:
    """工具的完整描述(供 LLM 理解)"""

    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

    def to_llm_description(self) -> str:
        """渲染为 LLM 可读的文本描述"""
        lines = [f"- {self.name}: {self.description}"]
        if self.parameters:
            lines.append("  参数:")
            for p in self.parameters:
                req = "(必填)" if p.required else "(可选)"
                enums = f",可选值:{p.enum}" if p.enum else ""
                lines.append(f"    - {p.name}{req}: {p.type}{enums} — {p.description}")
        if self.examples:
            lines.append("  示例:")
            for ex in self.examples:
                lines.append(f"    {ex}")
        return "\n".join(lines)

    def to_openai_schema(self) -> Dict[str, Any]:
        """导出 OpenAI function calling schema"""
        return tool_to_openai_schema(self)


@dataclass
class ToolSchemaAdapter:
    """把 BaseTool 转成 ToolSchema 的适配器(简化调用)"""
    tool: "BaseTool"

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.tool.name,
            description=self.tool.description,
            parameters=list(self.tool.parameters),
        )

    def to_openai_schema(self) -> Dict[str, Any]:
        return tool_to_openai_schema(self)


# ============================================================
#  通用结构化返回
# ============================================================
@dataclass
class ToolResult:
    """工具调用结构化结果(链式/并行场景必备)

    同步/异步路径都建议返回该类型,既支持 .to_str() 渲染给 LLM,
    也支持 .success/.data/.error 编程式消费。
    """
    success: bool
    data: Any = None
    error: Optional[str] = None
    tool_name: str = ""
    elapsed_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_str(self) -> str:
        """渲染为字符串(LLM 友好);success 时取 data,失败时返回错误描述。"""
        if self.success:
            if self.data is None:
                return ""
            if isinstance(self.data, str):
                return self.data
            try:
                return json.dumps(self.data, ensure_ascii=False)
            except (TypeError, ValueError):
                return str(self.data)
        return f"❌ {self.tool_name or 'tool'} 执行失败:{self.error}"

    @classmethod
    def ok(cls, data: Any = None, tool_name: str = "", **meta) -> "ToolResult":
        return cls(success=True, data=data, tool_name=tool_name, metadata=dict(meta))

    @classmethod
    def fail(cls, error: str, tool_name: str = "", **meta) -> "ToolResult":
        return cls(success=False, error=error, tool_name=tool_name, metadata=dict(meta))


# ============================================================
#  工具基类
# ============================================================
class BaseTool(ABC):
    """所有自定义工具的基类

    三层调用入口,语义清晰且可渐进增强:
        1. execute(**kwargs)        —— 同步抽象方法,子类必须实现
        2. aexecute(**kwargs)       —— 异步入口,默认在线程池里跑 execute,
                                       IO 密集型工具可重写以获得真正异步收益
        3. run/arun(**kwargs)       —— 统一调用入口(支持 dict / kwargs / 单字符串),
                                       执行参数归一化与必填校验

    标记 is_async_native=True 表示工具有原生异步实现(例如 httpx.AsyncClient),
    AsyncToolExecutor 会优先调用 aexecute() 而非线程池。
    """

    name: str = ""
    description: str = ""
    parameters: List[ToolParameter] = []
    is_async_native: bool = False   # 由 IO 密集型工具置 True

    def __init__(self) -> None:
        if not self.name:
            raise ValueError(f"{type(self).__name__}.name 不能为空")

    # ---------- 抽象:同步执行入口 ----------
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """工具执行入口,返回字符串结果(JSON 字符串或纯文本)"""
        raise NotImplementedError

    # ---------- 默认异步(可在子类重写以获得真正非阻塞 IO) ----------
    async def aexecute(self, **kwargs) -> str:
        """异步执行入口

        默认实现:把同步 execute() 丢进默认线程池。
        对于真正的 IO 密集型工具(如 httpx.AsyncClient),可以重写本方法以
        避免线程开销,例如::

            async def aexecute(self, url: str) -> str:
                async with httpx.AsyncClient() as cli:
                    r = await cli.get(url)
                return r.text
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.execute(**kwargs))

    # ============================================================
    #  参数归一化辅助
    # ============================================================
    @staticmethod
    def _normalize_params(
        tool: "BaseTool",
        params: Optional[Union[Dict[str, Any], str]],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """把 dict / str / kwargs 统一成 dict,并校验必填参数。"""
        if isinstance(params, dict):
            merged: Dict[str, Any] = {**params, **kwargs}
        elif isinstance(params, str):
            first = tool.parameters[0].name if tool.parameters else "input"
            merged = {**{first: params}, **kwargs}
        else:
            merged = dict(kwargs)

        for p in tool.parameters:
            if p.required and p.name not in merged:
                raise ValueError(f"工具 {tool.name} 缺少必填参数: {p.name}")
        return merged

    # ---------- 同步:统一调用入口 ----------
    def run(
        self,
        params: Optional[Union[Dict[str, Any], str]] = None,
        /,
        **kwargs,
    ) -> str:
        """同步统一调用入口。支持:
            tool.run({'name': 'alice'})
            tool.run(name='alice')
        """
        merged = self._normalize_params(self, params, kwargs)
        return self.execute(**merged)

    # ---------- 异步:统一调用入口 ----------
    async def arun(
        self,
        params: Optional[Union[Dict[str, Any], str]] = None,
        /,
        **kwargs,
    ) -> str:
        """异步统一调用入口,等价于 await tool.aexecute(**normalized)"""
        merged = self._normalize_params(self, params, kwargs)
        return await self.aexecute(**merged)

    # ---------- 结构化返回(便于链式/并行场景编程消费) ----------
    def run_with_result(
        self,
        params: Optional[Union[Dict[str, Any], str]] = None,
        /,
        **kwargs,
    ) -> ToolResult:
        """同步执行并返回 ToolResult(包含耗时)。"""
        import time
        t0 = time.perf_counter()
        try:
            merged = self._normalize_params(self, params, kwargs)
            data = self.execute(**merged)
            return ToolResult.ok(
                data=data,
                tool_name=self.name,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as e:
            return ToolResult.fail(
                error=str(e),
                tool_name=self.name,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

    async def arun_with_result(
        self,
        params: Optional[Union[Dict[str, Any], str]] = None,
        /,
        **kwargs,
    ) -> ToolResult:
        """异步执行并返回 ToolResult(包含耗时)。"""
        import time
        t0 = time.perf_counter()
        try:
            merged = self._normalize_params(self, params, kwargs)
            data = await self.aexecute(**merged)
            return ToolResult.ok(
                data=data,
                tool_name=self.name,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as e:
            return ToolResult.fail(
                error=str(e),
                tool_name=self.name,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

    # ---------- 元数据 ----------
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=list(self.parameters),
        )

    def to_openai_schema(self) -> Dict[str, Any]:
        """导出 OpenAI function calling schema"""
        return tool_to_openai_schema(self)


class FunctionTool(BaseTool):
    """将普通函数包装为 Tool 的便捷类"""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., str],
        parameters: Optional[List[ToolParameter]] = None,
        is_async_native: bool = False,
    ):
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters or []
        self.is_async_native = is_async_native
        super().__init__()

    def execute(self, **kwargs) -> str:
        return self.func(**kwargs)
