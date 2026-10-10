"""
Agent 基类

基于 polyagent 第七章:
- 所有 Agent 都继承自 BaseAgent
- 提供消息历史 / 上下文管理 / LLM 调用封装
- 支持原生 function calling(tools)
- 支持流式(stream_run)和同步(run)两种执行
- 内置 ToolChainManager / AsyncToolExecutor,链式 + 并行即开即用(7.5.4)
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple

from core.message import Message, ToolCall, ToolResult
from llm_client import PolyAgentLLM, tool_to_openai_schema
from tools.async_executor import AsyncToolExecutor
from tools.chain import ToolChain, ToolChainManager
from tools.registry import ToolRegistry


class BaseAgent(ABC):
    """所有 Agent 的基类"""

    def __init__(
        self,
        name: str,
        llm: PolyAgentLLM,
        system_prompt: Optional[str] = None,
        tool_registry: Optional[ToolRegistry] = None,
        max_async_workers: int = 4,
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt or "你是一个有用的AI助手。"
        self.tool_registry = tool_registry
        self.history: List[Message] = []
        self._max_async_workers = max_async_workers

        # 链式 + 并行的执行器(懒加载,首次访问时基于 tool_registry 构造)
        self._chain_manager: Optional[ToolChainManager] = None
        self._async_executor: Optional[AsyncToolExecutor] = None

        # 兼容旧代码:可能使用 agents.message.Message
        try:
            from agents.message import Message as LegacyMessage  # noqa: F401
        except ImportError:
            pass
        print(f"✅ {name} 初始化完成")

    # ===================================================================
    #  同步链式 + 异步并行(polyagent 7.5.4)
    # ===================================================================
    @property
    def chain_manager(self) -> Optional[ToolChainManager]:
        """ToolChainManager:管理预注册的工具链。

        若需要"开箱即用"的业务链(knowledge → history → template → validate
        → save),可用 tools.chains.register_default_chains(self.tool_registry)
        一次性注册。
        """
        if self._chain_manager is None and self.tool_registry is not None:
            self._chain_manager = ToolChainManager(self.tool_registry)
        return self._chain_manager

    @property
    def async_executor(self) -> Optional[AsyncToolExecutor]:
        """AsyncToolExecutor:异步 / 并行工具执行器。"""
        if self._async_executor is None and self.tool_registry is not None:
            self._async_executor = AsyncToolExecutor(
                self.tool_registry, max_workers=self._max_async_workers
            )
        return self._async_executor

    def register_chain(self, chain: ToolChain) -> None:
        """注册一条自定义工具链到本 Agent 的 manager。"""
        mgr = self.chain_manager
        if mgr is None:
            raise RuntimeError("请先配置 tool_registry,再注册工具链")
        mgr.register(chain)

    def run_chain(
        self,
        chain_name: str,
        initial_input: str = "",
        context: Optional[Dict[str, Any]] = None,
        stop_on_error: bool = True,
    ):
        """同步顺序执行一条已注册的工具链。

        返回 ToolResult;可直接 .to_str() 给 LLM,或编程消费 .data / .error。
        """
        mgr = self.chain_manager
        if mgr is None:
            return ToolResult.fail("ToolChainManager 未初始化(缺少 tool_registry)")
        return mgr.execute(
            chain_name,
            initial_input=initial_input,
            context=context,
            stop_on_error=stop_on_error,
        )

    async def run_chain_async(
        self,
        chain_name: str,
        initial_input: str = "",
        context: Optional[Dict[str, Any]] = None,
        stop_on_error: bool = True,
    ):
        """异步执行一条已注册的工具链(整条链丢进线程池,不阻塞事件循环)"""
        exe = self.async_executor
        mgr = self.chain_manager
        if exe is None or mgr is None:
            return ToolResult.fail("AsyncToolExecutor / ToolChainManager 未初始化")
        return await exe.execute_chain_async(
            chain_name, mgr, initial_input=initial_input,
            context=context, stop_on_error=stop_on_error,
        )

    async def run_tools_parallel(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """并行执行多个工具任务(异步),适合无依赖的工具一起调。

        用法:
            results = await agent.run_tools_parallel([
                {"tool_name": "search_docs",   "input_data": {"query": "release plan"}},
                {"tool_name": "calculator",    "input_data": "15*8+32"},
            ])
        """
        exe = self.async_executor
        if exe is None:
            return ["❌ AsyncToolExecutor 未初始化"] * max(len(tasks), 1)
        return await exe.execute_tools_parallel(tasks)

    async def run_tools_parallel_with_result(
        self,
        tasks: List[Dict[str, Any]],
    ):
        """并行执行多个工具任务(异步),返回 List[ToolResult]"""
        exe = self.async_executor
        if exe is None:
            return [ToolResult.fail("AsyncToolExecutor 未初始化")] * max(len(tasks), 1)
        return await exe.execute_tools_parallel_with_result(tasks)

    def shutdown_async(self, wait: bool = False) -> None:
        """关闭底层 ThreadPoolExecutor(建议在 Agent 生命周期结束时调用)"""
        if self._async_executor is not None:
            self._async_executor.shutdown(wait=wait)

    # ---------------- 历史管理 ----------------

    def add_message(self, message: Message) -> None:
        self.history.append(message)

    def clear_history(self) -> None:
        self.history.clear()

    def get_history(self) -> List[Message]:
        return list(self.history)

    def to_openai_messages(self, user_input: str) -> List[dict]:
        """构造发给 LLM 的 OpenAI 消息列表"""
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in self.history:
            messages.append(msg.to_openai())
        messages.append({"role": "user", "content": user_input})
        return messages

    # ---------------- 工具 schema ----------------

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """获取当前可用工具的 OpenAI schema 列表"""
        if not self.has_tools():
            return []
        return [tool_to_openai_schema(t) for t in self.tool_registry._tools.values()]

    # ---------------- LLM 调用 ----------------

    def call_llm(self, messages: List[dict], **kwargs) -> Optional[str]:
        return self.llm.invoke(messages, **kwargs)

    def call_llm_with_tools(self, messages: List[dict], **kwargs):
        return self.llm.invoke_with_tools(messages, **kwargs)

    # ---------------- 工具执行 ----------------

    def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """执行单个工具调用(透传到新 ToolRegistry,自动捕获异常)"""
        if not self.tool_registry:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content="",
                success=False,
                error="未配置工具注册表",
            )
        result = self.tool_registry.execute_tool_with_result(tool_call.name, tool_call.arguments)
        # 兼容 core.message.ToolResult 的字段(tool_call_id / name / content / success / error)
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content=str(result.data) if result.success else "",
            success=result.success,
            error=result.error,
        )

    # ---------------- 抽象方法 ----------------

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """执行入口:接收用户输入,返回最终答复"""
        raise NotImplementedError

    @abstractmethod
    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """流式执行入口"""
        raise NotImplementedError

    # ---------------- 工具 ----------------

    def has_tools(self) -> bool:
        return self.tool_registry is not None and len(self.tool_registry) > 0
