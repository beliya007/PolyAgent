"""
异步工具执行器 (AsyncToolExecutor)

支持:
  - execute_tool_async       单个工具异步执行
  - execute_tools_parallel   多个工具并行执行
  - execute_chain_async      同步链的异步包装(在线程池里跑整条链)

借鉴 hello-agents 第七章 7.5.4(2) 的设计:
当工具标了 is_async_native=True 时,直接 await tool.aexecute() 走原生异步路径;
否则回退到 ThreadPoolExecutor,把同步 execute 包在线程里,避免阻塞事件循环。

使用示例::

    executor = AsyncToolExecutor(registry, max_workers=4)
    r = await executor.execute_tool_async("search", "Python")
    results = await executor.execute_tools_parallel([
        {"tool_name": "search", "input_data": "Python"},
        {"tool_name": "calculator", "input_data": "2+3"},
    ])
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from tools.base import BaseTool, ToolResult
from tools.registry import ToolRegistry


class AsyncToolExecutor:
    """异步 / 并行工具执行器"""

    def __init__(self, registry: ToolRegistry, max_workers: int = 4):
        if max_workers <= 0:
            raise ValueError("max_workers 必须 > 0")
        self.registry = registry
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="tool-worker",
        )

    # ---------- 单工具 ----------
    async def execute_tool_async(self, tool_name: str, input_data: Any) -> str:
        """异步执行单个工具(返回字符串)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.registry.execute_tool(tool_name, input_data),
        )

    async def execute_tool_smart(
        self,
        tool_name: str,
        input_data: Any = None,
        /,
        **kwargs,
    ) -> str:
        """智能选择异步路径

        - 若工具存在且 is_async_native=True,直接调 arun()(真异步)
        - 否则退回到 ThreadPoolExecutor 跑 execute()
        """
        tool = self.registry.get_tool(tool_name)
        if tool is None:
            return f"❌ 工具 '{tool_name}' 未注册"
        if tool.is_async_native and hasattr(tool, "arun"):
            merged: Dict[str, Any] = {}
            if isinstance(input_data, dict):
                merged.update(input_data)
            merged.update(kwargs)
            if merged:
                return await tool.arun(**merged)
            return await tool.arun(input_data or "")
        return await self.execute_tool_async(tool_name, input_data or "")

    async def execute_tool_with_result(self, tool_name: str, input_data: Any) -> ToolResult:
        """异步执行单个工具(返回 ToolResult)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.registry.execute_tool_with_result(tool_name, input_data),
        )

    # ---------- 多工具并行 ----------
    async def execute_tools_parallel(
        self,
        tasks: Sequence[Dict[str, Any]],
    ) -> List[str]:
        """并行执行多个工具任务

        Args:
            tasks: 每个元素是 {"tool_name": str, "input_data": Any} 字典
        Returns:
            按入参顺序返回的字符串结果列表
        """
        if not tasks:
            return []
        print(f"🚀 开始并行执行 {len(tasks)} 个工具任务")
        coros = [self.execute_tool_smart(t["tool_name"], t.get("input_data")) for t in tasks]
        results = await asyncio.gather(*coros)
        print(f"✅ 所有工具任务执行完成")
        return list(results)

    async def execute_tools_parallel_with_result(
        self,
        tasks: Sequence[Dict[str, Any]],
    ) -> List[ToolResult]:
        if not tasks:
            return []
        coros = [self.execute_tool_with_result(t["tool_name"], t.get("input_data")) for t in tasks]
        return list(await asyncio.gather(*coros))

    # ---------- 整链异步 ----------
    async def execute_chain_async(
        self,
        chain_name: str,
        manager,                      # ToolChainManager,避免循环引用
        initial_input: Any = "",
        context: Optional[Dict[str, Any]] = None,
        stop_on_error: bool = True,
    ) -> ToolResult:
        """异步执行一条已注册的链(在线程池里跑同步链逻辑,避免阻塞)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: manager.execute(
                chain_name,
                initial_input=initial_input,
                context=context,
                stop_on_error=stop_on_error,
            ),
        )

    # ---------- 清理 ----------
    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)

    def __del__(self):
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
