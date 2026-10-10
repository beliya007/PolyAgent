"""
工具注册表 (ToolRegistry)

基于 hello-agents 第七章:
- register_tool(tool):注册一个 Tool
- get_tool(name):按名获取
- list_tools():列出所有工具名
- get_tools_description():渲染所有工具为 LLM 可读文本
- execute_tool(name, **kwargs):执行工具(返回 str)
- execute_tool_with_result(name, **kwargs):执行工具(返回 ToolResult)
- 同时提供异步入口 execute_tool_async,给上层 asyncio 流程使用
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from tools.base import BaseTool, ToolResult


class ToolRegistry:
    """工具注册中心"""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    # ---------- 注册 ----------
    def register_tool(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            print(f"⚠️ 工具 {tool.name!r} 已注册,将被覆盖")
        self._tools[tool.name] = tool
        print(f"🔧 工具 {tool.name!r} 已注册")

    def unregister_tool(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    # ---------- 查询 ----------
    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def get_tools_description(self, only_names: bool = False) -> str:
        """渲染所有工具描述,供 LLM 选择调用"""
        if not self._tools:
            return "暂无可用工具"
        if only_names:
            return "\n".join([f"- {t.name}" for t in self._tools.values()])
        return "\n".join([t.schema().to_llm_description() for t in self._tools.values()])

    # ---------- 执行(同步,str) ----------
    def execute_tool(self, name: str, params: Any = None, **kwargs) -> str:
        tool = self._tools.get(name)
        if not tool:
            return f"❌ 错误:未找到工具 '{name}'"
        try:
            return tool.run(params, **kwargs)
        except Exception as e:
            return f"❌ 工具 '{name}' 执行失败: {e}"

    # ---------- 执行(同步,ToolResult) ----------
    def execute_tool_with_result(self, name: str, params: Any = None, **kwargs) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.fail(
                error=f"工具 '{name}' 未注册",
                tool_name=name,
            )
        try:
            return tool.run_with_result(params, **kwargs)
        except Exception as e:
            return ToolResult.fail(error=str(e), tool_name=name)

    # ---------- 执行(异步,str) ----------
    async def execute_tool_async(self, name: str, params: Any = None, **kwargs) -> str:
        """原生异步入口:如果有原生异步工具(is_async_native)优先走 aexecute;
        否则把同步 run 丢进默认线程池。"""
        tool = self._tools.get(name)
        if tool is None:
            return f"❌ 错误:未找到工具 '{name}'"
        if tool.is_async_native:
            try:
                merged: Dict[str, Any] = {}
                if isinstance(params, dict):
                    merged.update(params)
                merged.update(kwargs)
                if merged:
                    return await tool.aexecute(**merged)
                return await tool.aexecute()
            except Exception as e:
                return f"❌ 工具 '{name}' 异步执行失败: {e}"
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.execute_tool(name, params, **kwargs),
        )

    async def execute_tool_async_with_result(
        self,
        name: str,
        params: Any = None,
        **kwargs,
    ) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.fail(error=f"工具 '{name}' 未注册", tool_name=name)
        if tool.is_async_native:
            try:
                merged: Dict[str, Any] = {}
                if isinstance(params, dict):
                    merged.update(params)
                merged.update(kwargs)
                if merged:
                    return await tool.arun_with_result(**merged)
                return await tool.arun_with_result()
            except Exception as e:
                return ToolResult.fail(error=str(e), tool_name=name)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.execute_tool_with_result(name, params, **kwargs),
        )

    # ---------- dunder ----------
    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
