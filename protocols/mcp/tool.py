"""
MCPTool - MCP 工具包装器

参考 hello-agents 第十章 2.4 节设计，将 MCP 服务器提供的工具包装成 BaseTool，
实现工具自动展开机制，让 Agent 可以像调用普通工具一样调用 MCP 工具。

核心特性：
1. 自动展开：MCP 服务器提供的所有工具自动展开为独立工具
2. 名称前缀：为避免工具名称冲突，自动添加前缀
3. 类型转换：自动处理参数类型转换
4. 统一接口：与 Agent 工具系统无缝集成

使用示例：
    from protocols.mcp import MCPTool

    # 方式1: 使用内置演示服务器（Memory 传输）
    mcp_tool = MCPTool()

    # 方式2: 连接到自定义服务器（Stdio 传输）
    mcp_tool = MCPTool(
        name="calculator",
        description="数学计算工具",
        server_command=["python", "my_mcp_server.py"]
    )

    # 方式3: 连接到社区服务器（npx 方式）
    mcp_tool = MCPTool(
        name="filesystem",
        description="文件系统访问",
        server_command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "."]
    )

    # 列出展开后的工具
    expanded_tools = mcp_tool.get_expanded_tools()
    for tool in expanded_tools:
        print(f"- {tool.name}: {tool.description}")

    # 调用工具
    result = mcp_tool.call_expanded_tool("calculator_add", {"a": 10, "b": 20})
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from tools.base import BaseTool, ToolParameter, ToolResult


@dataclass
class ExpandedTool:
    """展开后的 MCP 工具"""
    original_name: str  # 原始名称
    prefixed_name: str   # 带前缀的名称
    description: str
    input_schema: Dict[str, Any]
    mcp_tool: "MCPTool"  # 引用父 MCPTool

    def to_parameters(self) -> List[ToolParameter]:
        """转换为 ToolParameter 列表"""
        params = []
        schema = self.input_schema.get("properties", {})

        for param_name, param_info in schema.items():
            params.append(ToolParameter(
                name=param_name,
                type=param_info.get("type", "string"),
                description=param_info.get("description", ""),
                required=param_name in self.input_schema.get("required", []),
            ))

        return params


class MCPToolWrapper(BaseTool):
    """
    单个 MCP 工具的包装器

    将 MCP 服务器上的单个工具包装成 BaseTool，
    供 Agent 系统使用。
    """

    def __init__(
        self,
        original_name: str,
        prefixed_name: str,
        description: str,
        input_schema: Dict[str, Any],
        parent: "MCPTool",
    ):
        self.original_name = original_name
        self.prefixed_name = prefixed_name
        self.description = description
        self.input_schema = input_schema
        self.mcp_tool = parent

        # BaseTool 属性
        self.name = prefixed_name
        self.description = description
        self.parameters = self._build_parameters()

        super().__init__()

    def _build_parameters(self) -> List[ToolParameter]:
        """从 input_schema 构建参数列表"""
        params = []
        schema = self.input_schema.get("properties", {})

        for param_name, param_info in schema.items():
            param_type = param_info.get("type", "string")
            # 映射类型
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
                required=param_name in self.input_schema.get("required", []),
            ))

        return params

    def execute(self, **kwargs) -> str:
        """执行 MCP 工具"""
        try:
            result = asyncio.run(
                self.mcp_tool._async_call_tool(self.original_name, kwargs)
            )
            return self._format_result(result)
        except Exception as e:
            return f"工具执行失败: {e}"

    def _format_result(self, result: Any) -> str:
        """格式化结果"""
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)


class MCPTool(BaseTool):
    """
    MCP 工具包装器

    将 MCP 服务器提供的所有工具展开为独立的 BaseTool，
    实现与 Agent 工具系统的无缝集成。

    特性：
    - 自动展开 MCP 服务器的工具
    - 支持多种传输方式（Memory/Stdio/HTTP）
    - 自动添加前缀避免名称冲突
    - 线程安全的执行
    """

    def __init__(
        self,
        name: str = "mcp",
        description: str = "MCP 工具服务器",
        server_command: Optional[List[str]] = None,
        timeout: int = 30,
    ):
        """
        初始化 MCPTool

        Args:
            name: 工具名称前缀，用于避免多服务器工具名冲突
            description: 工具描述
            server_command: 服务器命令，如 ["python", "server.py"] 或 ["npx", "-y", "@modelcontextprotocol/server-filesystem", "."]
            timeout: 超时时间（秒）
        """
        self.name = name
        self.description = description
        self.server_command = server_command
        self.timeout = timeout

        # 展开的工具
        self._expanded_tools: List[ExpandedTool] = []
        self._tool_wrappers: Dict[str, MCPToolWrapper] = {}
        self._mcp_client = None
        self._connected = False

        # 内部状态
        self._async_client = None

        super().__init__()

        # 如果指定了命令，自动连接并展开工具
        if server_command is not None:
            self._ensure_connected()

    def _ensure_connected(self):
        """确保已连接到 MCP 服务器"""
        if not self._connected:
            try:
                self._discover_tools()
            except Exception as e:
                print(f"⚠️ 连接 MCP 服务器失败: {e}")

    def _discover_tools(self):
        """发现并展开 MCP 服务器的工具"""
        from protocols.mcp.client import MCPClient

        try:
            # 创建客户端并连接
            self._async_client = MCPClient(self.server_command)
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def _discover():
                async with self._async_client as client:
                    tools = await client.list_tools()
                    self._expand_tools(tools)

            loop.run_until_complete(_discover())
            self._connected = True

        except Exception as e:
            print(f"⚠️ 工具发现失败: {e}")
            # 使用内置 Mock 服务器
            self._use_builtin_server()

    def _use_builtin_server(self):
        """使用内置演示服务器"""
        from protocols.mcp.client import MCPClient, MockMCPSession

        async def _discover():
            async with MCPClient() as client:
                tools = await client.list_tools()
                self._expand_tools(tools)

        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_discover())
        self._connected = True

    def _expand_tools(self, tools: List[Dict[str, Any]]):
        """
        展开工具列表

        Args:
            tools: MCP 工具列表
        """
        self._expanded_tools = []
        self._tool_wrappers = {}

        for tool in tools:
            original_name = tool.get("name", "")
            prefixed_name = f"{self.name}_{original_name}" if self.name else original_name

            expanded = ExpandedTool(
                original_name=original_name,
                prefixed_name=prefixed_name,
                description=tool.get("description", ""),
                input_schema=tool.get("inputSchema", {}),
                mcp_tool=self,
            )

            wrapper = MCPToolWrapper(
                original_name=original_name,
                prefixed_name=prefixed_name,
                description=expanded.description,
                input_schema=expanded.input_schema,
                parent=self,
            )

            self._expanded_tools.append(expanded)
            self._tool_wrappers[prefixed_name] = wrapper

    def get_expanded_tools(self) -> List[ExpandedTool]:
        """获取展开的工具列表"""
        self._ensure_connected()
        return self._expanded_tools

    def get_tool_wrappers(self) -> Dict[str, MCPToolWrapper]:
        """获取工具包装器字典"""
        self._ensure_connected()
        return self._tool_wrappers

    def get_tool_names(self) -> List[str]:
        """获取所有展开后的工具名称"""
        self._ensure_connected()
        return list(self._tool_wrappers.keys())

    async def _async_call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        异步调用 MCP 工具

        Args:
            name: 工具原始名称
            arguments: 工具参数

        Returns:
            工具执行结果
        """
        from protocols.mcp.client import MCPClient

        if self._async_client is None:
            self._async_client = MCPClient(self.server_command)

        async with self._async_client as client:
            return await client.call_tool(name, arguments)

    def call_expanded_tool(
        self,
        prefixed_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        调用展开后的工具

        Args:
            prefixed_name: 带前缀的工具名称
            arguments: 工具参数

        Returns:
            工具执行结果
        """
        self._ensure_connected()

        if prefixed_name not in self._tool_wrappers:
            raise ValueError(f"未知工具: {prefixed_name}")

        wrapper = self._tool_wrappers[prefixed_name]
        return wrapper.run(arguments or {})

    def execute(self, **kwargs) -> str:
        """
        执行 MCP 工具（BaseTool 接口）

        支持两种调用方式：
        1. {"action": "list_tools"} - 列出所有展开的工具
        2. {"action": "call_tool", "tool_name": "...", "arguments": {...}} - 调用工具

        Args:
            **kwargs: 包含 action 和相关参数

        Returns:
            执行结果
        """
        action = kwargs.get("action", "")

        if action == "list_tools":
            return self._list_tools()
        elif action == "call_tool":
            tool_name = kwargs.get("tool_name", "")
            arguments = kwargs.get("arguments", {})
            return self._call_tool(tool_name, arguments)
        elif action == "get_expanded_tools":
            return self._get_expanded_tools()
        else:
            # 默认列出工具
            return self._list_tools()

    def _list_tools(self) -> str:
        """列出所有展开的工具"""
        self._ensure_connected()

        lines = [f"MCP 服务器工具 ({self.name}):"]
        for tool in self._expanded_tools:
            lines.append(f"\n- {tool.prefixed_name}: {tool.description}")
            schema = tool.input_schema.get("properties", {})
            if schema:
                lines.append("  参数:")
                for param_name, param_info in schema.items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    lines.append(f"    - {param_name} ({param_type}): {param_desc}")

        return "\n".join(lines)

    def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """调用指定的工具"""
        self._ensure_connected()

        # 查找工具
        wrapper = self._tool_wrappers.get(tool_name)
        if wrapper is None:
            return f"❌ 未知工具: {tool_name}"

        try:
            return wrapper.run(arguments)
        except Exception as e:
            return f"❌ 工具执行失败: {e}"

    def _get_expanded_tools(self) -> str:
        """获取展开工具的详细信息"""
        self._ensure_connected()

        result = []
        for tool in self._expanded_tools:
            result.append({
                "name": tool.prefixed_name,
                "original_name": tool.original_name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
            })

        return json.dumps(result, ensure_ascii=False, indent=2)

    def to_openai_schema(self) -> List[Dict[str, Any]]:
        """导出所有展开工具的 OpenAI schema"""
        self._ensure_connected()

        schemas = []
        for wrapper in self._tool_wrappers.values():
            schemas.append(wrapper.to_openai_schema())

        return schemas


def create_mcp_tool(
    name: str = "mcp",
    description: str = "MCP 工具服务器",
    server_command: Optional[List[str]] = None,
    timeout: int = 30,
) -> MCPTool:
    """
    工厂函数：创建 MCPTool

    Args:
        name: 工具名称前缀
        description: 工具描述
        server_command: 服务器命令
        timeout: 超时时间

    Returns:
        MCPTool 实例
    """
    return MCPTool(
        name=name,
        description=description,
        server_command=server_command,
        timeout=timeout,
    )
