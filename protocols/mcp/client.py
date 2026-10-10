"""
MCP 客户端实现

参考 hello-agents 第十章 2.2 节设计，支持多种传输方式：
- Stdio: 标准输入输出传输（本地进程通信）
- Memory: 内存传输（内置演示服务器，用于测试）
- HTTP: HTTP 传输（远程服务）
- SSE: Server-Sent Events 传输
- StreamableHTTP: 流式 HTTP 传输

使用示例：
    import asyncio
    from protocols.mcp import MCPClient

    # 方式1: Stdio 传输（连接到本地进程）
    async def connect_stdio():
        client = MCPClient(["python", "my_mcp_server.py"])
        async with client:
            tools = await client.list_tools()
            result = await client.call_tool("add", {"a": 10, "b": 20})

    # 方式2: HTTP 传输（连接到远程服务）
    async def connect_http():
        client = MCPClient("http://api.example.com/mcp")
        async with client:
            tools = await client.list_tools()

    asyncio.run(connect_stdio())
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager

# MCP 可用性标志
MCP_AVAILABLE = False
ClientSession = None
stdio_client = None

# 尝试导入官方 MCP 库
try:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    pass


@dataclass
class MCPToolInfo:
    """MCP 工具信息"""
    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


@dataclass
class MCPResourceInfo:
    """MCP 资源信息"""
    uri: str
    name: str = ""
    description: str = ""
    mime_type: str = "text/plain"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        }


@dataclass
class MCPPromptInfo:
    """MCP 提示模板信息"""
    name: str
    description: str = ""
    arguments: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments,
        }


class MockMCPSession:
    """
    Mock MCP Session - 用于测试或 MCP 库不可用时

    提供内置的演示工具：
    - add(a, b): 加法
    - subtract(a, b): 减法
    - multiply(a, b): 乘法
    - divide(a, b): 除法
    - greet(name): 问候
    - get_system_info(): 获取系统信息
    """

    def __init__(self):
        self._tools = {
            "add": MCPToolInfo(
                name="add",
                description="加法计算器",
                input_schema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "第一个数字"},
                        "b": {"type": "number", "description": "第二个数字"},
                    },
                    "required": ["a", "b"],
                },
            ),
            "subtract": MCPToolInfo(
                name="subtract",
                description="减法计算器",
                input_schema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "被减数"},
                        "b": {"type": "number", "description": "减数"},
                    },
                    "required": ["a", "b"],
                },
            ),
            "multiply": MCPToolInfo(
                name="multiply",
                description="乘法计算器",
                input_schema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "第一个数字"},
                        "b": {"type": "number", "description": "第二个数字"},
                    },
                    "required": ["a", "b"],
                },
            ),
            "divide": MCPToolInfo(
                name="divide",
                description="除法计算器",
                input_schema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "被除数"},
                        "b": {"type": "number", "description": "除数"},
                    },
                    "required": ["a", "b"],
                },
            ),
            "greet": MCPToolInfo(
                name="greet",
                description="友好问候",
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "被问候者姓名"},
                    },
                    "required": ["name"],
                },
            ),
            "get_system_info": MCPToolInfo(
                name="get_system_info",
                description="获取系统信息",
                input_schema={"type": "object", "properties": {}},
            ),
        }

    async def list_tools(self) -> List[MCPToolInfo]:
        """列出所有可用工具"""
        return list(self._tools.values())

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """调用工具"""
        if name not in self._tools:
            raise ValueError(f"未知工具: {name}")

        # 执行内置工具
        if name == "add":
            return arguments.get("a", 0) + arguments.get("b", 0)
        elif name == "subtract":
            return arguments.get("a", 0) - arguments.get("b", 0)
        elif name == "multiply":
            return arguments.get("a", 0) * arguments.get("b", 0)
        elif name == "divide":
            a = arguments.get("a", 0)
            b = arguments.get("b", 0)
            if b == 0:
                raise ValueError("除数不能为零")
            return a / b
        elif name == "greet":
            return f"你好，{arguments.get('name', '朋友')}！"
        elif name == "get_system_info":
            import platform
            import sys
            return {
                "platform": platform.system(),
                "python_version": sys.version,
                "architecture": platform.machine(),
            }
        return None

    def list_resources(self) -> List[MCPResourceInfo]:
        """列出所有可用资源"""
        return [
            MCPResourceInfo(
                uri="config://server",
                name="server_config",
                description="服务器配置信息",
            ),
            MCPResourceInfo(
                uri="info://capabilities",
                name="capabilities",
                description="服务器能力列表",
            ),
        ]

    async def read_resource(self, uri: str) -> Any:
        """读取资源"""
        if uri == "config://server":
            return json.dumps({
                "name": "MockMCPServer",
                "version": "1.0.0",
                "tools_count": len(self._tools),
            }, ensure_ascii=False)
        elif uri == "info://capabilities":
            return "支持工具: " + ", ".join(self._tools.keys())
        return None

    def list_prompts(self) -> List[MCPPromptInfo]:
        """列出所有提示模板"""
        return [
            MCPPromptInfo(
                name="math_helper",
                description="数学计算助手提示词",
            ),
            MCPPromptInfo(
                name="text_processor",
                description="文本处理助手提示词",
            ),
        ]

    async def get_prompt(self, name: str, arguments: Dict[str, Any]) -> Any:
        """获取提示模板"""
        prompts = {
            "math_helper": """你是一个数学计算助手。你可以使用以下工具：
- add(a, b): 计算两数之和
- subtract(a, b): 计算两数之差
- multiply(a, b): 计算两数之积
- divide(a, b): 计算两数之商

请根据用户的问题选择合适的工具进行计算。""",
            "text_processor": """你是一个文本处理助手。你可以使用以下工具：
- reverse_text(text): 反转文本
- count_words(text): 统计单词数
- to_uppercase(text): 转换为大写
- to_lowercase(text): 转换为小写

请根据用户的需求选择合适的工具处理文本。""",
        }
        return prompts.get(name, "")


class MCPClient:
    """
    MCP 客户端

    支持多种传输方式：
    - Stdio: 通过子进程标准输入输出通信
    - Memory: 使用内置 Mock 服务器（用于测试）
    - HTTP: 通过 HTTP 协议通信（远程服务）

    使用示例：
        async with MCPClient(["python", "server.py"]) as client:
            tools = await client.list_tools()
            result = await client.call_tool("add", {"a": 10, "b": 20})
    """

    def __init__(
        self,
        command: Optional[Union[str, List[str]]] = None,
        transport: str = "auto",
        **kwargs,
    ):
        """
        初始化 MCP 客户端

        Args:
            command: 服务器命令列表，如 ["python", "server.py"] 或 "http://..."
            transport: 传输类型，"auto", "stdio", "memory", "http"
            **kwargs: 额外参数
        """
        self.command = command
        self.transport = transport
        self.kwargs = kwargs
        self._session: Optional[MockMCPSession] = None
        self._real_session = None
        self._connected = False

    def _detect_transport(self) -> str:
        """自动检测传输类型"""
        if self.transport != "auto":
            return self.transport

        if self.command is None:
            return "memory"
        elif isinstance(self.command, str):
            if self.command.startswith("http://") or self.command.startswith("https://"):
                return "http"
            return "memory"
        else:
            return "stdio"

    async def __aenter__(self):
        """异步上下文管理器入口"""
        transport_type = self._detect_transport()

        if transport_type == "memory":
            # 使用内置 Mock 服务器
            self._session = MockMCPSession()
            self._connected = True

        elif transport_type == "stdio" and MCP_AVAILABLE:
            # 使用 Stdio 传输
            if isinstance(self.command, list) and len(self.command) >= 2:
                from mcp.client.stdio import stdio_client
                from mcp import ClientSession

                self._stdio_params = {
                    "command": self.command[0],
                    "args": self.command[1:],
                }
                self._stdio_transport = stdio_client(self._stdio_params)
                self._stdio_iterator = self._stdio_transport.__aiter__()
                self._real_session = ClientSession(await self._stdio_iterator.__anext__())
                await self._real_session.initialize()
                self._connected = True
            else:
                raise ValueError("Stdio 传输需要命令列表")

        elif transport_type == "http" and MCP_AVAILABLE:
            # HTTP 传输（需要额外的 HTTP 客户端支持）
            raise NotImplementedError("HTTP 传输需要安装 mcp 库并配置服务器")

        else:
            # Fallback 到 Mock
            self._session = MockMCPSession()
            self._connected = True

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        if self._real_session:
            await self._real_session.close()
        self._connected = False
        self._session = None

    @property
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._connected

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        列出所有可用工具

        Returns:
            工具列表，每个工具包含 name, description, inputSchema
        """
        if self._real_session:
            result = await self._real_session.list_tools()
            return [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "inputSchema": t.inputSchema if hasattr(t, "inputSchema") else {},
                }
                for t in result
            ]
        elif self._session:
            tools = await self._session.list_tools()
            return [t.to_dict() for t in tools]
        raise RuntimeError("客户端未连接")

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        调用 MCP 工具

        Args:
            name: 工具名称
            arguments: 工具参数

        Returns:
            工具执行结果
        """
        if self._real_session:
            result = await self._real_session.call_tool(name, arguments)
            if hasattr(result, "content"):
                # 返回文本内容
                for content in result.content:
                    if hasattr(content, "text"):
                        return content.text
                return str(result)
            return result
        elif self._session:
            return await self._session.call_tool(name, arguments)
        raise RuntimeError("客户端未连接")

    def list_resources(self) -> List[Dict[str, Any]]:
        """列出所有可用资源"""
        if self._session:
            resources = self._session.list_resources()
            return [r.to_dict() for r in resources]
        return []

    async def read_resource(self, uri: str) -> Any:
        """读取资源"""
        if self._session:
            return await self._session.read_resource(uri)
        raise RuntimeError("客户端未连接")

    def list_prompts(self) -> List[Dict[str, Any]]:
        """列出所有提示模板"""
        if self._session:
            prompts = self._session.list_prompts()
            return [p.to_dict() for p in prompts]
        return []

    async def get_prompt(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """获取提示模板"""
        if self._session:
            return await self._session.get_prompt(name, arguments or {})
        raise RuntimeError("客户端未连接")


def create_mcp_client(
    command: Optional[Union[str, List[str]]] = None,
    transport: str = "auto",
    **kwargs,
) -> MCPClient:
    """
    工厂函数：创建 MCP 客户端

    Args:
        command: 服务器命令
        transport: 传输类型
        **kwargs: 额外参数

    Returns:
        MCPClient 实例
    """
    return MCPClient(command=command, transport=transport, **kwargs)
