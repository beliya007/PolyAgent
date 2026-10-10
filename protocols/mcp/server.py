"""
MCP 服务器实现

参考 hello-agents 第十章设计，提供 MCP 服务器功能：
- 基于 FastMCP 封装
- 支持工具、资源、提示模板定义
- 自动处理 Stdio 传输

使用示例：
    from protocols.mcp import MCPServer

    server = MCPServer(name="MyServer")

    @server.tool()
    def add(a: float, b: float) -> float:
        '''加法计算器'''
        return a + b

    @server.resource("config://app")
    def get_config() -> str:
        return '{"version": "1.0.0"}'

    @server.prompt()
    def math_helper() -> str:
        return "你是一个数学计算助手..."

    # 运行服务器
    server.run()
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from functools import wraps

# MCP 服务器可用性标志
MCPServer_AVAILABLE = False
FastMCP = None

# 尝试导入 FastMCP
try:
    from fastmcp import FastMCP as FastMCPClass
    MCPServer_AVAILABLE = True
except ImportError:
    pass


@dataclass
class ToolDefinition:
    """工具定义"""
    name: str
    description: str
    func: Callable
    input_schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceDefinition:
    """资源定义"""
    uri: str
    name: str
    func: Callable
    mime_type: str = "text/plain"


@dataclass
class PromptDefinition:
    """提示模板定义"""
    name: str
    description: str
    func: Callable


class SimpleMCPServer:
    """
    简化的 MCP 服务器实现

    当 FastMCP 不可用时提供基础功能。
    支持通过装饰器定义工具、资源和提示模板。
    """

    def __init__(
        self,
        name: str = "SimpleMCPServer",
        description: str = "A simple MCP server",
        version: str = "1.0.0",
    ):
        self.name = name
        self.description = description
        self.version = version
        self._tools: Dict[str, ToolDefinition] = {}
        self._resources: Dict[str, ResourceDefinition] = {}
        self._prompts: Dict[str, PromptDefinition] = {}

    def tool(
        self,
        name: Optional[str] = None,
        description: str = "",
    ) -> Callable:
        """
        工具装饰器

        使用示例：
            @server.tool()
            def add(a: float, b: float) -> float:
                '''加法计算'''
                return a + b
        """
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_desc = description or func.__doc__ or ""

            # 从函数签名提取参数schema
            import inspect
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

            self._tools[tool_name] = ToolDefinition(
                name=tool_name,
                description=tool_desc,
                func=func,
                input_schema={
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            )

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

        return decorator

    def resource(
        self,
        uri: str,
        name: Optional[str] = None,
        mime_type: str = "text/plain",
    ) -> Callable:
        """
        资源装饰器

        使用示例：
            @server.resource("config://app")
            def get_config() -> str:
                return '{"version": "1.0.0"}'
        """
        def decorator(func: Callable) -> Callable:
            resource_name = name or func.__name__

            self._resources[uri] = ResourceDefinition(
                uri=uri,
                name=resource_name,
                func=func,
                mime_type=mime_type,
            )

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

        return decorator

    def prompt(
        self,
        name: Optional[str] = None,
        description: str = "",
    ) -> Callable:
        """
        提示模板装饰器

        使用示例：
            @server.prompt()
            def math_helper() -> str:
                return "你是一个数学计算助手..."
        """
        def decorator(func: Callable) -> Callable:
            prompt_name = name or func.__name__
            prompt_desc = description or func.__doc__ or ""

            self._prompts[prompt_name] = PromptDefinition(
                name=prompt_name,
                description=prompt_desc,
                func=func,
            )

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

        return decorator

    def run(self, host: str = "localhost", port: int = 8000):
        """
        运行服务器（仅输出信息，实际需要通过 MCP 协议通信）

        使用 Stdio 传输时，服务器会通过 stdin/stdout 与客户端通信。
        """
        print(f"启动 MCP 服务器: {self.name} v{self.version}")
        print(f"工具数量: {len(self._tools)}")
        print(f"资源数量: {len(self._resources)}")
        print(f"提示模板数量: {len(self._prompts)}")
        print("\n使用 Stdio 传输，请通过 MCP 客户端连接")

    def get_tools(self) -> List[Dict[str, Any]]:
        """获取所有工具定义"""
        return [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
            }
            for t in self._tools.values()
        ]

    def get_resources(self) -> List[Dict[str, Any]]:
        """获取所有资源定义"""
        return [
            {
                "uri": r.uri,
                "name": r.name,
                "mimeType": r.mime_type,
            }
            for r in self._resources.values()
        ]

    def get_prompts(self) -> List[Dict[str, Any]]:
        """获取所有提示模板定义"""
        return [
            {
                "name": p.name,
                "description": p.description,
            }
            for p in self._prompts.values()
        ]

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """调用工具"""
        if name not in self._tools:
            raise ValueError(f"未知工具: {name}")
        tool = self._tools[name]
        return tool.func(**arguments)

    async def read_resource(self, uri: str) -> Any:
        """读取资源"""
        if uri not in self._resources:
            raise ValueError(f"未知资源: {uri}")
        resource = self._resources[uri]
        result = resource.func()
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        return str(result)

    async def get_prompt(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """获取提示模板"""
        if name not in self._prompts:
            raise ValueError(f"未知提示模板: {name}")
        prompt = self._prompts[name]
        return prompt.func(**(arguments or {}))


class MCPServer:
    """
    MCP 服务器包装器

    优先使用 FastMCP，失败时回退到 SimpleMCPServer。

    使用示例：
        server = MCPServer(name="MyServer")

        @server.tool()
        def add(a: float, b: float) -> float:
            return a + b

        server.run()
    """

    def __init__(
        self,
        name: str = "MCPServer",
        description: str = "MCP Server",
        version: str = "1.0.0",
    ):
        self.name = name
        self.description = description
        self.version = version

        if MCPServer_AVAILABLE:
            # 使用 FastMCP
            self._server = FastMCPClass(name, description=description)
        else:
            # 回退到简化的实现
            self._server = SimpleMCPServer(
                name=name,
                description=description,
                version=version,
            )

    def tool(
        self,
        name: Optional[str] = None,
        description: str = "",
    ) -> Callable:
        """工具装饰器"""
        return self._server.tool(name=name, description=description)

    def resource(
        self,
        uri: str,
        name: Optional[str] = None,
        mime_type: str = "text/plain",
    ) -> Callable:
        """资源装饰器"""
        return self._server.resource(uri=uri, name=name, mime_type=mime_type)

    def prompt(
        self,
        name: Optional[str] = None,
        description: str = "",
    ) -> Callable:
        """提示模板装饰器"""
        return self._server.prompt(name=name, description=description)

    def run(self, host: str = "localhost", port: int = 8000):
        """运行服务器"""
        if isinstance(self._server, SimpleMCPServer):
            self._server.run(host=host, port=port)
        else:
            # FastMCP 服务器会通过 run() 自动处理 Stdio 传输
            self._server.run()

    def get_tools(self) -> List[Dict[str, Any]]:
        """获取所有工具定义"""
        return self._server.get_tools()

    def get_resources(self) -> List[Dict[str, Any]]:
        """获取所有资源定义"""
        return self._server.get_resources()

    def get_prompts(self) -> List[Dict[str, Any]]:
        """获取所有提示模板定义"""
        return self._server.get_prompts()


def create_mcp_server(
    name: str = "MCPServer",
    description: str = "MCP Server",
    version: str = "1.0.0",
) -> MCPServer:
    """
    工厂函数：创建 MCP 服务器

    Args:
        name: 服务器名称
        description: 服务器描述
        version: 服务器版本

    Returns:
        MCPServer 实例
    """
    return MCPServer(name=name, description=description, version=version)
