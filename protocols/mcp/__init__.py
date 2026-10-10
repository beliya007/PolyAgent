"""
MCP (Model Context Protocol) 子模块

参考 polyagent 第十章设计，实现智能体与工具的标准化通信协议。

MCP 协议设计理念：
- 标准化接口：不同服务提供统一的访问方式
- 互操作性：不同开发者的工具可以无缝集成
- 动态发现：智能体在运行时发现新的服务和能力
- 可扩展性：系统能够轻松添加新的功能模块

MCP 三层架构：
1. Host（宿主层）：智能体运行时环境
2. Client（客户端层）：与 MCP Server 通信
3. Server（服务器层）：提供工具、资源、提示模板

主要组件：
- MCPClient: 连接到 MCP 服务器，支持多种传输方式
- MCPServer: 创建 MCP 服务器（基于 FastMCP）
- MCPTool: 工具包装器，自动展开 MCP 服务器的工具到 Agent
"""

# MCP 协议是否可用（需要安装 mcp 包）
try:
    from mcp import ClientSession, Server
    from mcp.client.stdio import stdio_client
    from mcp.server import Server as MCPServerClass
    from mcp.types import Tool, Resource, Prompt
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# 导出核心组件
from protocols.mcp.client import MCPClient
from protocols.mcp.server import MCPServer
from protocols.mcp.tool import MCPTool

__all__ = [
    "MCPClient",
    "MCPServer",
    "MCPTool",
    "MCP_AVAILABLE",
]
