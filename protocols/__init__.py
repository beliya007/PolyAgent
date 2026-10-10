"""
Protocols 模块 - 智能体通信协议实现

基于 polyagent 第十章设计，提供三种协议：
- MCP (Model Context Protocol): 智能体与工具的标准化通信
- A2A (Agent-to-Agent Protocol): 智能体间的点对点协作
- ANP (Agent Network Protocol): 大规模智能体网络

模块结构：
    protocols/
    ├── mcp/           # MCP 协议实现
    │   ├── client.py   # MCP 客户端
    │   ├── server.py   # MCP 服务器
    │   ├── tool.py    # MCPTool 工具包装器
    │   └── utils.py   # 工具函数
    ├── a2a/           # A2A 协议实现
    │   ├── server.py  # A2A 服务端
    │   ├── client.py  # A2A 客户端
    │   └── tool.py    # A2ATool 工具包装器
    └── anp/           # ANP 协议实现
        ├── implementation.py  # 服务发现与路由
        └── tool.py    # ANPTool 工具包装器

三种协议对比：
- MCP: 智能体 ↔ 工具（标准化接口）
- A2A: 智能体 ↔ 智能体（点对点协作）
- ANP: 大规模智能体网络（去中心化服务发现）
"""

# MCP 协议
from protocols.mcp import MCPClient, MCPTool, MCPServer, MCP_AVAILABLE

# A2A 协议
from protocols.a2a import (
    A2AServer,
    A2AClient,
    A2ATool,
    SkillDefinition,
    TaskStatus,
    Task,
)

# ANP 协议
from protocols.anp import (
    ANPDiscovery,
    ANPRouter,
    ANPAgentInfo,
    register_service,
    discover_services_by_type,
)

# ANP Tool（从 tool.py 导入）
try:
    from protocols.anp.tool import ANPTool
except ImportError:
    ANPTool = None

__all__ = [
    # MCP 协议
    "MCPClient",
    "MCPServer",
    "MCPTool",
    "MCP_AVAILABLE",
    # A2A 协议
    "A2AServer",
    "A2AClient",
    "A2ATool",
    "SkillDefinition",
    "TaskStatus",
    "Task",
    # ANP 协议
    "ANPDiscovery",
    "ANPRouter",
    "ANPAgentInfo",
    "ANPTool",
    "register_service",
    "discover_services_by_type",
]