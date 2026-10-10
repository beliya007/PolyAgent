# MCP 协议模块

基于 hello-agents 第十章设计实现的 Model Context Protocol (MCP) 协议模块。

## 目录结构

```
protocols/
├── __init__.py              # 模块入口
└── mcp/                     # MCP 协议实现
    ├── __init__.py          # MCP 子模块入口
    ├── client.py            # MCP 客户端（支持多种传输方式）
    ├── server.py            # MCP 服务器
    ├── tool.py              # MCPTool 工具包装器
    ├── utils.py             # 工具函数
    └── test_mcp.py          # 测试脚本
```

## 核心组件

### 1. MCPClient - MCP 客户端

支持多种传输方式：
- **Memory**: 内存传输（内置演示服务器，用于测试）
- **Stdio**: 标准输入输出传输（本地进程通信）
- **HTTP**: HTTP 传输（远程服务）
- **SSE**: Server-Sent Events 传输
- **StreamableHTTP**: 流式 HTTP 传输

```python
import asyncio
from protocols.mcp import MCPClient

async def example():
    # Memory 传输（使用内置 Mock 服务器）
    async with MCPClient() as client:
        tools = await client.list_tools()
        result = await client.call_tool("add", {"a": 10, "b": 20})
        print(result)  # 30

    # Stdio 传输（连接到本地 MCP 服务器）
    async with MCPClient(["python", "my_mcp_server.py"]) as client:
        tools = await client.list_tools()
        result = await client.call_tool("read_file", {"path": "README.md"})

asyncio.run(example())
```

### 2. MCPServer - MCP 服务器

创建自定义 MCP 服务器，提供工具、资源和提示模板：

```python
from protocols.mcp import MCPServer

server = MCPServer(name="MyServer", description="我的 MCP 服务器")

@server.tool()
def add(a: float, b: float) -> float:
    """加法计算"""
    return a + b

@server.resource("config://app")
def get_config() -> str:
    return '{"version": "1.0.0"}'

@server.prompt()
def math_helper() -> str:
    return "你是一个数学计算助手..."

server.run()
```

### 3. MCPTool - MCP 工具包装器

将 MCP 服务器提供的工具自动展开为独立的 BaseTool，与 Agent 工具系统无缝集成：

```python
from protocols.mcp import MCPTool

# 方式1: 使用内置演示服务器
mcp_tool = MCPTool()  # name 默认为 "mcp"

# 方式2: 连接到自定义服务器
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

# 获取展开的工具
expanded_tools = mcp_tool.get_expanded_tools()
for tool in expanded_tools:
    print(f"- {tool.prefixed_name}: {tool.description}")

# 调用展开的工具
result = mcp_tool.call_expanded_tool("calculator_add", {"a": 10, "b": 20})
print(result)  # 30.0
```

## 内置工具

当使用默认的 Memory 传输时，提供以下内置工具：

| 工具名 | 描述 | 参数 |
|--------|------|------|
| add | 加法计算器 | a, b (number) |
| subtract | 减法计算器 | a, b (number) |
| multiply | 乘法计算器 | a, b (number) |
| divide | 除法计算器 | a, b (number) |
| greet | 友好问候 | name (string) |
| get_system_info | 获取系统信息 | - |

## 与 Agent 集成

MCPTool 可以直接添加到 Agent 的工具注册表中：

```python
from agents import SimpleAgent
from llm_client import build_llm
from protocols.mcp import MCPTool

# 创建 Agent
llm = build_llm()
agent = SimpleAgent(name="助手", llm=llm)

# 添加 MCP 工具
mcp_tool = MCPTool(name="calculator")
agent.add_tool(mcp_tool)

# Agent 现在可以使用展开后的工具
response = agent.run("计算 25 乘以 16")
```

## 工具自动展开机制

MCPTool 的核心特性是**自动展开**：

1. 连接到 MCP 服务器后，自动发现所有可用工具
2. 为每个工具创建带前缀的包装器（避免名称冲突）
3. 将展开后的工具注册到 Agent 的工具注册表

例如，服务器提供 `add` 工具，展开后变为 `calculator_add`（假设 `name="calculator"`）。

## 工具函数 (utils.py)

提供格式转换和工具函数：

```python
from protocols.mcp.utils import (
    mcp_tool_to_openai_schema,    # MCP schema 转 OpenAI schema
    openai_schema_to_mcp_tool,    # OpenAI schema 转 MCP schema
    expand_mcp_tools,             # 展开工具列表（添加前缀）
    parse_tool_arguments,         # 解析工具参数
    validate_tool_arguments,      # 验证工具参数
    build_llm_tools_context,      # 构建供 LLM 使用的工具描述
)
```

## 运行测试

```bash
cd agent
python -m protocols.mcp.test_mcp
```

## 依赖

- `mcp`: 官方 MCP SDK（可选，未安装时使用内置 Mock 实现）
- `fastmcp`: FastMCP 服务器框架（可选）

未安装这些依赖时，模块会自动使用内置的 Mock 实现，确保功能可用。

## 参考

- [hello-agents 第十章](https://github.com/datawhalechina/Hello-Agents)
- [MCP 官方文档](https://modelcontextprotocol.io/)
- [Awesome MCP Servers](https://github.com/punkpeye/awesome-mcp-servers)
