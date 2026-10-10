"""
MCP 模块演示脚本

展示 MCP 协议模块的各种使用方式。

运行方式：
    cd agent
    python demo_mcp.py
"""

import asyncio
import json
from protocols.mcp import MCPClient, MCPTool, MCPServer


def demo_mcp_client():
    """演示 MCP 客户端功能"""
    print("\n" + "=" * 70)
    print("MCP 客户端演示")
    print("=" * 70)

    async def run():
        print("\n【1】连接到内置 Mock 服务器（Memory 传输）")
        client = MCPClient()

        async with client:
            # 列出所有可用工具
            tools = await client.list_tools()
            print(f"\n发现 {len(tools)} 个工具:")
            for t in tools:
                print(f"  - {t['name']}: {t['description']}")

            # 演示数学计算
            print("\n【2】演示数学计算工具")
            for op, fn in [("add", lambda a, b: a + b),
                           ("subtract", lambda a, b: a - b),
                           ("multiply", lambda a, b: a * b),
                           ("divide", lambda a, b: a / b)]:
                result = await client.call_tool(op, {"a": 10, "b": 3})
                expected = fn(10, 3)
                status = "✅" if abs(result - expected) < 0.001 else "❌"
                print(f"  {op}(10, 3) = {result} {status}")

            # 演示问候
            print("\n【3】演示问候工具")
            result = await client.call_tool("greet", {"name": "警官"})
            print(f"  {result}")

            # 演示系统信息
            print("\n【4】演示系统信息工具")
            result = await client.call_tool("get_system_info", {})
            print(f"  平台: {result['platform']}")
            print(f"  Python: {result['python_version'][:20]}...")

    asyncio.run(run())


def demo_mcp_tool():
    """演示 MCPTool 工具包装器"""
    print("\n" + "=" * 70)
    print("MCPTool 包装器演示")
    print("=" * 70)

    print("\n【1】创建 MCPTool（使用内置服务器）")
    mcp_tool = MCPTool(name="calc")
    print(f"  MCPTool 已创建，name='calc'")

    print("\n【2】获取展开的工具")
    expanded = mcp_tool.get_expanded_tools()
    print(f"\n展开 {len(expanded)} 个工具:")
    for tool in expanded:
        params = list(tool.input_schema.get("properties", {}).keys())
        print(f"  {tool.prefixed_name}: {tool.description}")
        if params:
            print(f"    参数: {params}")

    print("\n【3】调用展开的工具")
    test_cases = [
        ("calc_add", {"a": 123, "b": 456}),
        ("calc_multiply", {"a": 12, "b": 8}),
        ("calc_greet", {"name": "用户"}),
    ]

    for tool_name, args in test_cases:
        try:
            result = mcp_tool.call_expanded_tool(tool_name, args)
            print(f"  {tool_name}({args}): {result}")
        except Exception as e:
            print(f"  {tool_name}: ❌ {e}")

    print("\n【4】使用 action 接口")
    result = mcp_tool.run({"action": "list_tools"})
    lines = result.split("\n")
    print(f"  列出 {len([l for l in lines if l.startswith('- ')])} 个工具")


def demo_mcp_server():
    """演示 MCPServer"""
    print("\n" + "=" * 70)
    print("MCPServer 服务器演示")
    print("=" * 70)

    print("\n【1】创建 MCP 服务器")
    server = MCPServer(
        name="math-assistant-server",
        description="Math and unit conversion utilities exposed via MCP",
        version="1.0.0"
    )

    print("\n【2】定义工具（使用装饰器）")

    @server.tool()
    def calculate_ratio(a: float, b: float) -> float:
        """
        Compute the ratio between two numbers.

        Args:
            a: numerator
            b: denominator

        Returns:
            a / b (returns 0 when b is zero)
        """
        if b == 0:
            return 0
        return a / b

    @server.tool()
    def estimate_capacity(
        workload_type: str,
        volume: int,
        priority: str = "medium"
    ) -> dict:
        """
        Estimate the capacity required for a workload.

        Args:
            workload_type: type of workload (batch / stream / api)
            volume: expected request volume per minute
            priority: priority level (low / medium / high)

        Returns:
            recommended capacity configuration
        """
        baselines = {"batch": 200, "stream": 100, "api": 50}
        baseline = baselines.get(workload_type, 100)

        multipliers = {"low": 0.8, "medium": 1.0, "high": 1.5}
        mult = multipliers.get(priority, 1.0)

        workers = max(1, int(volume / baseline * mult))

        return {
            "workload_type": workload_type,
            "volume": volume,
            "priority": priority,
            "recommended_workers": workers,
            "ratio": f"1:{int(volume / max(workers, 1))}",
            "mode": "dedicated" if workers > 5 else "shared",
        }

    print("  ✅ 已定义 2 个工具:")
    print("    - calculate_ratio(a, b)")
    print("    - estimate_capacity(workload_type, volume, priority)")

    print("\n【3】获取服务器信息")
    print(f"  服务器名称: {server.name}")
    print(f"  服务器描述: {server.description}")
    print(f"  服务器版本: {server.version}")

    print("\n【4】获取工具列表")
    tools = server.get_tools()
    for t in tools:
        print(f"\n  工具: {t['name']}")
        print(f"  描述: {t['description']}")
        props = t["inputSchema"].get("properties", {})
        print(f"  参数: {list(props.keys())}")


def demo_use_cases():
    """演示实际使用场景"""
    print("\n" + "=" * 70)
    print("实际使用场景演示")
    print("=" * 70)

    print("\n【场景1】在 Agent 中使用 MCP 工具")
    print("""
    from agents import SimpleAgent
    from llm_client import build_llm
    from protocols.mcp import MCPTool

    # 创建 Agent
    llm = build_llm()
    agent = SimpleAgent(name="math-assistant", llm=llm)

    # 添加 MCP 工具
    mcp_tool = MCPTool(name="math")
    agent.add_tool(mcp_tool)

    # Agent 可以自动使用展开后的工具
    response = agent.run("If revenue grew 18% quarter-over-quarter, what is the new total?")
    """)

    print("\n【场景2】连接社区 MCP 服务器")
    print("""
    # 连接 GitHub MCP 服务器
    github_tool = MCPTool(
        name="github",
        description="GitHub 访问工具",
        server_command=["npx", "-y", "@modelcontextprotocol/server-github"]
    )

    # 连接文件系统 MCP 服务器
    fs_tool = MCPTool(
        name="filesystem",
        description="本地文件系统访问",
        server_command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "."]
    )

    # 连接后可以自动使用这些工具
    expanded = github_tool.get_expanded_tools()
    for tool in expanded:
        print(f"- {tool.prefixed_name}")
    """)

    print("\n【场景3】创建自定义 MCP 服务器")
    print("""
    from protocols.mcp import MCPServer

    server = MCPServer(name="ops_server")

    @server.tool()
    def queue_depth(queue: str) -> dict:
        '''Returns the current depth of a named queue.'''
        # 业务逻辑
        return {"queue": queue, "depth": 50}

    @server.tool()
    def validate_config(config: dict) -> bool:
        '''验证配置合法性'''
        # 业务逻辑
        return True

    # 运行服务器（通过 Stdio 与客户端通信）
    server.run()
    """)


def main():
    print("=" * 70)
    print("🤖 MCP 协议模块演示")
    print("=" * 70)
    print("""
MCP (Model Context Protocol) 是智能体与外部工具/资源通信的标准协议。
本模块参考 hello-agents 第十章设计，提供完整的 MCP 协议实现。
    """)

    # 演示 MCP 客户端
    demo_mcp_client()

    # 演示 MCPTool
    demo_mcp_tool()

    # 演示 MCPServer
    demo_mcp_server()

    # 演示使用场景
    demo_use_cases()

    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)
    print("""
更多信息请参考:
  - protocols/mcp/README.md
  - protocols/mcp/test_mcp.py
  - hello-agents/docs/chapter10/第十章 智能体通信协议.md
    """)


if __name__ == "__main__":
    main()
