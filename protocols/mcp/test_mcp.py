"""
MCP 模块测试脚本

测试 MCP 客户端、服务器和工具包装器功能。

运行方式：
    cd agent
    python -m protocols.mcp.test_mcp
"""

import asyncio
import json
from protocols.mcp import MCPClient, MCPTool, MCPServer, MCP_AVAILABLE


def test_mcp_client_memory():
    """测试 MCP 客户端（Memory 传输 - 内置演示服务器）"""
    print("\n" + "=" * 60)
    print("测试 1: MCP 客户端 (Memory 传输)")
    print("=" * 60)

    async def run():
        client = MCPClient()  # 默认使用内置 Mock 服务器
        async with client:
            # 列出工具
            tools = await client.list_tools()
            print(f"\n发现 {len(tools)} 个工具:")
            for tool in tools:
                print(f"  - {tool['name']}: {tool['description']}")

            # 调用工具
            print("\n调用 add 工具 (10 + 20):")
            result = await client.call_tool("add", {"a": 10, "b": 20})
            print(f"  结果: {result}")

            print("\n调用 greet 工具 (name=World):")
            result = await client.call_tool("greet", {"name": "World"})
            print(f"  结果: {result}")

            print("\n调用 get_system_info 工具:")
            result = await client.call_tool("get_system_info", {})
            print(f"  结果: {json.dumps(result, indent=2)}")

            # 列出资源
            resources = client.list_resources()
            print(f"\n发现 {len(resources)} 个资源:")
            for r in resources:
                print(f"  - {r['uri']}: {r['description']}")

    asyncio.run(run())


def test_mcp_tool():
    """测试 MCPTool 包装器"""
    print("\n" + "=" * 60)
    print("测试 2: MCPTool 包装器")
    print("=" * 60)

    # 创建 MCPTool（使用内置服务器）
    mcp_tool = MCPTool(name="calculator")

    # 获取展开的工具
    expanded_tools = mcp_tool.get_expanded_tools()
    print(f"\n展开 {len(expanded_tools)} 个工具:")

    for tool in expanded_tools:
        print(f"\n  工具: {tool.prefixed_name}")
        print(f"  描述: {tool.description}")
        print(f"  参数: {list(tool.input_schema.get('properties', {}).keys())}")

    # 调用展开的工具
    print("\n调用展开的工具:")
    result = mcp_tool.call_expanded_tool("calculator_add", {"a": 100, "b": 200})
    print(f"  calculator_add: {result}")

    result = mcp_tool.call_expanded_tool("calculator_multiply", {"a": 6, "b": 7})
    print(f"  calculator_multiply: {result}")

    # 使用 action 接口
    print("\n使用 action 接口:")
    result = mcp_tool.run({"action": "list_tools"})
    print(result[:500] + "..." if len(result) > 500 else result)


def test_mcp_server():
    """测试 MCPServer"""
    print("\n" + "=" * 60)
    print("测试 3: MCPServer")
    print("=" * 60)

    # 创建服务器
    server = MCPServer(name="TestServer", description="测试服务器")

    # 定义工具
    @server.tool()
    def add(a: float, b: float) -> float:
        """加法计算"""
        return a + b

    @server.tool()
    def multiply(a: float, b: float) -> float:
        """乘法计算"""
        return a * b

    # 定义资源
    @server.resource("config://app")
    def get_config() -> str:
        return '{"version": "1.0.0", "name": "TestServer"}'

    # 定义提示模板
    @server.prompt()
    def math_helper() -> str:
        return "你是一个数学计算助手..."

    # 获取服务器信息
    print(f"\n服务器名称: {server.name}")
    print(f"服务器版本: {server.version}")

    tools = server.get_tools()
    print(f"\n工具列表 ({len(tools)} 个):")
    for t in tools:
        print(f"  - {t['name']}: {t['description']}")

    resources = server.get_resources()
    print(f"\n资源列表 ({len(resources)} 个):")
    for r in resources:
        print(f"  - {r['uri']}")

    prompts = server.get_prompts()
    print(f"\n提示模板 ({len(prompts)} 个):")
    for p in prompts:
        print(f"  - {p['name']}: {p['description']}")


def test_mcp_availability():
    """测试 MCP 可用性"""
    print("\n" + "=" * 60)
    print("测试 4: MCP 库可用性检查")
    print("=" * 60)
    print(f"\nMCP_AVAILABLE: {MCP_AVAILABLE}")
    print(f"MCPServer_AVAILABLE: {MCPServer_AVAILABLE if 'MCPServer_AVAILABLE' in dir() else 'N/A'}")


def main():
    print("=" * 60)
    print("MCP 模块测试")
    print("=" * 60)

    # 测试 MCP 库可用性
    test_mcp_availability()

    # 测试 MCP 客户端
    test_mcp_client_memory()

    # 测试 MCPTool
    test_mcp_tool()

    # 测试 MCPServer
    test_mcp_server()

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
