"""
MCP 工具函数

参考 hello-agents 第十章设计，提供 MCP 相关的工具函数：
- 工具格式转换（OpenAI schema / MCP schema）
- 上下文创建与解析
- 类型转换
- 参数验证
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union


def mcp_tool_to_openai_schema(tool: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 MCP 工具 schema 转换为 OpenAI function calling schema

    MCP 使用 inputSchema，OpenAI 使用 parameters。
    两者结构类似但字段名不同。

    Args:
        tool: MCP 工具定义，包含 name, description, inputSchema

    Returns:
        OpenAI function calling schema
    """
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("inputSchema", {
                "type": "object",
                "properties": {},
            }),
        },
    }


def openai_schema_to_mcp_tool(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 OpenAI function calling schema 转换为 MCP 工具 schema

    Args:
        schema: OpenAI function calling schema

    Returns:
        MCP 工具定义
    """
    function = schema.get("function", {})
    return {
        "name": function.get("name", ""),
        "description": function.get("description", ""),
        "inputSchema": function.get("parameters", {
            "type": "object",
            "properties": {},
        }),
    }


def expand_mcp_tools(
    tools: List[Dict[str, Any]],
    prefix: str = "",
) -> List[Dict[str, Any]]:
    """
    展开 MCP 工具列表，添加前缀以避免名称冲突

    Args:
        tools: MCP 工具列表
        prefix: 工具名称前缀

    Returns:
        展开后的工具列表
    """
    expanded = []
    for tool in tools:
        original_name = tool.get("name", "")
        new_name = f"{prefix}{original_name}" if prefix else original_name
        expanded.append({
            **tool,
            "name": new_name,
            "original_name": original_name,
            "prefix": prefix,
        })
    return expanded


def parse_tool_arguments(
    schema: Dict[str, Any],
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """
    解析并验证工具参数

    Args:
        schema: 参数 schema
        arguments: 原始参数

    Returns:
        解析后的参数
    """
    if not schema:
        return arguments

    properties = schema.get("properties", {})
    parsed = {}

    for key, value in arguments.items():
        if key in properties:
            prop_schema = properties[key]
            prop_type = prop_schema.get("type", "string")

            # 类型转换
            if prop_type == "number":
                if isinstance(value, str):
                    try:
                        value = float(value)
                    except ValueError:
                        value = 0.0
                elif isinstance(value, int):
                    value = float(value)
            elif prop_type == "integer":
                if isinstance(value, str):
                    try:
                        value = int(value)
                    except ValueError:
                        value = 0
                elif isinstance(value, float):
                    value = int(value)
            elif prop_type == "boolean":
                if isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes", "on")
                else:
                    value = bool(value)
            elif prop_type == "array":
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        value = []
            elif prop_type == "object":
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        value = {}

            parsed[key] = value
        else:
            parsed[key] = value

    return parsed


def validate_tool_arguments(
    schema: Dict[str, Any],
    arguments: Dict[str, Any],
) -> tuple[bool, Optional[str]]:
    """
    验证工具参数是否符合 schema

    Args:
        schema: 参数 schema
        arguments: 参数

    Returns:
        (是否有效, 错误信息)
    """
    if not schema:
        return True, None

    required = schema.get("required", [])
    properties = schema.get("properties", {})

    # 检查必填参数
    for param in required:
        if param not in arguments:
            return False, f"缺少必填参数: {param}"

    # 检查类型
    for key, value in arguments.items():
        if key in properties:
            prop_schema = properties[key]
            expected_type = prop_schema.get("type", "string")

            # 简单类型检查
            if expected_type == "number" and not isinstance(value, (int, float)):
                return False, f"参数 {key} 应为数字"
            elif expected_type == "integer" and not isinstance(value, int):
                return False, f"参数 {key} 应为整数"
            elif expected_type == "boolean" and not isinstance(value, bool):
                return False, f"参数 {key} 应为布尔值"
            elif expected_type == "array" and not isinstance(value, list):
                return False, f"参数 {key} 应为数组"
            elif expected_type == "object" and not isinstance(value, dict):
                return False, f"参数 {key} 应为对象"

    return True, None


def format_tool_result(result: Any, to_string: bool = True) -> Union[str, Any]:
    """
    格式化工具结果

    Args:
        result: 原始结果
        to_string: 是否转换为字符串

    Returns:
        格式化后的结果
    """
    if result is None:
        return "" if to_string else None

    if isinstance(result, str):
        return result

    if isinstance(result, (dict, list)):
        if to_string:
            return json.dumps(result, ensure_ascii=False, indent=2)
        return result

    return str(result)


def create_context(
    tools: List[Dict[str, Any]],
    resources: Optional[List[Dict[str, Any]]] = None,
    prompts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    创建 MCP 上下文

    Args:
        tools: 工具列表
        resources: 资源列表（可选）
        prompts: 提示模板列表（可选）

    Returns:
        MCP 上下文
    """
    return {
        "tools": tools,
        "resources": resources or [],
        "prompts": prompts or [],
    }


def parse_context(context: Dict[str, Any]) -> tuple:
    """
    解析 MCP 上下文

    Args:
        context: MCP 上下文

    Returns:
        (tools, resources, prompts)
    """
    return (
        context.get("tools", []),
        context.get("resources", []),
        context.get("prompts", []),
    )


def build_llm_tools_context(tools: List[Dict[str, Any]]) -> str:
    """
    构建供 LLM 使用的工具描述文本

    Args:
        tools: 工具列表

    Returns:
        格式化的工具描述文本
    """
    if not tools:
        return "无可用工具"

    lines = ["你可以使用以下工具："]
    for tool in tools:
        name = tool.get("name", "")
        desc = tool.get("description", "")
        schema = tool.get("inputSchema", {})
        props = schema.get("properties", {})

        lines.append(f"\n- {name}: {desc}")

        if props:
            lines.append("  参数:")
            for param_name, param_info in props.items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                lines.append(f"    - {param_name} ({param_type}): {param_desc}")

    return "\n".join(lines)


def merge_tool_schemas(
    tool1: Dict[str, Any],
    tool2: Dict[str, Any],
) -> Dict[str, Any]:
    """
    合并两个工具 schema（用于冲突解决）

    Args:
        tool1: 第一个工具
        tool2: 第二个工具

    Returns:
        合并后的工具
    """
    name = tool1.get("name", tool2.get("name", ""))
    description = tool1.get("description", "") or tool2.get("description", "")
    schema1 = tool1.get("inputSchema", {})
    schema2 = tool2.get("inputSchema", {})

    # 合并 properties
    properties = {**schema2.get("properties", {}), **schema1.get("properties", {})}

    # 合并 required（取并集）
    required = list(set(schema1.get("required", [])) | set(schema2.get("required", [])))

    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


class MCPTransportFactory:
    """MCP 传输方式工厂"""

    @staticmethod
    def create_stdio(command: List[str]) -> Dict[str, Any]:
        """
        创建 Stdio 传输配置

        Args:
            command: 命令列表

        Returns:
            传输配置
        """
        return {
            "type": "stdio",
            "command": command[0],
            "args": command[1:],
        }

    @staticmethod
    def create_http(url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        创建 HTTP 传输配置

        Args:
            url: 服务器 URL
            headers: 请求头

        Returns:
            传输配置
        """
        return {
            "type": "http",
            "url": url,
            "headers": headers or {},
        }

    @staticmethod
    def create_sse(url: str) -> Dict[str, Any]:
        """
        创建 SSE 传输配置

        Args:
            url: SSE 服务器 URL

        Returns:
            传输配置
        """
        return {
            "type": "sse",
            "url": url,
        }

    @staticmethod
    def create_streamable_http(
        url: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        创建 StreamableHTTP 传输配置

        Args:
            url: 服务器 URL
            headers: 请求头

        Returns:
            传输配置
        """
        return {
            "type": "streamable_http",
            "url": url,
            "headers": headers or {},
        }

    @staticmethod
    def create_memory() -> Dict[str, Any]:
        """
        创建 Memory 传输配置（用于测试）

        Returns:
            传输配置
        """
        return {
            "type": "memory",
        }
