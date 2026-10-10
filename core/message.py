"""
核心消息类

基于 hello-agents 第七章 + OpenAI function calling 协议。
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    """统一的工具调用(对齐 OpenAI function calling)"""
    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_openai(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }

    @classmethod
    def from_openai(cls, tc: Any) -> "ToolCall":
        args_raw = getattr(getattr(tc, "function", None), "arguments", "{}") or "{}"
        try:
            args = json.loads(args_raw)
        except (json.JSONDecodeError, TypeError):
            args = {"_raw": args_raw}
        return cls(
            id=getattr(tc, "id", str(uuid.uuid4())[:8]),
            name=getattr(getattr(tc, "function", None), "name", "") or "",
            arguments=args,
        )


@dataclass
class ToolResult:
    """工具执行结果"""
    tool_call_id: str
    name: str
    content: str
    success: bool = True
    error: Optional[str] = None

    def to_message(self) -> Dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": self.content if self.success else f"[ERROR] {self.error}",
        }


@dataclass
class Message:
    """对话消息"""
    role: str                          # system / user / assistant / tool
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None # for role='tool'
    name: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 工厂方法
    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str, tool_calls: Optional[List[ToolCall]] = None) -> "Message":
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: str = "") -> "Message":
        return cls(role="tool", content=content, tool_call_id=tool_call_id, name=name)

    def to_openai(self) -> Dict[str, Any]:
        """转为 OpenAI ChatMessage 格式"""
        m: Dict[str, Any] = {"role": self.role}
        if self.role == "tool":
            m["tool_call_id"] = self.tool_call_id or ""
            m["content"] = self.content
            if self.name:
                m["name"] = self.name
        else:
            m["content"] = self.content
            if self.tool_calls:
                m["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
            if self.name:
                m["name"] = self.name
        return m