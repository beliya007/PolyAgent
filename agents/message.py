"""
消息类(向后兼容,推荐使用 core.message.Message)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

# 直接复用 core 的 Message / ToolCall / ToolResult,保持 API 兼容
from core.message import Message, ToolCall, ToolResult  # noqa: F401


@dataclass
class LegacyMessage:
    """旧版 Message(兼容旧代码)"""
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    name: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_openai(self) -> Dict[str, str]:
        m: Dict[str, str] = {"role": self.role, "content": self.content}
        if self.name:
            m["name"] = self.name
        return m

    @classmethod
    def system(cls, content: str) -> "LegacyMessage":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "LegacyMessage":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> "LegacyMessage":
        return cls(role="assistant", content=content)

    @classmethod
    def tool(cls, content: str, name: str) -> "LegacyMessage":
        return cls(role="tool", content=content, name=name)


# 默认导出新的 Message
__all__ = ["Message", "LegacyMessage", "ToolCall", "ToolResult"]