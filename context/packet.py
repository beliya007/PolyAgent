"""
上下文候选信息包 (ContextPacket)

hello-agents 第九章 §9.3.2:
- 候选信息包是 GSSC 流水线的基本单元
- 每个 packet 携带 content / timestamp / token_count / relevance_score / metadata
- metadata["type"] 决定 Structure 阶段它会落到 [Role&Policies] / [Evidence] / [Context] / [State] 哪个分区
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


# metadata["type"] 的合法取值 —— 与 _structure() 中的分区一一对应
PACKET_TYPE_SYSTEM = "system_instruction"
PACKET_TYPE_TIME = "state_time"
PACKET_TYPE_RAG = "evidence_rag"
PACKET_TYPE_MEMORY = "context_memory"
PACKET_TYPE_HISTORY = "context_history"
PACKET_TYPE_NOTE = "context_note"
PACKET_TYPE_TERMINAL = "evidence_terminal"
PACKET_TYPE_CUSTOM = "context_custom"
PACKET_TYPE_USER_QUERY = "task_user_query"


@dataclass
class ContextPacket:
    """候选信息包

    Attributes:
        content: 信息内容(原始字符串,Structure 阶段才渲染成最终模板)
        timestamp: 时间戳(用于 _calculate_recency)
        token_count: Token 数量(由 ContextBuilder 在 gather 时计算)
        relevance_score: 相关性分数(0.0-1.0),Select 阶段会读这个值
        metadata: 可选的元数据(type / role / note_id / path 等)
    """

    content: str
    timestamp: datetime
    token_count: int
    relevance_score: float = 0.5
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}
        # 相关性分数必须落在 [0, 1]
        self.relevance_score = max(0.0, min(1.0, float(self.relevance_score)))
        if self.token_count < 0:
            self.token_count = 0

    @property
    def type(self) -> str:
        return self.metadata.get("type", PACKET_TYPE_CUSTOM)

    @property
    def priority(self) -> str:
        return self.metadata.get("priority", "normal")
