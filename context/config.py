"""
上下文构建配置

基于 hello-agents 第九章 GSSC 流水线:
- 控制上下文长度、保留比例、压缩阈值
- recency_weight / relevance_weight 用于 Select 阶段评分
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ContextConfig:
    """上下文构建配置

    Attributes:
        max_tokens: 最大 token 数量(系统+用户侧合计预算)
        reserve_ratio: 为系统指令预留的比例(0.0-1.0)
        min_relevance: 最低相关性阈值(0.0-1.0),低于此值的 packet 被丢弃
        enable_compression: 是否启用压缩(超 budget 时触发)
        recency_weight: 新近性权重(0.0-1.0)
        relevance_weight: 相关性权重(0.0-1.0),二者之和需为 1
        max_history_messages: 最多保留多少条对话历史
        max_memory_items: 最多召回多少条记忆
        max_rag_results: 最多召回多少条 RAG 结果
        max_note_items: 最多召回多少条 NoteTool 笔记
        max_terminal_items: 最多召回多少条 TerminalTool JIT 结果
    """

    max_tokens: int = 3000
    reserve_ratio: float = 0.2
    min_relevance: float = 0.1
    enable_compression: bool = True
    recency_weight: float = 0.3
    relevance_weight: float = 0.7

    max_history_messages: int = 10
    max_memory_items: int = 5
    max_rag_results: int = 3
    max_note_items: int = 3
    max_terminal_items: int = 3

    def __post_init__(self) -> None:
        if not 0.0 <= self.reserve_ratio <= 1.0:
            raise ValueError("reserve_ratio 必须在 [0, 1] 范围内")
        if not 0.0 <= self.min_relevance <= 1.0:
            raise ValueError("min_relevance 必须在 [0, 1] 范围内")
        if not 0.0 <= self.recency_weight <= 1.0 or not 0.0 <= self.relevance_weight <= 1.0:
            raise ValueError("recency_weight / relevance_weight 必须在 [0, 1] 范围内")
        if abs(self.recency_weight + self.relevance_weight - 1.0) > 1e-6:
            raise ValueError(
                f"recency_weight + relevance_weight 必须等于 1.0 "
                f"(got {self.recency_weight + self.relevance_weight:.4f})"
            )

    @property
    def available_tokens(self) -> int:
        """扣除系统指令预留后的可用 token 数"""
        return int(self.max_tokens * (1 - self.reserve_ratio))
