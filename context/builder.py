"""
上下文构建器 (ContextBuilder)

polyagent 第九章 §9.3 GSSC 流水线:
    Gather     — 从系统指令 / 时间 / RAG / 记忆 / 对话历史 / 笔记 / 终端 汇集候选
    Select     — 计算 relevance × recency 综合分,贪心填满 token 预算
    Structure  — 渲染为 [Role & Policies] / [Task] / [State] / [Evidence] / [Context] / [Output] 模板
    Compress   — 超 budget 时按分区截断(委托给 ContextCompressor)

每个候选在 Gather 阶段被包成 ContextPacket,后续阶段不再访问原始数据源,
保证 GSSC 流水线的纯函数特性(便于单测 / A/B / 缓存)。
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from context.compressor import ContextCompressor, _estimate_tokens
from context.config import ContextConfig
from context.packet import (
    ContextPacket,
    PACKET_TYPE_CUSTOM,
    PACKET_TYPE_HISTORY,
    PACKET_TYPE_MEMORY,
    PACKET_TYPE_NOTE,
    PACKET_TYPE_RAG,
    PACKET_TYPE_SYSTEM,
    PACKET_TYPE_TERMINAL,
    PACKET_TYPE_TIME,
    PACKET_TYPE_USER_QUERY,
)
from memory.manager import MemoryManager
from rag.rag_tool import RAGTool


class ContextBuilder:
    """GSSC 流水线实现

    典型用法::

        builder = ContextBuilder(
            config=ContextConfig(max_tokens=4000),
            memory_manager=memory,
            rag_tool=rag,
            note_tool=note,           # 可选
            terminal_tool=terminal,   # 可选
        )
        context_text = builder.build(
            user_query="...",
            conversation_history=[...],
            system_instructions="...",
            include_rag=True, include_memory=True,
            include_history=True, include_notes=True,
        )
    """

    def __init__(
        self,
        config: Optional[ContextConfig] = None,
        memory_manager: Optional[MemoryManager] = None,
        rag_tool: Optional[RAGTool] = None,
        note_tool: Optional[Any] = None,
        terminal_tool: Optional[Any] = None,
        compressor: Optional[ContextCompressor] = None,
    ):
        self.config = config or ContextConfig()
        self.memory = memory_manager
        self.rag = rag_tool
        self.note_tool = note_tool
        self.terminal_tool = terminal_tool
        self.compressor = compressor or ContextCompressor(max_tokens=self.config.max_tokens)

    # ==================================================================
    #  公开入口
    # ==================================================================
    def build(
        self,
        user_query: str,
        conversation_history: Optional[Sequence[Any]] = None,
        system_instructions: Optional[str] = None,
        custom_packets: Optional[List[ContextPacket]] = None,
        include_memory: bool = True,
        include_rag: bool = True,
        include_history: bool = True,
        include_notes: bool = True,
        include_terminal: bool = True,
        terminal_query: Optional[str] = None,
    ) -> str:
        """构建结构化上下文

        Args:
            user_query: 当前用户问题(必填)
            conversation_history: 对话历史(Message-like 对象,有 role/content)
            system_instructions: 系统指令
            custom_packets: 调用方手动注入的候选 packet(§9.4 NoteTool 整合点)
            include_memory / include_rag / include_history / include_notes / include_terminal:
                各数据源开关
            terminal_query: 传给 TerminalTool 的即时查询(默认复用 user_query)

        Returns:
            最终结构化上下文字符串(可直接作为 system message 喂给 LLM)
        """
        # 1) Gather
        packets = self._gather(
            user_query=user_query,
            conversation_history=conversation_history,
            system_instructions=system_instructions,
            custom_packets=custom_packets,
            include_memory=include_memory,
            include_rag=include_rag,
            include_history=include_history,
            include_notes=include_notes,
            include_terminal=include_terminal,
            terminal_query=terminal_query,
        )

        # 2) Select
        selected = self._select(packets, user_query=user_query)

        # 3) Structure
        sections = self._structure(selected, user_query=user_query)

        # 4) Compress(超 budget 时按分区截断)
        final = self._compress(sections)
        return final

    # ==================================================================
    #  GSSC - Gather
    # ==================================================================
    def _gather(
        self,
        user_query: str,
        conversation_history: Optional[Sequence[Any]],
        system_instructions: Optional[str],
        custom_packets: Optional[List[ContextPacket]],
        include_memory: bool,
        include_rag: bool,
        include_history: bool,
        include_notes: bool,
        include_terminal: bool,
        terminal_query: Optional[str],
    ) -> List[ContextPacket]:
        """汇集所有候选信息(每个外部源独立 try/except,失败不影响整体)"""
        now = datetime.now()
        packets: List[ContextPacket] = []

        # 1. 系统指令(最高优先级,score=1.0,不可被 Select 阶段丢弃)
        if system_instructions:
            packets.append(ContextPacket(
                content=system_instructions,
                timestamp=now,
                token_count=_estimate_tokens(system_instructions),
                relevance_score=1.0,
                metadata={"type": PACKET_TYPE_SYSTEM, "priority": "high"},
            ))

        # 2. 当前时间 —— 充当 [State] 段
        time_text = now.strftime("%Y-%m-%d %H:%M:%S")
        packets.append(ContextPacket(
            content=time_text,
            timestamp=now,
            token_count=_estimate_tokens(time_text),
            relevance_score=0.5,
            metadata={"type": PACKET_TYPE_TIME, "priority": "low"},
        ))

        # 3. RAG 检索
        if include_rag and self.rag and user_query:
            try:
                rag_packets = self._gather_from_rag(user_query)
                packets.extend(rag_packets)
            except Exception as e:  # noqa: BLE001
                print(f"[ContextBuilder] RAG 检索失败: {e}")

        # 4. 记忆检索
        if include_memory and self.memory and user_query:
            try:
                mem_packets = self._gather_from_memory(user_query)
                packets.extend(mem_packets)
            except Exception as e:  # noqa: BLE001
                print(f"[ContextBuilder] 记忆检索失败: {e}")

        # 5. 对话历史(只保留最近 N 条)
        if include_history and conversation_history:
            packets.extend(self._gather_from_history(conversation_history))

        # 6. NoteTool(§9.4 集成点)
        if include_notes and self.note_tool and user_query:
            try:
                packets.extend(self._gather_from_notes(user_query))
            except Exception as e:  # noqa: BLE001
                print(f"[ContextBuilder] 笔记检索失败: {e}")

        # 7. TerminalTool(§9.5 JIT 上下文)
        if include_terminal and self.terminal_tool:
            try:
                packets.extend(self._gather_from_terminal(terminal_query or user_query))
            except Exception as e:  # noqa: BLE001
                print(f"[ContextBuilder] 终端检索失败: {e}")

        # 8. 自定义 packet(高优先级)
        if custom_packets:
            packets.extend(custom_packets)

        # 9. 当前问题(始终保留,不参与 Select)
        packets.append(ContextPacket(
            content=user_query,
            timestamp=now,
            token_count=_estimate_tokens(user_query),
            relevance_score=1.0,
            metadata={"type": PACKET_TYPE_USER_QUERY, "priority": "high"},
        ))

        return packets

    def _gather_from_rag(self, query: str) -> List[ContextPacket]:
        """从 RAG 工具汇集候选"""
        if not self.rag:
            return []
        rag_block = self.rag.build_context(query, top_k=self.config.max_rag_results)
        if not rag_block:
            return []
        return [ContextPacket(
            content=rag_block,
            timestamp=datetime.now(),
            token_count=_estimate_tokens(rag_block),
            relevance_score=0.8,
            metadata={"type": PACKET_TYPE_RAG, "source": "rag_tool"},
        )]

    def _gather_from_memory(self, query: str) -> List[ContextPacket]:
        """从记忆系统汇集候选"""
        if not self.memory:
            return []
        items = self.memory.search(query, limit=self.config.max_memory_items)
        if not items:
            return []
        packets: List[ContextPacket] = []
        for item in items:
            content = f"[{item.memory_type}|imp={item.importance:.2f}] {item.content}"
            packets.append(ContextPacket(
                content=content,
                timestamp=getattr(item, "created_at", None) or datetime.now(),
                token_count=_estimate_tokens(content),
                relevance_score=float(getattr(item, "importance", 0.5) or 0.5),
                metadata={
                    "type": PACKET_TYPE_MEMORY,
                    "memory_type": getattr(item, "memory_type", "semantic"),
                    "memory_id": getattr(item, "id", None),
                },
            ))
        return packets

    def _gather_from_history(self, history: Sequence[Any]) -> List[ContextPacket]:
        """从对话历史汇集候选(只取最近 max_history_messages 条)"""
        recent = list(history)[-self.config.max_history_messages:]
        packets: List[ContextPacket] = []
        for msg in recent:
            role = getattr(msg, "role", "user")
            content = getattr(msg, "content", str(msg))
            text = f"[{role}] {content}"
            packets.append(ContextPacket(
                content=text,
                timestamp=getattr(msg, "timestamp", None) or datetime.now(),
                token_count=_estimate_tokens(text),
                relevance_score=0.6,
                metadata={"type": PACKET_TYPE_HISTORY, "role": role},
            ))
        return packets

    def _gather_from_notes(self, query: str) -> List[ContextPacket]:
        """从 NoteTool 汇集候选(§9.4 集成)"""
        if not self.note_tool:
            return []
        try:
            raw = self.note_tool.run({
                "action": "search",
                "query": query,
                "limit": self.config.max_note_items,
            })
        except Exception:
            return []
        notes = self._unwrap_tool_result(raw)
        if not notes:
            return []
        packets: List[ContextPacket] = []
        for note in notes:
            title = note.get("title", "")
            content = note.get("content") or ""
            # search 返回的 content 默认是空的(_note_dict 不展开正文以省 IO),
            # 这里补一次 read 拿到真实正文,失败则用空
            if not content:
                try:
                    detail_raw = self.note_tool.run({
                        "action": "read",
                        "note_id": note.get("note_id", ""),
                    })
                    detail_data = self._unwrap_tool_result(detail_raw)
                    if detail_data and isinstance(detail_data[0], dict):
                        content = detail_data[0].get("content", "")
                except Exception:
                    pass
            note_type = note.get("type", "general")
            text = f"[笔记:{title}]\n{content}".strip()
            updated_at = note.get("updated_at")
            ts = self._parse_iso(updated_at) or datetime.now()
            # blocker > action > conclusion > reference > general
            score = {
                "blocker": 0.9,
                "action": 0.8,
                "conclusion": 0.7,
                "reference": 0.6,
            }.get(note_type, 0.5)
            packets.append(ContextPacket(
                content=text,
                timestamp=ts,
                token_count=_estimate_tokens(text),
                relevance_score=score,
                metadata={
                    "type": PACKET_TYPE_NOTE,
                    "note_type": note_type,
                    "note_id": note.get("note_id"),
                    "title": title,
                },
            ))
        return packets

    def _gather_from_terminal(self, query: str) -> List[ContextPacket]:
        """从 TerminalTool 汇集 JIT 结果(§9.5 集成)"""
        if not self.terminal_tool:
            return []
        try:
            output = self.terminal_tool.run({"command": f"quick_context {query}"})
        except Exception:
            return []
        if not output:
            return []
        text = str(output).strip()
        if not text:
            return []
        return [ContextPacket(
            content=text,
            timestamp=datetime.now(),
            token_count=_estimate_tokens(text),
            relevance_score=0.7,
            metadata={"type": PACKET_TYPE_TERMINAL, "source": "terminal_tool"},
        )]

    @staticmethod
    def _parse_iso(value: Any) -> Optional[datetime]:
        """容忍多种 ISO 字符串格式,失败返回 None"""
        if not value or not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00").replace("T", " "))
        except Exception:
            return None

    @staticmethod
    def _unwrap_tool_result(raw: Any) -> List[Dict[str, Any]]:
        """把 BaseTool 的 run() 返回(可能是 JSON 字符串 / dict / list)解析成 list

        兼容三种返回结构:
            - '[...]'
            - {"success": true, "data": [...]}
            - {"success": true, "data": {...}}   ← 单 dict 也视为 [dict]
            - {"success": true, "data": {"data": [...]}}  ← 双重包装,极少见
        """
        import json as _json
        if raw is None:
            return []
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            data = raw.get("data")
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                # 双重包装或单 dict —— 当 list 处理
                nested = data.get("data")
                if isinstance(nested, list):
                    return nested
                return [data]
            return []
        if isinstance(raw, str):
            try:
                obj = _json.loads(raw)
            except Exception:
                return []
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                data = obj.get("data")
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    nested = data.get("data")
                    if isinstance(nested, list):
                        return nested
                    return [data]
            return []
        return []

    # ==================================================================
    #  GSSC - Select
    # ==================================================================
    def _select(self, packets: List[ContextPacket], user_query: str) -> List[ContextPacket]:
        """按相关性 + 新近性评分,贪心填满 token 预算

        算法:
            1. 分离 high-priority(system_instruction / user_query / state_time)与其他
            2. 对其他 packet 计算 combined_score = relevance_weight*rel + recency_weight*rec
            3. 过滤 min_relevance
            4. 按分数降序,贪心累加 token_count 直到 budget
        """
        cfg = self.config
        budget = cfg.available_tokens

        high = [p for p in packets if p.metadata.get("priority") == "high"]
        others = [p for p in packets if p.metadata.get("priority") != "high"]

        high_tokens = sum(p.token_count for p in high)
        if high_tokens > budget:
            print(f"[ContextBuilder] 高优先级内容({high_tokens}t) 已超预算({budget}t)")
            # 仍保留 system instruction(可截断);user_query 截断
            return high

        # 相关性:若 packet 没设过 relevance,根据关键词重叠补一个
        scored: List[tuple[float, ContextPacket]] = []
        for p in others:
            rel = p.relevance_score
            if rel == 0.5 and p.type != PACKET_TYPE_TIME:
                # 默认值,尝试用关键词重叠补一下
                rel = self._calculate_relevance(p.content, user_query)
                p.relevance_score = rel
            rec = self._calculate_recency(p.timestamp)
            combined = cfg.relevance_weight * rel + cfg.recency_weight * rec
            if rel < cfg.min_relevance:
                continue
            scored.append((combined, p))

        scored.sort(key=lambda x: x[0], reverse=True)

        selected: List[ContextPacket] = list(high)
        used = high_tokens
        for _score, p in scored:
            if used + p.token_count <= budget:
                selected.append(p)
                used += p.token_count
            # 否则跳过(不剪枝,后续更小 token 的可能还能进)
        print(
            f"[ContextBuilder] Select:候选 {len(packets)} → 选中 {len(selected)},"
            f"tokens {used}/{budget}"
        )
        return selected

    @staticmethod
    def _calculate_relevance(content: str, query: str) -> float:
        """Jaccard 关键词重叠(简单实现;生产可替换为向量相似度)"""
        if not query or not content:
            return 0.0
        q = {w for w in query.lower().split() if w}
        c = {w for w in content.lower().split() if w}
        if not q or not c:
            return 0.0
        inter = q & c
        union = q | c
        return len(inter) / len(union) if union else 0.0

    @staticmethod
    def _calculate_recency(timestamp: datetime) -> float:
        """24h 内保持高分,之后指数衰减"""
        age_hours = max(0.0, (datetime.now() - timestamp).total_seconds() / 3600)
        decay = 0.1  # 24h 后约 0.9
        return max(0.1, min(1.0, math.exp(-decay * age_hours / 24)))

    # ==================================================================
    #  GSSC - Structure
    # ==================================================================
    def _structure(self, selected: List[ContextPacket], user_query: str) -> List[str]:
        """把选中的 packet 组织成结构化分区,返回每个分区的字符串列表

        分区(按文档 §9.3.3):
            [Role & Policies]   system_instruction
            [Task]              user_query
            [State]             state_time
            [Evidence]          evidence_rag / evidence_terminal
            [Context]           context_memory / context_history / context_note / context_custom
        最后再追加 [Output] 指引。
        """
        buckets: Dict[str, List[str]] = {
            "role": [],
            "task": [],
            "state": [],
            "evidence": [],
            "context": [],
        }

        for p in selected:
            t = p.type
            content = p.content
            if t == PACKET_TYPE_SYSTEM:
                buckets["role"].append(content)
            elif t == PACKET_TYPE_USER_QUERY:
                buckets["task"].append(content)
            elif t == PACKET_TYPE_TIME:
                buckets["state"].append(f"当前时间:{content}")
            elif t in (PACKET_TYPE_RAG, PACKET_TYPE_TERMINAL):
                buckets["evidence"].append(content)
            else:  # memory / history / note / custom
                buckets["context"].append(content)

        sections: List[str] = []
        if buckets["role"]:
            sections.append("[Role & Policies]\n" + "\n".join(buckets["role"]))
        if buckets["task"]:
            sections.append(f"[Task]\n{user_query}")
        if buckets["state"]:
            sections.append("[State]\n" + "\n".join(buckets["state"]))
        if buckets["evidence"]:
            sections.append("[Evidence]\n" + "\n---\n".join(buckets["evidence"]))
        if buckets["context"]:
            sections.append("[Context]\n" + "\n".join(buckets["context"]))

        # 始终追加 [Output] 段(对回答格式的指引,文档要求)
        sections.append(
            "[Output]\n请基于以上信息,提供准确、有据的回答。"
            "如证据不足请明确说明,不要编造。"
        )
        return sections

    # ==================================================================
    #  GSSC - Compress
    # ==================================================================
    def _compress(self, sections: List[str]) -> str:
        """如果拼接后超 budget,按分区截断"""
        if not sections:
            return ""
        if not self.config.enable_compression:
            return "\n\n".join(sections)
        joined = "\n\n".join(sections)
        if _estimate_tokens(joined) <= self.config.max_tokens:
            return joined
        return self.compressor.partition_compress(sections, max_tokens=self.config.max_tokens)
