"""
记忆工具 (MemoryTool)

hello-agents 第八章 MemoryTool 设计:
- 作为 Agent 可调用的工具,统一入口分发到 MemoryManager
- 支持操作:add / search / summary / stats / update / remove /
  forget / consolidate / clear_all / get
- 自动管理会话 ID(session_id),为每个新会话生成一次
- 元数据自动补充 user_id / timestamp / session_id
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from memory.manager import MemoryManager
from memory.config import MemoryConfig, get_memory_config
from tools.base import BaseTool, ToolParameter


_TYPE_LABEL = {
    "working": "工作记忆",
    "episodic": "情景记忆",
    "semantic": "语义记忆",
    "perceptual": "感知记忆",
}


class MemoryTool(BaseTool):
    """记忆工具 - 为 Agent 提供记忆能力"""

    name = "memory"
    description = (
        "记忆工具 — 存储和检索对话历史、知识、经验。"
        "支持 4 种记忆(working/episodic/semantic/perceptual),"
        "支持向量/图/SQLite 多后端。"
    )
    parameters = [
        ToolParameter(
            "action", "string",
            "操作类型:add / search / summary / stats / update / remove / forget / consolidate / clear_all / get",
            required=True,
            enum=["add", "search", "summary", "stats", "update", "remove",
                  "forget", "consolidate", "clear_all", "get"],
        ),
        ToolParameter("content", "string", "记忆内容(add 用)", required=False),
        ToolParameter(
            "memory_type", "string",
            "记忆类型(默认 working)", required=False, default="working",
            enum=["working", "episodic", "semantic", "perceptual"],
        ),
        ToolParameter("importance", "number", "重要性 0~1(默认 0.5)", required=False, default=0.5),
        ToolParameter("query", "string", "查询文本(search 用)", required=False),
        ToolParameter("limit", "integer", "返回条数(默认 5)", required=False, default=5),
        ToolParameter("min_importance", "number", "重要性阈值(默认 0.0)", required=False, default=0.0),
        ToolParameter("memory_id", "string", "记忆 ID(update/remove/get 用)", required=False),
        ToolParameter(
            "strategy", "string",
            "遗忘策略(importance_based/time_based/capacity_based)", required=False,
            default="importance_based",
            enum=["importance_based", "time_based", "capacity_based"],
        ),
        ToolParameter("threshold", "number", "遗忘阈值(默认 0.2)", required=False, default=0.2),
        ToolParameter("max_age_days", "integer", "时间遗忘上限(天)", required=False, default=30),
        ToolParameter("from_type", "string", "整合源类型", required=False, default="working"),
        ToolParameter("to_type", "string", "整合目标类型", required=False, default="episodic"),
        ToolParameter("importance_threshold", "number", "整合重要性阈值(默认 0.6)", required=False, default=0.6),
    ]

    def __init__(
        self,
        user_id: str = "default_user",
        memory_config: Optional[MemoryConfig] = None,
        memory_manager: Optional[MemoryManager] = None,
        enabled_types: Optional[List[str]] = None,
    ):
        super().__init__()
        self.user_id = user_id
        self.memory_config = memory_config or get_memory_config()
        self.memory_manager = memory_manager or MemoryManager(
            user_id=user_id,
            config=self.memory_config,
            enabled_types=enabled_types,
        )
        self.current_session_id: Optional[str] = None

    # ============================================================
    #  入口
    # ============================================================
    def execute(self, **kwargs) -> str:
        action = kwargs.get("action", "").lower()
        try:
            if action == "add":
                return self._add(**kwargs)
            elif action == "search":
                return self._search(**kwargs)
            elif action == "summary":
                return self._summary(**kwargs)
            elif action == "stats":
                return self._stats()
            elif action == "update":
                return self._update(**kwargs)
            elif action == "remove":
                return self._remove(**kwargs)
            elif action == "forget":
                return self._forget(**kwargs)
            elif action == "consolidate":
                return self._consolidate(**kwargs)
            elif action == "clear_all":
                return self._clear_all()
            elif action == "get":
                return self._get(**kwargs)
            else:
                return f"❌ 未知操作: {action}"
        except Exception as e:
            return f"❌ 记忆操作失败 ({action}): {e}"

    # ============================================================
    #  会话管理
    # ============================================================
    def _ensure_session(self) -> str:
        if self.current_session_id is None:
            self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return self.current_session_id

    # ============================================================
    #  操作: add
    # ============================================================
    def _add(
        self,
        content: str = "",
        memory_type: str = "working",
        importance: float = 0.5,
        **_unused,
    ) -> str:
        if not content:
            return "❌ add 需要 content 参数"
        meta: Dict[str, Any] = dict(_unused)
        meta.pop("file_path", None)
        meta.pop("modality", None)
        session_id = self._ensure_session()
        meta.setdefault("session_id", session_id)
        meta.setdefault("timestamp", datetime.now().isoformat())
        meta.setdefault("user_id", self.user_id)
        mid = self.memory_manager.add(
            content=content,
            memory_type=memory_type,
            importance=float(importance),
            metadata=meta,
        )
        if mid is None:
            return f"❌ 未启用 {memory_type} 类型的记忆"
        return f"✅ 记忆已添加 (ID:{mid[:8]}…, type={memory_type}, imp={importance:.2f})"

    # ============================================================
    #  操作: search
    # ============================================================
    def _search(
        self,
        query: str = "",
        memory_type: Optional[str] = None,
        limit: int = 5,
        min_importance: float = 0.0,
        **_unused,
    ) -> str:
        if not query:
            return "❌ search 需要 query 参数"
        results = self.memory_manager.search(
            query=query,
            memory_type=memory_type,
            limit=int(limit),
            min_importance=float(min_importance),
        )
        if not results:
            return f"🔍 未找到与 '{query}' 相关的记忆"
        lines = [f"🔍 找到 {len(results)} 条相关记忆:"]
        for i, m in enumerate(results, 1):
            label = _TYPE_LABEL.get(m.memory_type, m.memory_type)
            preview = m.content[:80] + ("..." if len(m.content) > 80 else "")
            lines.append(
                f"  ({i}) [{label}|imp={m.importance:.2f}|{m.id[:8]}] {preview}"
            )
        return "\n".join(lines)

    # ============================================================
    #  操作: summary / stats
    # ============================================================
    def _summary(self, query: Optional[str] = None, **_unused) -> str:
        if query:
            return f"📋 与 '{query}' 相关记忆摘要:\n{self.memory_manager.summary(query=query)}"
        return f"📋 记忆摘要:\n{self.memory_manager.summary()}"

    def _stats(self) -> str:
        s = self.memory_manager.stats()
        return f"📊 记忆统计:\n{json.dumps(s, ensure_ascii=False, indent=2, default=str)}"

    # ============================================================
    #  操作: get / update / remove
    # ============================================================
    def _get(self, memory_id: str = "", memory_type: str = "episodic", **_unused) -> str:
        if not memory_id:
            return "❌ get 需要 memory_id"
        for mem in self.memory_manager.memory_types.values():
            for it in mem.items:
                if it.id == memory_id or it.id.startswith(memory_id):
                    return json.dumps(it.to_dict(), ensure_ascii=False, indent=2)
        return f"❌ 未找到 memory_id={memory_id}"

    def _update(
        self,
        memory_id: str = "",
        memory_type: str = "episodic",
        content: Optional[str] = None,
        importance: Optional[float] = None,
        **_unused,
    ) -> str:
        if not memory_id:
            return "❌ update 需要 memory_id"
        mem = self.memory_manager.memory_types.get(memory_type)
        if mem is None:
            return f"❌ 未启用 {memory_type} 类型"
        for it in mem.items:
            if it.id == memory_id or it.id.startswith(memory_id):
                if content is not None:
                    it.content = content
                if importance is not None:
                    it.importance = float(importance)
                return f"✅ 已更新记忆 {memory_id[:8]}"
        return f"❌ 未找到 memory_id={memory_id}"

    def _remove(
        self,
        memory_id: str = "",
        memory_type: str = "episodic",
        **_unused,
    ) -> str:
        if not memory_id:
            return "❌ remove 需要 memory_id"
        mem = self.memory_manager.memory_types.get(memory_type)
        if mem is None:
            return f"❌ 未启用 {memory_type} 类型"
        before = len(mem.items)
        mem.items = [i for i in mem.items if not (i.id == memory_id or i.id.startswith(memory_id))]
        diff = before - len(mem.items)
        return f"🗑️ 已删除 {diff} 条记忆" if diff else f"❌ 未找到 memory_id={memory_id}"

    # ============================================================
    #  操作: forget / consolidate / clear_all
    # ============================================================
    def _forget(
        self,
        strategy: str = "importance_based",
        threshold: float = 0.2,
        max_age_days: int = 30,
        **_unused,
    ) -> str:
        n = self.memory_manager.forget(
            strategy=strategy,
            threshold=float(threshold),
            max_age_days=int(max_age_days),
        )
        return f"🧹 已遗忘 {n} 条记忆 (策略:{strategy})"

    def _consolidate(
        self,
        from_type: str = "working",
        to_type: str = "episodic",
        importance_threshold: float = 0.6,
        **_unused,
    ) -> str:
        n = self.memory_manager.consolidate(
            from_type=from_type, to_type=to_type,
            importance_threshold=float(importance_threshold),
        )
        return f"🔄 已整合 {n} 条记忆 ({from_type}→{to_type}, 阈值={importance_threshold})"

    def _clear_all(self) -> str:
        result = self.memory_manager.clear_all()
        return f"🗑️ 已清空全部记忆: {result}"
