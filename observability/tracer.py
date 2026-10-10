"""
可观测性模块 (Observability)

基于 polyagent 第十二章:
- 调用追踪(Tracer):每次 LLM 调用记录
- 结构化日志(Logger)
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TraceEvent:
    """追踪事件"""
    event_type: str         # "llm_call" / "tool_call" / "intent" / "plan" / "reflect"
    timestamp: float = field(default_factory=time.time)
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    span_id: str = ""
    parent_id: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    error: Optional[str] = None


class Tracer:
    """
    调用追踪器。
    记录每个 LLM 调用、工具调用、意图识别、规划、反思等事件。
    支持内存缓冲和文件持久化。
    """

    def __init__(self, persist_path: Optional[str] = None, max_buffer: int = 100):
        self.persist_path = Path(persist_path) if persist_path else None
        self.max_buffer = max_buffer
        self.buffer: List[TraceEvent] = []
        self._trace_stack: List[str] = []   # 用于构建 span hierarchy

    # -------- 上下文管理 --------
    def start_trace(self, trace_id: str, parent_id: str = "") -> str:
        """开始一个 trace"""
        self._trace_stack.append(trace_id)
        return trace_id

    def end_trace(self) -> None:
        self._trace_stack.pop()

    def _current_trace_id(self) -> str:
        return self._trace_stack[-1] if self._trace_stack else ""

    def _current_span_id(self) -> str:
        return str(uuid.uuid4())[:8]

    # -------- 记录事件 --------
    def record(self, event_type: str, data: Dict[str, Any], duration_ms: float = 0.0, error: Optional[str] = None) -> TraceEvent:
        event = TraceEvent(
            event_type=event_type,
            trace_id=self._current_trace_id(),
            span_id=self._current_span_id(),
            parent_id=self._trace_stack[-2] if len(self._trace_stack) >= 2 else "",
            data=data,
            duration_ms=duration_ms,
            error=error,
        )
        self.buffer.append(event)
        if len(self.buffer) >= self.max_buffer:
            self._flush()
        return event

    def llm_call(
        self,
        model: str,
        prompt_preview: str,
        response_preview: str = "",
        tokens: int = 0,
        duration_ms: float = 0.0,
        error: Optional[str] = None,
    ) -> None:
        self.record("llm_call", {
            "model": model,
            "prompt_preview": prompt_preview[:200],
            "response_preview": response_preview[:200],
            "tokens": tokens,
        }, duration_ms, error)

    def tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: str = "",
        success: bool = True,
        duration_ms: float = 0.0,
    ) -> None:
        self.record("tool_call", {
            "tool_name": tool_name,
            "args": {k: str(v)[:100] for k, v in args.items()},
            "result_preview": result[:200],
            "success": success,
        }, duration_ms, None if success else "Tool failed")

    def intent_recognized(
        self, intent: str, confidence: float, entities: Dict[str, str], duration_ms: float = 0.0
    ) -> None:
        self.record("intent", {
            "intent": intent,
            "confidence": confidence,
            "entities": entities,
        }, duration_ms)

    def plan_created(self, steps: List[Dict], duration_ms: float = 0.0) -> None:
        self.record("plan", {"step_count": len(steps), "steps": steps}, duration_ms)

    def reflect_done(self, score: float, is_satisfactory: bool, duration_ms: float = 0.0) -> None:
        self.record("reflect", {
            "score": score,
            "is_satisfactory": is_satisfactory,
        }, duration_ms)

    # -------- 持久化 --------
    def _flush(self) -> None:
        if not self.buffer or not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, "a", encoding="utf-8") as f:
            for event in self.buffer:
                f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
        self.buffer.clear()

    def flush(self) -> int:
        n = len(self.buffer)
        self._flush()
        return n

    def get_recent(self, limit: int = 20) -> List[TraceEvent]:
        """返回 buffer 中最近 N 条事件(不 flush)"""
        return self.buffer[-limit:]

    def stats(self) -> Dict[str, Any]:
        """统计当前 buffer(不 flush,避免丢失待写数据)"""
        llm = [e for e in self.buffer if e.event_type == "llm_call"]
        tool = [e for e in self.buffer if e.event_type == "tool_call"]
        return {
            "total_events": len(self.buffer),
            "llm_calls": len(llm),
            "tool_calls": len(tool),
            "avg_duration_ms": sum(e.duration_ms for e in self.buffer) / max(len(self.buffer), 1),
            "errors": sum(1 for e in self.buffer if e.error),
        }