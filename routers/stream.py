"""
SSE 流式响应接口

基于 hello-agents 第七章:
- Server-Sent Events (SSE) 流式返回完整路由过程
- 支持分阶段输出:意图/计划/执行/反思/答案
- 兼容所有主流前端(EventSource)
"""
from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from orchestrator import CentralRouter, get_router

router = APIRouter()


# ============================================================
#  SSE 事件类型
# ============================================================
def sse_event(event: str, data: Any) -> Dict[str, str]:
    """构造 SSE 事件"""
    return {
        "event": event,
        "data": json.dumps(data, ensure_ascii=False),
    }


# ============================================================
#  核心:流式中心化路由
# ============================================================
async def stream_central_router(
    router: CentralRouter,
    query: str,
    mode: str = "auto",
    session_id: Optional[str] = None,
) -> AsyncIterator[Dict[str, str]]:
    """
    流式执行完整路由流程。
    Yields SSE 事件 dict。
    """
    session_id = session_id or str(uuid.uuid4())[:8]
    emit = sse_event  # 简写

    # Step 1: 意图识别
    yield emit("step", {"step": 1, "name": "intent", "status": "start", "message": "正在识别意图..."})
    try:
        intent = router.intent_recognizer.recognize(query)
        yield emit("intent", {
            "intent": intent.intent,
            "confidence": intent.confidence,
            "mode": intent.mode,
            "entities": intent.entities,
            "reasoning": intent.reasoning,
        })
        yield emit("step", {"step": 1, "name": "intent", "status": "done", "message": f"意图: {intent.intent}"})
    except Exception as e:
        yield emit("error", {"step": "intent", "error": str(e)})
        intent = None

    # 写记忆
    if router.memory:
        router.memory.add_working(f"用户: {query[:50]}", importance=0.6, session_id=session_id)

    # Step 2: 规划
    chosen = mode if mode != "auto" else (intent.mode if intent else "simple")
    yield emit("step", {"step": 2, "name": "plan", "status": "start", "message": f"选择范式: {chosen}"})
    try:
        if intent:
            plan = router.planner.plan(intent, query)
            yield emit("plan", {"mode": chosen, "steps": [s.to_dict() for s in plan.steps]})
            yield emit("step", {"step": 2, "name": "plan", "status": "done", "message": f"计划 {len(plan.steps)} 步"})
        else:
            plan = None
            yield emit("step", {"step": 2, "name": "plan", "status": "done", "message": "跳过规划"})
    except Exception as e:
        yield emit("error", {"step": "plan", "error": str(e)})
        plan = None

    # Step 3: 执行(流式 LLM 输出)
    yield emit("step", {"step": 3, "name": "execute", "status": "start", "message": "正在执行..."})

    # 根据范式选择执行方式
    if chosen == "react" and plan:
        answer = ""
        try:
            exec_result = router.executor.execute(plan, query, session_id=session_id)
            answer = exec_result.final_answer
            yield emit("execution", {"steps": [
                {"step": s.step, "action": s.action, "success": s.success, "output": str(s.output)[:200]}
                for s in exec_result.steps
            ]})
        except Exception as e:
            yield emit("error", {"step": "execute", "error": str(e)})
            answer = "执行出错"
    elif chosen == "reflection":
        try:
            for chunk in router.reflection_agent.stream_run(query):
                yield emit("chunk", {"content": chunk, "source": "reflection"})
                answer = chunk
        except Exception as e:
            yield emit("error", {"step": "execute", "error": str(e)})
            answer = "反思出错"
    elif chosen == "plan_solve":
        try:
            for chunk in router.plan_solve_agent.stream_run(query):
                yield emit("chunk", {"content": chunk, "source": "plan_solve"})
                answer = chunk
        except Exception as e:
            yield emit("error", {"step": "execute", "error": str(e)})
            answer = "计划执行出错"
    else:
        # simple
        try:
            for chunk in router.simple_agent.stream_run(query):
                yield emit("chunk", {"content": chunk, "source": "simple"})
                answer = chunk
        except Exception as e:
            yield emit("error", {"step": "execute", "error": str(e)})
            answer = "Simple 执行出错"

    yield emit("step", {"step": 3, "name": "execute", "status": "done", "message": "执行完成"})

    # Step 4: 反思
    yield emit("step", {"step": 4, "name": "reflect", "status": "start", "message": "反思评估..."})
    try:
        from orchestrator.executor import ExecutionResult
        from orchestrator.planner import Plan
        exec_result = ExecutionResult(
            plan=plan or Plan(steps=[]),
            final_answer=answer or "",
        )
        reflection = router.reflector.reflect(query, exec_result, answer or "")
        yield emit("reflection", {
            "score": reflection.score,
            "is_satisfactory": reflection.is_satisfactory,
            "issues": reflection.issues,
            "suggestions": reflection.suggestions,
        })
        yield emit("step", {"step": 4, "name": "reflect", "status": "done",
                            "message": f"评分: {reflection.score:.2f}"})
    except Exception as e:
        yield emit("error", {"step": "reflect", "error": str(e)})

    # Step 5: 完成
    yield emit("step", {"step": 5, "name": "done", "status": "done", "message": "完成"})
    yield emit("final", {
        "answer": answer or "",
        "session_id": session_id,
        "used_paradigm": chosen,
    })
    yield emit("done", {"message": "所有流程已完成"})


# ============================================================
#  FastAPI 路由
# ============================================================
@router.get(
    "/stream/chat",
    summary="★ SSE 流式中心化路由",
    description="SSE 流式输出完整路由过程,兼容 EventSource",
)
async def stream_chat(
    query: str,
    mode: str = "auto",
    session_id: Optional[str] = None,
):
    """
    SSE 流式接口。
    前端使用 EventSource 接收:
      source = new EventSource('/api/agent/stream/chat?query=...')
      source.addEventListener('intent', e => ...)
      source.addEventListener('chunk', e => ...)
      source.addEventListener('final', e => ...)
    """
    router = get_router()
    return StreamingResponse(
        stream_central_router(router, query, mode, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/stream/chat/reasoning",
    summary="流式深度思考模式",
    description="流式输出 LLM 的推理过程(reasoning_content) + 最终答案",
)
async def stream_reasoning(
    query: str,
    mode: str = "simple",
):
    """流式输出推理过程和最终答案"""
    r = get_router()

    async def _gen():
        # 构建消息
        system_prompt = "You are a helpful AI assistant. Answer the user question directly."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        # 流式
        thinking_buffer = ""
        answer_buffer = ""
        is_answering = False

        for reasoning, content in r.llm.stream_with_reasoning(messages):
            if reasoning and not is_answering:
                thinking_buffer += reasoning
                yield f"event: thinking\ndata: {json.dumps({'reasoning': reasoning}, ensure_ascii=False)}\n\n"
            if content:
                if not is_answering:
                    is_answering = True
                    yield f"event: answer_start\ndata: {{}}\n\n"
                answer_buffer += content
                yield f"event: answer\ndata: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"

        yield f"event: done\ndata: {json.dumps({'answer': answer_buffer}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )