"""
LLM 客户端

基于 hello-agents 第四章 + 第七章 + 第十章设计:
- 单一第三方依赖: openai SDK
- 兼容 OpenAI 接口(支持阿里云百炼 / OpenAI / Ollama)
- 支持三种调用:think / stream_invoke / astream
- 支持深度思考模式(enable_thinking)用于阿里云百炼
- 支持原生 function calling 协议(tools 参数)
- 支持 JSON 结构化输出(response_format)
- 内置 Embedding 客户端(用于 RAG)
- 调用追踪(observability)

设计原则:
- 无高层抽象
- 一处配置,多处使用(通过 config.Config)
- 流式优先(默认)
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple, Union

from openai import AsyncOpenAI, OpenAI

from config import get_config


# ============================================================
#  数据结构
# ============================================================
@dataclass
class ToolCall:
    """统一的工具调用结构"""
    id: str
    name: str
    arguments: Dict[str, Any]

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
    def from_openai(cls, tc) -> "ToolCall":
        args_raw = getattr(tc.function, "arguments", "{}") or "{}"
        try:
            args = json.loads(args_raw)
        except (json.JSONDecodeError, TypeError):
            args = {"_raw": args_raw}
        return cls(
            id=getattr(tc, "id", str(uuid.uuid4())),
            name=getattr(tc.function, "name", ""),
            arguments=args,
        )


@dataclass
class LLMResponse:
    """统一的 LLM 响应"""
    content: str = ""
    reasoning: str = ""                  # 阿里云深度思考 / OpenAI o1
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: Dict[str, int] = field(default_factory=dict)
    raw: Any = None


# ============================================================
#  工具 schema 转换
# ============================================================
def tool_to_openai_schema(tool) -> Dict[str, Any]:
    """把 BaseTool 转为 OpenAI function calling schema"""
    schema = tool.schema()
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for p in schema.parameters:
        # type 映射
        type_map = {
            "string": "string",
            "number": "number",
            "integer": "integer",
            "boolean": "boolean",
            "array": "array",
            "object": "object",
        }
        prop: Dict[str, Any] = {
            "type": type_map.get(p.type, "string"),
            "description": p.description or "",
        }
        if p.enum:
            prop["enum"] = list(p.enum)
        properties[p.name] = prop
        if p.required:
            required.append(p.name)
    return {
        "type": "function",
        "function": {
            "name": schema.name,
            "description": schema.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


# ============================================================
#  LLM 客户端
# ============================================================
class HelloAgentsLLM:
    """兼容 OpenAI 接口的 LLM 客户端"""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        embedding_model: Optional[str] = None,
        default_temperature: float = 0.7,
        enable_thinking: Optional[bool] = None,
    ):
        cfg = get_config().llm

        self.model = model or cfg.model
        self.api_key = api_key or cfg.api_key
        self.base_url = base_url or cfg.base_url
        self.timeout = timeout or cfg.timeout
        self.embedding_model = embedding_model or cfg.embedding_model
        self.default_temperature = default_temperature
        self.enable_thinking = (
            enable_thinking if enable_thinking is not None else cfg.enable_thinking
        )

        if not self.api_key:
            raise ValueError(
                "LLM_API_KEY 未配置。请在 agent/.env 中设置,或显式传入 api_key。"
            )

        self._client = OpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )
        self._async_client = AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )

        # 调用统计
        self.call_count = 0
        self.total_tokens = 0
        self.provider = self._guess_provider(self.base_url)
        # 调用记录(用于可观测性)
        self.call_history: List[Dict[str, Any]] = []

    @staticmethod
    def _guess_provider(base_url: str) -> str:
        u = base_url.lower()
        if "dashscope" in u or "aliyun" in u or "maas" in u:
            return "aliyun_maas"
        if "modelscope" in u:
            return "modelscope"
        if "openai.com" in u:
            return "openai"
        if "localhost" in u or "ollama" in u or "127.0.0.1" in u:
            return "ollama"
        return "custom"

    # ============================================================
    #  ★ 核心:同步调用(支持 function calling)
    # ============================================================
    def invoke(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        enable_thinking: Optional[bool] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """同步调用。返回纯文本(content),如果需要工具调用,使用 invoke_with_tools。"""
        kwargs = self._build_kwargs(
            messages, temperature, max_tokens, stream, enable_thinking,
            tools, tool_choice, response_format,
        )
        try:
            response = self._client.chat.completions.create(**kwargs)
            if not stream:
                content = response.choices[0].message.content or ""
                self._record_usage(response.usage)
                return content
            # 流式
            parts: List[str] = []
            for chunk in response:
                if not chunk.choices:
                    continue
                piece = chunk.choices[0].delta.content or ""
                parts.append(piece)
            self.call_count += 1
            return "".join(parts)
        except Exception as e:
            print(f"❌ LLM 调用失败: {e}", file=sys.stderr)
            return None

    def invoke_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """
        同步调用,返回结构化响应(含 tool_calls）。
        即使传了 tools,模型也可能选择不调用,此时 tool_calls 为空。
        """
        kwargs = self._build_kwargs(
            messages, temperature, max_tokens, False, enable_thinking,
            tools, tool_choice, response_format,
        )
        t0 = time.time()
        try:
            response = self._client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            msg = choice.message
            tool_calls: List[ToolCall] = []
            if getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    tool_calls.append(ToolCall.from_openai(tc))
            result = LLMResponse(
                content=msg.content or "",
                reasoning=getattr(msg, "reasoning_content", "") or "",
                tool_calls=tool_calls,
                finish_reason=choice.finish_reason or "stop",
                usage={
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(response.usage, "total_tokens", 0) or 0,
                },
                raw=response,
            )
            self._record_usage(response.usage)
            self._trace(messages, result, kwargs, t0)
            return result
        except Exception as e:
            print(f"❌ LLM 调用失败: {e}", file=sys.stderr)
            self._trace(messages, None, kwargs, t0, error=str(e))
            return LLMResponse(content="", finish_reason="error")

    # ============================================================
    #  ★ 核心:流式
    # ============================================================
    def stream_invoke(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        kwargs = self._build_kwargs(messages, temperature, max_tokens, True, None, None, None, None)
        try:
            response = self._client.chat.completions.create(**kwargs)
            for chunk in response:
                if not chunk.choices:
                    continue
                piece = chunk.choices[0].delta.content or ""
                if piece:
                    yield piece
            self.call_count += 1
        except Exception as e:
            print(f"❌ 流式调用失败: {e}", file=sys.stderr)
            yield ""

    def stream_with_reasoning(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
    ) -> Iterator[Tuple[Optional[str], Optional[str]]]:
        """
        流式迭代器,分离推理过程与最终答案。
        Yields: (reasoning_content, final_content)
        """
        thinking = (
            enable_thinking if enable_thinking is not None else self.enable_thinking
        )
        kwargs = self._build_kwargs(messages, temperature, max_tokens, True, thinking, None, None, None)
        try:
            response = self._client.chat.completions.create(**kwargs)
            seen_content = False
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                reasoning = getattr(delta, "reasoning_content", None)
                content = getattr(delta, "content", None)
                if reasoning and not seen_content:
                    yield reasoning, None
                if content:
                    seen_content = True
                    yield None, content
            self.call_count += 1
        except Exception as e:
            print(f"❌ 流式调用失败: {e}", file=sys.stderr)
            yield None, ""

    def stream_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        流式 + function calling。
        Yields:
          {"type": "content", "delta": "..."}               内容增量
          {"type": "tool_call_delta", "index": 0, "delta": "..."}   工具参数增量
          {"type": "tool_call_done", "tool_call": {...}}    工具调用完成
          {"type": "finish", "finish_reason": "..."}
        """
        kwargs = self._build_kwargs(messages, temperature, max_tokens, True, None, tools, None, None)
        try:
            response = self._client.chat.completions.create(**kwargs)
            tool_call_buffers: Dict[int, Dict[str, str]] = {}
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                # 普通内容
                if getattr(delta, "content", None):
                    yield {"type": "content", "delta": delta.content}
                # 工具调用增量
                if getattr(delta, "tool_calls", None):
                    for tc in delta.tool_calls:
                        idx = tc.index
                        buf = tool_call_buffers.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                        if tc.id:
                            buf["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                buf["name"] = tc.function.name
                            if tc.function.arguments:
                                buf["arguments"] += tc.function.arguments
                                yield {"type": "tool_call_delta", "index": idx, "delta": tc.function.arguments}
            # 收尾:把所有 buffer 转成 ToolCall
            for idx, buf in tool_call_buffers.items():
                try:
                    args = json.loads(buf["arguments"]) if buf["arguments"] else {}
                except json.JSONDecodeError:
                    args = {"_raw": buf["arguments"]}
                yield {
                    "type": "tool_call_done",
                    "tool_call": ToolCall(
                        id=buf["id"] or str(uuid.uuid4()),
                        name=buf["name"],
                        arguments=args,
                    ).to_openai(),
                }
            yield {"type": "finish", "finish_reason": "stop"}
            self.call_count += 1
        except Exception as e:
            print(f"❌ 流式调用失败: {e}", file=sys.stderr)
            yield {"type": "error", "error": str(e)}

    # ============================================================
    #  ★ 异步接口
    # ============================================================
    async def ainvoke(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Optional[Union[str, LLMResponse]]:
        """异步调用。返回字符串或 LLMResponse。"""
        use_tools = bool(tools)
        kwargs = self._build_kwargs(
            messages, temperature, max_tokens, False, None,
            tools if use_tools else None,
            tool_choice if use_tools else None,
            None,
        )
        try:
            response = await self._async_client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            msg = choice.message
            self._record_usage(response.usage)
            if use_tools and getattr(msg, "tool_calls", None):
                tool_calls = [ToolCall.from_openai(tc) for tc in msg.tool_calls]
                return LLMResponse(
                    content=msg.content or "",
                    reasoning=getattr(msg, "reasoning_content", "") or "",
                    tool_calls=tool_calls,
                    finish_reason=choice.finish_reason or "stop",
                    usage={
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                        "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                        "total_tokens": getattr(response.usage, "total_tokens", 0) or 0,
                    },
                )
            return msg.content or ""
        except Exception as e:
            print(f"❌ 异步调用失败: {e}", file=sys.stderr)
            return None if not use_tools else LLMResponse()

    async def astream(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        kwargs = self._build_kwargs(messages, temperature, max_tokens, True, None, None, None, None)
        try:
            response = await self._async_client.chat.completions.create(**kwargs)
            async for chunk in response:
                if not chunk.choices:
                    continue
                piece = chunk.choices[0].delta.content or ""
                if piece:
                    yield piece
            self.call_count += 1
        except Exception as e:
            print(f"❌ 异步流式失败: {e}", file=sys.stderr)
            yield ""

    # ============================================================
    #  ★ Embedding (用于 RAG)
    # ============================================================
    def embed(self, texts: List[str]) -> List[List[float]]:
        """同步获取文本向量。"""
        if not texts:
            return []
        try:
            response = self._client.embeddings.create(
                model=self.embedding_model,
                input=texts,
            )
            vectors = [d.embedding for d in response.data]
            self.call_count += 1
            self.total_tokens += getattr(response.usage, "total_tokens", 0) or 0
            return vectors
        except Exception as e:
            print(f"❌ Embedding 调用失败: {e}", file=sys.stderr)
            dim = 1024
            return [[0.0] * dim for _ in texts]

    # ============================================================
    #  工具
    # ============================================================
    def _build_kwargs(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        stream: bool,
        enable_thinking: Optional[bool],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Union[str, Dict[str, Any]]],
        response_format: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": (
                temperature if temperature is not None else self.default_temperature
            ),
            "stream": stream,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if enable_thinking:
            kwargs["extra_body"] = {"enable_thinking": True}
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
        if response_format:
            kwargs["response_format"] = response_format
        return kwargs

    def _record_usage(self, usage: Any) -> None:
        self.call_count += 1
        if usage and getattr(usage, "total_tokens", None):
            self.total_tokens += usage.total_tokens

    def _trace(
        self,
        messages: List[Dict[str, Any]],
        response: Optional[LLMResponse],
        kwargs: Dict[str, Any],
        t0: float,
        error: Optional[str] = None,
    ) -> None:
        """记录调用(用于可观测性)"""
        # 只保留最后一条 user 消息的预览,避免冗长
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                last_user = content[:200] + ("..." if len(content) > 200 else "")
                break
        rec = {
            "ts": time.time(),
            "elapsed": time.time() - t0,
            "model": kwargs.get("model"),
            "has_tools": bool(kwargs.get("tools")),
            "user_preview": last_user,
            "finish_reason": response.finish_reason if response else None,
            "tokens": response.usage if response else {},
            "error": error,
        }
        self.call_history.append(rec)
        # 上限保护
        if len(self.call_history) > 200:
            self.call_history = self.call_history[-200:]

    def stats(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "embedding_model": self.embedding_model,
            "base_url": self.base_url,
            "enable_thinking": self.enable_thinking,
            "call_count": self.call_count,
            "total_tokens": self.total_tokens,
        }


# ============================================================
#  便捷工厂
# ============================================================
_llm_instance: Optional[HelloAgentsLLM] = None


def build_llm() -> HelloAgentsLLM:
    """单例 LLM 客户端"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = HelloAgentsLLM()
    return _llm_instance


def reset_llm() -> None:
    """重置单例(配置变更时使用)"""
    global _llm_instance
    _llm_instance = None


# ============================================================
#  向后兼容别名(供旧代码使用)
# ============================================================
def think(
    messages: List[Dict[str, str]],
    **kwargs,
) -> Optional[str]:
    """兼容 hello-agents 第四章 think 接口"""
    return build_llm().invoke(messages, **kwargs)


if __name__ == "__main__":
    llm = build_llm()
    print(f"Provider: {llm.provider}")
    print(f"Model:    {llm.model}")
    print(f"Think:    {llm.enable_thinking}")
    print()
    resp = llm.invoke([{"role": "user", "content": "用一句话介绍你自己"}])
    print("\n回答:", resp)
    print()
    print("Embedding 测试:")
    vecs = llm.embed(["测试文本", "hello world"])
    print(f"向量维度: {len(vecs[0]) if vecs else 0}")
    print()
    print("Function calling 测试:")
    tools = [tool_to_openai_schema(t) for t in [
        type("T", (), {
            "name": "echo",
            "description": "回显输入",
            "parameters": [type("P", (), {"name": "text", "type": "string", "description": "文本", "required": True, "default": None, "enum": None})()],
            "schema": lambda self=None: type("S", (), {
                "name": "echo", "description": "回显输入",
                "parameters": [type("P", (), {"name": "text", "type": "string", "description": "文本", "required": True, "default": None, "enum": None})()]
            })()
        })()
    ]]
    # 上面太绕,直接用 schema 转换示例跳过
    print("(已支持原生 function calling)")