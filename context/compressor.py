"""
上下文压缩器 (ContextCompressor)

polyagent 第九章 §9.3.3 Compress 阶段:
- 兜底压缩超 token 预算的上下文
- 支持两种策略:
    1. partition_compress: 按 [Role]/[Task]/[Evidence]/... 分区截断,保结构
    2. summarize_messages: 对超长对话历史做 LLM 摘要(单独 API,不直接被 Builder 调用)
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from llm_client import PolyAgentLLM


# 粗估:中文 1.5 字符/token,英文 4 字符/token(与 builder 保持一致)
def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cn = sum(1 for c in text if "一" <= c <= "鿿")
    other = len(text) - cn
    return int(cn / 1.5 + other / 4) + 1


COMPRESS_PROMPT = """请将以下对话历史压缩为精简摘要,保留关键信息:

{history}

请输出 JSON 格式:
{{
  "summary": "<压缩后的摘要,200字以内>",
  "key_points": ["<关键点1>", "<关键点2>"]
}}

仅输出 JSON。
"""

TRUNCATED_TAG = "\n[... 内容已压缩 ...]"


class ContextCompressor:
    """上下文压缩器

    使用方式:
        compressor = ContextCompressor(llm=..., max_tokens=2000)

        # 策略1:按分区截断(被 ContextBuilder 在 budget 溢出时调用)
        context = compressor.partition_compress(sections, max_tokens=2000)

        # 策略2:对长对话历史做 LLM 摘要(独立 API)
        messages = compressor.summarize_messages(messages)
    """

    def __init__(
        self,
        llm: Optional[PolyAgentLLM] = None,
        max_tokens: int = 2000,
        truncate_floor: int = 50,
    ):
        self.llm = llm
        self.max_tokens = max_tokens
        self.truncate_floor = truncate_floor  # 截断时最少保留多少 token

    # ==================================================================
    #  向后兼容:旧 API compress() 等价于 summarize_messages()
    # ==================================================================
    def compress(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """向后兼容旧版 API;实际调用 summarize_messages()"""
        return self.summarize_messages(messages)

    # ==================================================================
    #  策略1:按分区截断 —— 文档 §9.3.3 Compress 的实现
    # ==================================================================
    def partition_compress(self, sections: List[str], max_tokens: Optional[int] = None) -> str:
        """按分区压缩超限上下文,保持结构完整性

        Args:
            sections: 候选分区字符串列表(每个是一个完整的 [Section] 块)
            max_tokens: 压缩目标;若为 None 使用 self.max_tokens

        Returns:
            压缩后的拼接字符串
        """
        budget = max_tokens if max_tokens is not None else self.max_tokens
        compressed: List[str] = []
        current_total = 0

        for section in sections:
            section_tokens = _estimate_tokens(section)

            if current_total + section_tokens <= budget:
                # 完整保留
                compressed.append(section)
                current_total += section_tokens
                continue

            # 超出预算,部分保留
            remaining = budget - current_total
            if remaining > self.truncate_floor:
                truncated = self._truncate_text(section, remaining - _estimate_tokens(TRUNCATED_TAG))
                compressed.append(truncated + TRUNCATED_TAG)
                current_total = budget
            break

        return "\n\n".join(compressed)

    @staticmethod
    def _truncate_text(text: str, max_tokens: int) -> str:
        """按 token 预算等比例截断(粗估字符/token)"""
        if max_tokens <= 0 or not text:
            return ""
        total_tokens = _estimate_tokens(text)
        if total_tokens <= max_tokens:
            return text
        # 按字符/token 比估出最大字符数
        char_per_token = len(text) / total_tokens
        max_chars = int(max_tokens * char_per_token)
        return text[:max_chars]

    # ==================================================================
    #  策略2:对长对话历史做 LLM 摘要
    # ==================================================================
    def summarize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """压缩超长对话历史,返回精简版本

        策略:
            - 始终保留 system 消息
            - 若去掉早期消息后 token 已经够用,直接丢弃早期消息
            - 否则对早期消息调 LLM 生成摘要(需要 self.llm),失败回退到首条消息截断
        """
        total = sum(_estimate_tokens(m.get("content", "")) for m in messages)
        if total <= self.max_tokens:
            return messages

        system = [m for m in messages if m.get("role") == "system"]
        others = [m for m in messages if m.get("role") != "system"]

        system_tokens = sum(_estimate_tokens(m.get("content", "")) for m in system)
        budget = self.max_tokens - system_tokens
        others_tokens = sum(_estimate_tokens(m.get("content", "")) for m in others)
        if others_tokens <= budget:
            return system + others

        # 超出预算:对早期消息做摘要,保留最近的消息
        kept: List[Dict[str, Any]] = []
        accumulated = 0
        to_compress: List[Dict[str, Any]] = []
        for m in others:
            t = _estimate_tokens(m.get("content", ""))
            if accumulated + t <= budget:
                kept.append(m)
                accumulated += t
            else:
                to_compress.append(m)

        if not to_compress:
            return system + kept

        summary_msg = self._build_summary_message(to_compress)
        return system + [summary_msg] + kept

    def _build_summary_message(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """对 messages 做摘要(LLM 优先,失败回退)"""
        if self.llm is None:
            return self._fallback_summary(messages)

        history_text = "\n".join(
            f"[{m.get('role')}] {m.get('content', '')[:300]}" for m in messages
        )
        prompt = COMPRESS_PROMPT.format(history=history_text)
        try:
            raw = self.llm.invoke([{"role": "user", "content": prompt}]) or ""
        except Exception:
            return self._fallback_summary(messages)

        m2 = re.search(r"\{[\s\S]*\}", raw)
        if m2:
            try:
                data = json.loads(m2.group(0))
                summary = data.get("summary", "")
                key_points = data.get("key_points", [])
                return {
                    "role": "system",
                    "content": f"[历史摘要] {summary}\n关键点: {', '.join(key_points)}",
                }
            except Exception:
                pass

        return self._fallback_summary(messages)

    @staticmethod
    def _fallback_summary(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """无 LLM 或 LLM 解析失败时的兜底"""
        snippet = "; ".join(m.get("content", "")[:100] for m in messages[:5])
        return {
            "role": "system",
            "content": f"[历史摘要] {snippet}",
        }
