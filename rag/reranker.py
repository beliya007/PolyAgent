"""
RAG 重排序 (Reranker)

hello-agents 第八章:
- 用 LLM 对初筛结果做相关性重排序
- 提升检索精度
"""
from __future__ import annotations

from typing import Any, Dict, List

from llm_client import HelloAgentsLLM


RERANK_PROMPT = """你是相关性评估专家,请对候选文档与查询的相关性打分(0~10)。

# 查询
{query}

# 候选文档(已编号)
{candidates}

请输出严格的 JSON 数组,每个元素为 `{{"id": <编号>, "score": <0~10>}}`,按相关性从高到低排序。
仅输出 JSON,不要其他文字。
"""


class LLMReranker:
    """基于 LLM 的重排序器"""

    def __init__(self, llm: HelloAgentsLLM, enabled: bool = True):
        self.llm = llm
        self.enabled = enabled

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """对候选重排序,返回 top_k"""
        if not self.enabled or len(candidates) <= 1:
            return candidates[:top_k]

        # 构造候选展示
        lines = []
        for i, c in enumerate(candidates):
            content = c.get("content", "")[:300]
            lines.append(f"({i}) {content}")
        prompt = RERANK_PROMPT.format(query=query, candidates="\n".join(lines))
        raw = self.llm.invoke([{"role": "user", "content": prompt}]) or ""

        import json
        import re
        m = re.search(r"\[[\s\S]*\]", raw)
        if not m:
            return candidates[:top_k]
        try:
            data = json.loads(m.group(0))
            scored = []
            for item in data:
                idx = int(item.get("id", -1))
                score = float(item.get("score", 0))
                if 0 <= idx < len(candidates):
                    scored.append((idx, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [candidates[i] for i, _ in scored[:top_k]]
        except Exception:
            return candidates[:top_k]