"""
评估模块 (Evaluation)

基于 polyagent 第十二章:
- 回答质量评估
- 工具调用成功率
- 执行效率评估
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llm_client import PolyAgentLLM


@dataclass
class EvaluationResult:
    """单次评估结果"""
    query: str
    answer: str
    intent: str
    paradigm: str
    # 各维度评分
    relevance: float       # 0~1, 回答与问题相关性
    completeness: float    # 0~1, 信息完整性
    accuracy: float       # 0~1, 事实准确性(依赖规则/工具)
    efficiency: float      # 0~1, 执行效率(步骤数/时间)
    overall: float        # 综合评分
    tool_call_success: float  # 工具调用成功率
    execution_time: float
    metadata: Dict[str, Any]


EVALUATE_PROMPT = """你是质量评估专家,请对以下 Agent 回答打分。

# 问题
{query}

# 回答
{answer}

# 维度
1. relevance (相关性): 回答是否切题?(0.0~1.0)
2. completeness (完整性): 信息是否完整?(0.0~1.0)
3. accuracy (准确性): 事实是否准确?(0.0~1.0)

请输出严格 JSON:
{{
  "relevance": <0.0~1.0>,
  "completeness": <0.0~1.0>,
  "accuracy": <0.0~1.0>
}}

仅输出 JSON。
"""


class Evaluator:
    """评估器"""

    def __init__(self, llm: PolyAgentLLM):
        self.llm = llm

    def evaluate(
        self,
        query: str,
        answer: str,
        intent: str = "",
        paradigm: str = "",
        tool_calls: Optional[List[Dict]] = None,
        execution_time: float = 0.0,
    ) -> EvaluationResult:
        """
        评估 Agent 回答质量。
        包含 LLM 评估(relevance/completeness/accuracy) + 规则评估(efficiency/tool_call_success)。
        """
        # LLM 评估
        prompt = EVALUATE_PROMPT.format(query=query, answer=answer)
        raw = self.llm.invoke([{"role": "user", "content": prompt}]) or ""

        import json
        import re
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                data = json.loads(m.group(0))
                relevance = float(data.get("relevance", 0.7))
                completeness = float(data.get("completeness", 0.7))
                accuracy = float(data.get("accuracy", 0.7))
            except Exception:
                relevance = completeness = accuracy = 0.7
        else:
            relevance = completeness = accuracy = 0.7

        # 规则评估
        # 效率:基于执行时间和步骤数
        if execution_time > 0:
            # 10s 以内满分,超过 60s 最低
            efficiency = max(0.0, 1.0 - (execution_time - 10) / 50)
        else:
            efficiency = 0.7

        # 工具调用成功率
        if tool_calls:
            successes = sum(1 for tc in tool_calls if tc.get("success", True))
            tool_call_success = successes / len(tool_calls)
        else:
            tool_call_success = 1.0

        overall = (relevance * 0.3 + completeness * 0.3 + accuracy * 0.2 + efficiency * 0.1 + tool_call_success * 0.1)

        return EvaluationResult(
            query=query,
            answer=answer,
            intent=intent,
            paradigm=paradigm,
            relevance=relevance,
            completeness=completeness,
            accuracy=accuracy,
            efficiency=efficiency,
            overall=overall,
            tool_call_success=tool_call_success,
            execution_time=execution_time,
            metadata={
                "llm_eval_raw": raw,
            },
        )

    def batch_evaluate(
        self,
        cases: List[Dict[str, Any]],
    ) -> List[EvaluationResult]:
        """批量评估"""
        return [
            self.evaluate(
                query=c.get("query", ""),
                answer=c.get("answer", ""),
                intent=c.get("intent", ""),
                paradigm=c.get("paradigm", ""),
                tool_calls=c.get("tool_calls"),
                execution_time=c.get("execution_time", 0.0),
            )
            for c in cases
        ]

    def summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """汇总统计"""
        if not results:
            return {}
        n = len(results)
        return {
            "count": n,
            "avg_relevance": sum(r.relevance for r in results) / n,
            "avg_completeness": sum(r.completeness for r in results) / n,
            "avg_accuracy": sum(r.accuracy for r in results) / n,
            "avg_efficiency": sum(r.efficiency for r in results) / n,
            "avg_overall": sum(r.overall for r in results) / n,
            "avg_tool_success": sum(r.tool_call_success for r in results) / n,
            "avg_execution_time": sum(r.execution_time for r in results) / n,
        }