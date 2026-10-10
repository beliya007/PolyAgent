"""
工具链 (ToolChain / ToolChainManager)

支持多个工具按顺序执行的链式调用,每一步可引用上一步输出。
借鉴 polyagent 第七章 7.5.4(1) 的设计,并做了工程化增强:
  - ChainStep 支持 input_template 变量替换
  - on_step 回调可插入日志 / 监控 / 短路
  - 返回 ToolResult 而非裸字符串,便于上层判断
  - ToolChainManager.register 返回 self,便于链式注册
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from tools.base import BaseTool, ToolResult
from tools.registry import ToolRegistry


@dataclass
class ChainStep:
    """链中的一个执行步骤

    Attributes:
        tool_name:  要调用的工具名
        input_template: 输入模板,支持 {var} 替换;内置 "input" 变量=初始输入
        output_key: 该步骤输出存入 context 的键名,默认 step_{idx}_result
        name: 可读步骤名,默认 step_{idx}
        description: 步骤描述,用于日志
    """
    tool_name: str
    input_template: str
    output_key: Optional[str] = None
    name: Optional[str] = None
    description: str = ""

    def __post_init__(self):
        if self.output_key is None:
            # 给一个稳定但人类可读的默认值
            self.output_key = f"step_{self.name or self.tool_name}_result"


StepHook = Callable[[int, ChainStep, ToolResult], None]


class ToolChain:
    """顺序执行的工具链(同步)

    使用示例::

        chain = ToolChain(name="research_and_calc", description="search -> calc -> summarize")
        chain.add_step("search", "{input}", output_key="search_result")
        chain.add_step("calculator", "estimate:{search_result}", output_key="calc_result")
        result = chain.execute(registry, initial_input="quarterly revenue")
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        steps: Optional[Sequence[ChainStep]] = None,
    ):
        if not name:
            raise ValueError("ToolChain.name 不能为空")
        self.name = name
        self.description = description
        self.steps: List[ChainStep] = list(steps or [])

    # ---------- 构造 ----------
    def add_step(
        self,
        tool_name: str,
        input_template: str,
        output_key: Optional[str] = None,
        name: Optional[str] = None,
        description: str = "",
    ) -> "ToolChain":
        """链式追加步骤,返回 self 便于连续 .add_step()"""
        step = ChainStep(
            tool_name=tool_name,
            input_template=input_template,
            output_key=output_key,
            name=name or f"step_{len(self.steps) + 1}",
            description=description,
        )
        self.steps.append(step)
        return self

    def __len__(self) -> int:
        return len(self.steps)

    # ---------- 执行 ----------
    def execute(
        self,
        registry: ToolRegistry,
        initial_input: Any = "",
        context: Optional[Dict[str, Any]] = None,
        on_step: Optional[StepHook] = None,
        stop_on_error: bool = True,
    ) -> ToolResult:
        """同步顺序执行整条链

        Args:
            registry:      工具注册表
            initial_input: 起点输入,会自动放入 context["input"]
            context:       预置 context 变量(可被模板引用)
            on_step:       每步执行后的回调 (idx, step, ToolResult)
            stop_on_error: 任一步失败是否立即终止

        Returns:
            ToolResult: data 为最后一步的字符串结果;success=False 表示中途失败
        """
        ctx: Dict[str, Any] = dict(context or {})
        ctx["input"] = initial_input

        print(f"🔗 开始执行工具链 '{self.name}' (共 {len(self.steps)} 步)")
        last: ToolResult = ToolResult.ok(data="", tool_name=self.name)

        for i, step in enumerate(self.steps, 1):
            # 1. 渲染模板
            try:
                tool_input = step.input_template.format(**ctx)
            except KeyError as e:
                last = ToolResult.fail(
                    error=f"step{i} 模板变量 {e} 未在 context 中找到",
                    tool_name=self.name,
                    metadata={"failed_step": step.name, "step_idx": i},
                )
                print(f"  step {i} [{step.tool_name}] ❌ {last.error}")
                return last

            # 2. 调工具(透过 registry 的统一异常捕获)
            step_result = registry.execute_tool_with_result(step.tool_name, tool_input)
            step_result.metadata.setdefault("step_name", step.name)
            step_result.metadata.setdefault("step_idx", i)
            ctx[step.output_key] = step_result.data
            last = step_result

            tag = "✅" if step_result.success else "❌"
            preview = (str(step_result.data)[:60] + "…") if (
                step_result.success and isinstance(step_result.data, str) and len(step_result.data) > 60
            ) else str(step_result.data)
            print(f"  step {i} [{step.tool_name}] {tag} {preview}")

            # 3. 回调
            if on_step is not None:
                try:
                    on_step(i, step, step_result)
                except Exception as e:  # 回调异常不影响主流程
                    print(f"  ⚠️ step {i} hook 异常: {e}")

            if not step_result.success and stop_on_error:
                return last

        print(f"✅ 工具链 '{self.name}' 执行完成")
        return last


class ToolChainManager:
    """多链统一管理器,可按名执行"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.chains: Dict[str, ToolChain] = {}

    def register(self, chain: ToolChain) -> "ToolChainManager":
        """注册一条链;同名会被覆盖"""
        if chain.name in self.chains:
            print(f"⚠️ 工具链 {chain.name!r} 已存在,将被覆盖")
        self.chains[chain.name] = chain
        print(f"🔗 工具链 {chain.name!r} 已注册")
        return self

    def unregister(self, name: str) -> bool:
        return self.chains.pop(name, None) is not None

    def get(self, name: str) -> Optional[ToolChain]:
        return self.chains.get(name)

    def list(self) -> List[str]:
        return list(self.chains.keys())

    def execute(
        self,
        chain_name: str,
        initial_input: Any = "",
        context: Optional[Dict[str, Any]] = None,
        on_step: Optional[StepHook] = None,
        stop_on_error: bool = True,
    ) -> ToolResult:
        chain = self.chains.get(chain_name)
        if chain is None:
            return ToolResult.fail(
                error=f"工具链 '{chain_name}' 未注册",
                tool_name="ToolChainManager",
            )
        return chain.execute(
            self.registry,
            initial_input=initial_input,
            context=context,
            on_step=on_step,
            stop_on_error=stop_on_error,
        )

    def __len__(self) -> int:
        return len(self.chains)

    def __contains__(self, name: str) -> bool:
        return name in self.chains
