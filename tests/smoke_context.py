"""smoke test — 验证 GSSC + NoteTool + TerminalTool 端到端可用"""
import sys
import os
import shutil
import tempfile

# 把 agent 根目录加到 sys.path(当前文件在 agent/tests/ 下)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

from context import ContextBuilder, ContextConfig, ContextPacket, ContextCompressor
from tools import NoteTool, TerminalTool


def main():
    work = tempfile.mkdtemp(prefix="ctx_smoke_")
    print(f"=== smoke test workdir: {work} ===")

    # ---------- 准备一些素材 ----------
    (f"{work}/project").replace  # noop
    proj = os.path.join(work, "project")
    os.makedirs(proj, exist_ok=True)
    with open(os.path.join(proj, "README.md"), "w", encoding="utf-8") as f:
        f.write("# 重构项目\n\n## 第一阶段\n\n完成了数据模型层重构\n\n## 第二阶段\n\n业务逻辑层正在重构中,遇到依赖冲突")
    with open(os.path.join(proj, "notes.md"), "w", encoding="utf-8") as f:
        f.write("- 阻塞:依赖版本冲突\n- 计划:升级 pip 后重装\n")

    # ---------- NoteTool ----------
    print("\n[1] NoteTool 测试")
    note = NoteTool(workspace=os.path.join(work, "notes"), project_name="refactor")
    note.run({"action": "create", "title": "依赖冲突", "content": "业务逻辑层有 3 个包冲突",
              "note_type": "blocker", "tags": ["urgent", "phase2"]})
    note.run({"action": "create", "title": "下一步", "content": "升级 pip 并锁定版本",
              "note_type": "action", "tags": ["phase2"]})
    note.run({"action": "create", "title": "结论", "content": "Pandas 在数据处理方面非常强大",
              "note_type": "conclusion", "tags": []})

    search_result = note.run({"action": "search", "query": "依赖", "limit": 5})
    print(f"search '依赖' -> {search_result[:200]}")
    summary = note.run({"action": "summary"})
    print(f"summary -> {summary[:200]}")

    # ---------- TerminalTool ----------
    print("\n[2] TerminalTool 测试")
    term = TerminalTool(workspace=proj)
    print(term.run({"command": "ls"}))
    print("---")
    print(term.run({"command": "cat README.md", "limit": 5}))
    print("---")
    print(term.run({"command": "grep 依赖", "limit": 10}))
    print("---")
    print(term.run({"command": "quick_context 依赖", "limit": 3}))

    # 越界测试
    out_of_root = term.run({"command": "cat C:/Windows/System32/drivers/etc/hosts"})
    print(f"\n越界测试 -> {out_of_root}")

    # ---------- ContextBuilder (无 RAG / 无 Memory,只注入 custom_packets) ----------
    print("\n[3] ContextBuilder GSSC 测试 (无 RAG/Memory)")
    cfg = ContextConfig(max_tokens=800, reserve_ratio=0.2,
                        min_relevance=0.05, recency_weight=0.3, relevance_weight=0.7)
    builder = ContextBuilder(
        config=cfg,
        memory_manager=None,
        rag_tool=None,
        note_tool=note,
        terminal_tool=term,
    )

    fake_history = [
        type("M", (), {"role": "user", "content": "我正在重构业务逻辑层",
                       "timestamp": datetime.now()})(),
        type("M", (), {"role": "assistant", "content": "好的,需要我帮你做什么?",
                       "timestamp": datetime.now()})(),
    ]

    ctx = builder.build(
        user_query="依赖冲突",
        conversation_history=fake_history,
        system_instructions="你是资深 Python 数据工程顾问,回答需有具体建议。",
        include_memory=False,
        include_rag=False,
        include_history=True,
        include_notes=True,
        include_terminal=True,
    )
    print("\n=== 生成的上下文 ===")
    print(ctx)

    # ---------- 校验结构 ----------
    print("\n[4] 校验分区结构")
    required_sections = ["[Role & Policies]", "[Task]", "[State]", "[Output]"]
    # Evidence/Context 可能为空被省略,允许缺
    for sec in required_sections:
        assert sec in ctx, f"缺少必要分区: {sec}"
        print(f"  ✓ {sec}")
    # ContextBuilder._estimate_tokens_local 这个 helper 用来 debug
    print(f"  ✓ Context 长度={len(ctx)} chars")

    # ---------- Compressor 单独测试 ----------
    print("\n[5] ContextCompressor 测试")
    cmp = ContextCompressor(max_tokens=80)
    sections = [
        "[Role & Policies]\n" + "你是一个助手的角色描述。" * 20,
        "[Task]\n" + "完成某项复杂任务。" * 10,
        "[Evidence]\n" + "大量证据数据" * 30,
    ]
    compressed = cmp.partition_compress(sections, max_tokens=80)
    print(compressed)
    assert "[... 内容已压缩 ...]" in compressed, "压缩后应包含截断标记"

    print("\n=== ALL OK ===")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
