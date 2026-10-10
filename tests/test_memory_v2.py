"""冒烟测试:新记忆系统(内存 / SQLite / Qdrant / Neo4j + 统一嵌入)

直接 python -m tests.test_memory_v2 即可
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# 保证相对导入稳定
HERE = Path(__file__).resolve().parent
AGENT_ROOT = HERE.parent
sys.path.insert(0, str(AGENT_ROOT))


def section(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def main() -> None:
    section("1. 内存后端默认可用性")
    from memory import MemoryManager, MemoryConfig
    from memory.config import DocumentBackendConfig
    cfg = MemoryConfig(
        vector_backend="memory",
        graph_backend="memory",
        document_backend="memory",
        embedding_prefer="tfidf",
    )
    mm = MemoryManager(user_id="smoke", config=cfg)
    mm.add_working("用户问 Python 函数", importance=0.5)
    mm.add_episodic("2026-09-27 user submitted a release plan", importance=0.8, event_type="release")
    mm.add_semantic(
        "Python 是一种解释型、面向对象的编程语言",
        importance=0.9, concept="language", domain="tech",
    )
    mm.add_perceptual("用户上传了一张警情截图", importance=0.6, modality="image", source="/tmp/case.png")
    print("→ stats:", json.dumps(mm.stats(), ensure_ascii=False, indent=2, default=str))
    res = mm.search("Python", limit=3)
    print("→ search 'Python':", [it.memory_type for it in res])

    section("2. SQLite 文档后端")
    cfg_sqlite = MemoryConfig(
        vector_backend="memory",
        graph_backend="memory",
        document_backend="sqlite",
        embedding_prefer="tfidf",
        document=DocumentBackendConfig(sqlite_path=str(AGENT_ROOT / "data" / "memory" / "_smoke_memory.db")),
    )
    (AGENT_ROOT / "data" / "memory").mkdir(parents=True, exist_ok=True)
    mm2 = MemoryManager(user_id="smoke", config=cfg_sqlite)
    mm2.add_episodic("夜间系统巡检记录", importance=0.7, event_type="ops", session_id="S-001")
    mm2.add_episodic("高峰流量调度记录", importance=0.7, event_type="traffic", session_id="S-002")
    print("→ doc_store.count:", mm2.doc_store.count())
    print("→ by_session(S-001):", [
        (it.content[:30], it.metadata.get("event_type")) for it in mm2.memory_types["episodic"].by_session("S-001")
    ])

    section("3. MemoryTool - 统一 execute() 接口")
    from tools import MemoryTool, ToolRegistry
    mt = MemoryTool(user_id="smoke")
    reg = ToolRegistry()
    reg.register_tool(mt)
    print(reg.execute_tool("memory", action="add",
                           content="张三是一名 Python 开发者", memory_type="semantic", importance=0.8))
    print(reg.execute_tool("memory", action="add",
                           content="李四是前端工程师,擅长 React", memory_type="semantic", importance=0.7))
    print(reg.execute_tool("memory", action="search", query="前端", limit=2))
    print(reg.execute_tool("memory", action="stats"))
    print(reg.execute_tool("memory", action="summary", query="Python"))

    section("4. 嵌入 fallback 链(DashScope → Local → TFIDF)")
    from memory.embedding import FallbackEmbedding, TFIDFEmbedding
    emb = FallbackEmbedding([TFIDFEmbedding()])  # 不依赖外部
    emb.warmup(["hello world", "Python programming", "release plan"])
    v = emb.encode(["hello", "release"])
    print(f"→ TFIDF dim={emb.dimension}, sample vec[:5]={v[0][:5]}")

    section("5. 语义记忆 - 实体抽取 + 图存储")
    cfg_sem = MemoryConfig(embedding_prefer="tfidf")
    mm3 = MemoryManager(user_id="smoke", config=cfg_sem)
    mm3.add_semantic(
        "项目 A 发布了新版本,团队部署了 10 个实例",
        importance=0.9, domain="release", entities=["项目 A", "新版本", "团队"],
        relations=[("团队", "DEPLOYED", "新版本")],
    )
    mm3.add_semantic(
        "项目 A 再次发布新版本,部署 12 个实例",
        importance=0.8, domain="release", entities=["项目 A", "新版本", "实例"],
    )
    items = mm3.memory_types["semantic"].search("版本 发布", limit=3)
    for it in items:
        print(f"  - [imp={it.importance:.2f}] {it.content} | entities={it.metadata.get('entities')}")
    print("→ related_entities('项目 A'):",
          mm3.memory_types["semantic"].related_entities("项目 A"))

    section("6. consolidate + forget")
    mm3.add_working("用户输入: 帮我规划 release", importance=0.9)
    mm3.add_working("用户输入: 好的,确认", importance=0.3)
    promoted = mm3.consolidate(from_type="working", to_type="episodic", importance_threshold=0.7)
    print(f"→ consolidate promoted={promoted}")
    forgotten = mm3.forget(strategy="importance_based", threshold=0.4)
    print(f"→ forget (imp<0.4) ={forgotten}")

    section("✅ 全部冒烟测试通过")


if __name__ == "__main__":
    main()
