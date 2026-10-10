"""
记忆持久化模块

polyagent 第八章:
- 基于 JSONL 文件存储,每行一条 MemoryItem
- 支持按类型分文件、按用户 ID 分目录
- 启动时自动加载,运行中增量保存
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from memory.base import MemoryItem


class JSONLPersistence:
    """JSONL 持久化器"""

    def __init__(self, base_dir: str = "./data/memory", user_id: str = "default"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.user_id = user_id
        self.user_dir = self.base_dir / user_id
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _file_for(self, memory_type: str) -> Path:
        return self.user_dir / f"{memory_type}.jsonl"

    # ---------------- 保存 ----------------

    def save_items(self, memory_type: str, items: Iterable[MemoryItem]) -> None:
        """覆盖写入"""
        path = self._file_for(memory_type)
        with self._lock:
            with open(path, "w", encoding="utf-8") as f:
                for item in items:
                    f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

    def append_item(self, item: MemoryItem) -> None:
        """追加一条"""
        path = self._file_for(item.memory_type)
        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

    # ---------------- 加载 ----------------

    def load_items(self, memory_type: str) -> List[MemoryItem]:
        path = self._file_for(memory_type)
        if not path.exists():
            return []
        items: List[MemoryItem] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        items.append(MemoryItem.from_dict(d))
                    except Exception:
                        continue
        except Exception:
            pass
        return items

    def load_all(self) -> Dict[str, List[MemoryItem]]:
        """加载所有类型"""
        types = ["working", "episodic", "semantic", "perceptual"]
        return {t: self.load_items(t) for t in types if self._file_for(t).exists()}

    # ---------------- 清理 ----------------

    def clear(self, memory_type: Optional[str] = None) -> int:
        """清理指定类型或全部"""
        n = 0
        types = [memory_type] if memory_type else ["working", "episodic", "semantic", "perceptual"]
        for t in types:
            path = self._file_for(t)
            if path.exists():
                n += self._count_lines(path)
                path.unlink()
        return n

    @staticmethod
    def _count_lines(path: Path) -> int:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    def stats(self) -> Dict[str, int]:
        """统计各类型条数"""
        out: Dict[str, int] = {}
        if self.user_dir.exists():
            for p in self.user_dir.glob("*.jsonl"):
                out[p.stem] = self._count_lines(p)
        return out