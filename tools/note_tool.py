"""
笔记工具 (NoteTool) — polyagent 第九章 §9.4

设计要点:
- 每篇笔记是独立 .md 文件,YAML 前置元数据 + Markdown 正文
- 索引 notes_index.json 维护快速检索
- 不引入 pyyaml 依赖,内置一个最小 YAML 子集序列化(只处理 key: value)
- 7 个动作:create / read / update / search / list / summary / delete
- 与 ContextBuilder 通过 run({"action": "search", "query": ...}) 集成
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from tools.base import BaseTool, ToolParameter


# =====================================================================
#  极简 YAML —— 只支持 key: value 与 list-of-scalars,够 NoteTool 用
# =====================================================================
def _yaml_dump(meta: Dict[str, Any]) -> str:
    """把一层 dict 序列化为 YAML 片段"""
    lines: List[str] = []
    for k, v in meta.items():
        if isinstance(v, list):
            if not v:
                lines.append(f"{k}: []")
            else:
                lines.append(f"{k}:")
                for item in v:
                    lines.append(f"  - {item}")
        elif isinstance(v, dict):
            lines.append(f"{k}:")
            for kk, vv in v.items():
                lines.append(f"  {kk}: {vv}")
        elif v is None:
            lines.append(f"{k}: null")
        elif isinstance(v, bool):
            lines.append(f"{k}: {'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            lines.append(f"{k}: {v}")
        else:
            s = str(v).replace('"', '\\"')
            lines.append(f'{k}: "{s}"')
    return "\n".join(lines)


def _yaml_load(blob: str) -> Dict[str, Any]:
    """解析 _yaml_dump 输出的格式(顶层 key: value / key:\n  - item)"""
    meta: Dict[str, Any] = {}
    current_key: Optional[str] = None
    for raw in blob.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if line.startswith("  - "):
            if current_key is not None:
                meta.setdefault(current_key, []).append(line[4:].strip())
            continue
        if line.startswith("  ") and current_key is not None:
            # 嵌套 dict(暂不支持多级,落到 list 里)
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value == "":
            current_key = key
            meta[key] = []
        elif value == "[]":
            meta[key] = []
            current_key = None
        elif value == "null":
            meta[key] = None
            current_key = None
        elif value == "true":
            meta[key] = True
            current_key = None
        elif value == "false":
            meta[key] = False
            current_key = None
        else:
            current_key = None
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1].replace('\\"', '"')
            # 数字
            try:
                if "." in value:
                    meta[key] = float(value)
                else:
                    meta[key] = int(value)
            except ValueError:
                meta[key] = value
    return meta


VALID_TYPES = {"task_state", "conclusion", "blocker", "action", "reference", "general"}


# =====================================================================
#  NoteTool
# =====================================================================
class NoteTool(BaseTool):
    """结构化笔记工具 — Markdown + YAML 元数据 + 索引

    使用::

        notes = NoteTool(workspace="./notes", project_name="demo")
        notes.run({"action": "create", "title": "...", "content": "...",
                   "note_type": "blocker", "tags": ["urgent"]})
        results = notes.run({"action": "search", "query": "..."})
    """

    name = "note"
    description = (
        "结构化笔记工具 — 长期项目追踪。用 Markdown+YAML 存储任务状态、结论、阻塞点、行动项。"
        "支持 create/read/update/search/list/summary/delete 七种操作。"
    )
    parameters = [
        ToolParameter("action", "string",
                      "操作类型", required=True,
                      enum=["create", "read", "update", "search", "list",
                            "summary", "delete"]),
        ToolParameter("title", "string", "笔记标题(create 用)", required=False),
        ToolParameter("content", "string", "Markdown 正文(create/update 用)", required=False),
        ToolParameter("note_type", "string",
                      "笔记类型(create/update 用)", required=False, default="general",
                      enum=sorted(VALID_TYPES)),
        ToolParameter("tags", "array", "标签列表(create/update 用)", required=False),
        ToolParameter("query", "string", "搜索关键词(search 用)", required=False),
        ToolParameter("limit", "integer", "返回数量(默认 10)", required=False, default=10),
        ToolParameter("note_id", "string", "笔记 ID(read/update/delete 用)", required=False),
        ToolParameter("note_type_filter", "string", "按类型过滤(list 用)", required=False),
        ToolParameter("tags_filter", "array", "按标签过滤(任一命中,list/search 用)", required=False),
    ]

    def __init__(self, workspace: str = "./notes", project_name: str = "default"):
        super().__init__()
        self.workspace = os.path.abspath(workspace)
        self.project_name = project_name
        os.makedirs(self.workspace, exist_ok=True)
        self.index_path = os.path.join(self.workspace, "notes_index.json")
        self.index: Dict[str, Dict[str, Any]] = self._load_index()

    # ---------- 入口 ----------
    def execute(self, **kwargs) -> str:
        action = kwargs.get("action")
        if not action:
            raise ValueError("action 必填")
        handler = {
            "create": self._create,
            "read": self._read,
            "update": self._update,
            "search": self._search,
            "list": self._list,
            "summary": self._summary,
            "delete": self._delete,
        }.get(action)
        if handler is None:
            raise ValueError(f"未知 action: {action}")
        try:
            data = handler(**kwargs)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)
        return json.dumps({"success": True, "data": data}, ensure_ascii=False, default=str)

    # ---------- 索引 ----------
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_index(self) -> None:
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)

    def _build_markdown(self, meta: Dict[str, Any], content: str) -> str:
        return f"---\n{_yaml_dump(meta)}\n---\n\n{content.strip()}\n"

    def _parse_markdown(self, raw: str) -> tuple[Dict[str, Any], str]:
        parts = raw.split("---\n", 2)
        if len(parts) >= 3:
            try:
                meta = _yaml_load(parts[1])
            except Exception:
                meta = {}
            return meta, parts[2].strip()
        return {}, raw.strip()

    # ---------- create ----------
    def _create(self, title: str, content: str,
                note_type: str = "general", tags: Optional[List[str]] = None,
                **_) -> Dict[str, Any]:
        if not title or not content:
            raise ValueError("title / content 必填")
        if note_type not in VALID_TYPES:
            raise ValueError(f"note_type 必须 ∈ {VALID_TYPES}")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        note_id = f"note_{ts}_{len(self.index)}"
        meta = {
            "id": note_id,
            "title": title,
            "type": note_type,
            "tags": list(tags or []),
            "project": self.project_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        path = os.path.join(self.workspace, f"{note_id}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._build_markdown(meta, content))
        meta["file_path"] = path
        self.index[note_id] = meta
        self._save_index()
        return {"note_id": note_id, "title": title, "type": note_type}

    # ---------- read ----------
    def _read(self, note_id: str, **_) -> Dict[str, Any]:
        if note_id not in self.index:
            raise ValueError(f"笔记不存在: {note_id}")
        path = self.index[note_id]["file_path"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"笔记文件丢失: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        meta, content = self._parse_markdown(raw)
        return {"metadata": meta, "content": content}

    # ---------- update ----------
    def _update(self, note_id: str, title: Optional[str] = None,
                content: Optional[str] = None, note_type: Optional[str] = None,
                tags: Optional[List[str]] = None, **_) -> Dict[str, Any]:
        if note_id not in self.index:
            raise ValueError(f"笔记不存在: {note_id}")
        if note_type and note_type not in VALID_TYPES:
            raise ValueError(f"note_type 必须 ∈ {VALID_TYPES}")
        path = self.index[note_id]["file_path"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"笔记文件丢失: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        meta, old_content = self._parse_markdown(raw)
        if title:
            meta["title"] = title
        if note_type:
            meta["type"] = note_type
        if tags is not None:
            meta["tags"] = list(tags)
        if content is not None:
            old_content = content
        meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._build_markdown(meta, old_content))
        self.index[note_id].update(meta)
        self._save_index()
        return {"note_id": note_id, "title": meta["title"], "type": meta["type"]}

    # ---------- search ----------
    def _search(self, query: str, limit: int = 10,
                note_type_filter: Optional[str] = None,
                tags_filter: Optional[List[str]] = None, **_) -> List[Dict[str, Any]]:
        if not query:
            return []
        results: List[Dict[str, Any]] = []
        q = query.lower()
        for note_id, meta in self.index.items():
            if note_type_filter and meta.get("type") != note_type_filter:
                continue
            if tags_filter:
                note_tags = set(meta.get("tags", []))
                if not note_tags.intersection(tags_filter):
                    continue
            title = (meta.get("title") or "").lower()
            if q in title:
                results.append(self._note_dict(note_id, meta))
                continue
            try:
                content = self._read(note_id=note_id)["content"].lower()
            except Exception:
                continue
            if q in content:
                results.append(self._note_dict(note_id, meta))
        results.sort(key=lambda x: x.get("updated_at") or "", reverse=True)
        return results[:limit]

    # ---------- list ----------
    def _list(self, limit: int = 20, note_type_filter: Optional[str] = None,
              tags_filter: Optional[List[str]] = None, **_) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for note_id, meta in self.index.items():
            if note_type_filter and meta.get("type") != note_type_filter:
                continue
            if tags_filter:
                note_tags = set(meta.get("tags", []))
                if not note_tags.intersection(tags_filter):
                    continue
            results.append(self._note_dict(note_id, meta))
        results.sort(key=lambda x: x.get("updated_at") or "", reverse=True)
        return results[:limit]

    def _note_dict(self, note_id: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "note_id": note_id,
            "title": meta.get("title", ""),
            "type": meta.get("type"),
            "tags": meta.get("tags", []),
            "updated_at": meta.get("updated_at"),
            "project": meta.get("project"),
            "content": meta.get("_cached_content") or "",  # 读取时不展开正文
        }

    # ---------- summary ----------
    def _summary(self, **_) -> Dict[str, Any]:
        total = len(self.index)
        type_counts: Dict[str, int] = {}
        for m in self.index.values():
            t = m.get("type", "general")
            type_counts[t] = type_counts.get(t, 0) + 1
        recent = sorted(
            self.index.values(),
            key=lambda x: x.get("updated_at") or "",
            reverse=True,
        )[:5]
        return {
            "total_notes": total,
            "type_distribution": type_counts,
            "recent_notes": [
                {
                    "id": m.get("id"),
                    "title": m.get("title", ""),
                    "type": m.get("type"),
                    "updated_at": m.get("updated_at"),
                }
                for m in recent
            ],
        }

    # ---------- delete ----------
    def _delete(self, note_id: str, **_) -> Dict[str, Any]:
        if note_id not in self.index:
            raise ValueError(f"笔记不存在: {note_id}")
        path = self.index[note_id].get("file_path")
        if path and os.path.exists(path):
            os.remove(path)
        title = self.index[note_id].get("title", note_id)
        del self.index[note_id]
        self._save_index()
        return {"note_id": note_id, "title": title, "deleted": True}
