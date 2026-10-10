"""
SQLite 文档存储 (SQLiteDocumentStore)

hello-agents 第八章设计:
- 结构化元数据 + 全文内容
- SQL WHERE 过滤 / 排序 / 分页
- 标准库 sqlite3,零外部依赖
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from memory.storage.base import BaseDocumentStore, DocumentRecord


class SQLiteDocumentStore(BaseDocumentStore):
    backend_name = "sqlite"

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id          TEXT PRIMARY KEY,
                    content     TEXT NOT NULL,
                    metadata    TEXT NOT NULL DEFAULT '{}',
                    created_at  REAL DEFAULT (strftime('%s','now')),
                    updated_at  REAL DEFAULT (strftime('%s','now'))
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_docs_created_at ON documents(created_at)"
            )

    # ---------------- CRUD ----------------
    def upsert(self, doc: DocumentRecord) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO documents(id, content, metadata, updated_at)
                VALUES(?,?,?,strftime('%s','now'))
                ON CONFLICT(id) DO UPDATE SET
                    content=excluded.content,
                    metadata=excluded.metadata,
                    updated_at=strftime('%s','now')
                """,
                (doc.id, doc.content, json.dumps(doc.metadata, ensure_ascii=False)),
            )

    def upsert_batch(self, docs: List[DocumentRecord]) -> int:
        n = 0
        with self._lock, self._conn:
            for d in docs:
                self._conn.execute(
                    """
                    INSERT INTO documents(id, content, metadata, updated_at)
                    VALUES(?,?,?,strftime('%s','now'))
                    ON CONFLICT(id) DO UPDATE SET
                        content=excluded.content,
                        metadata=excluded.metadata,
                        updated_at=strftime('%s','now')
                    """,
                    (d.id, d.content, json.dumps(d.metadata, ensure_ascii=False)),
                )
                n += 1
        return n

    def get(self, doc_id: str) -> Optional[DocumentRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, content, metadata FROM documents WHERE id=?", (doc_id,)
            ).fetchone()
        if row is None:
            return None
        try:
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
        except Exception:
            meta = {}
        return DocumentRecord(id=row["id"], content=row["content"], metadata=meta)

    def delete(self, ids: List[str]) -> int:
        if not ids:
            return 0
        with self._lock, self._conn:
            cur = self._conn.executemany(
                "DELETE FROM documents WHERE id=?", [(i,) for i in ids]
            )
            return cur.rowcount

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()
        return int(row["c"] or 0)

    def query(
        self,
        where: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        order_by: str = "created_at DESC",
    ) -> List[DocumentRecord]:
        sql = "SELECT id, content, metadata FROM documents"
        clauses: List[str] = []
        params: List[Any] = []
        if where:
            for k, v in where.items():
                # 使用 json_extract 读 metadata 内的字段
                clauses.append("json_extract(metadata, '$.' || ?) = ?")
                params.extend([k, v])
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
        sql += f" ORDER BY {order_by} LIMIT ?"
        params.append(limit)

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        out: List[DocumentRecord] = []
        for row in rows:
            try:
                meta = json.loads(row["metadata"]) if row["metadata"] else {}
            except Exception:
                meta = {}
            out.append(DocumentRecord(id=row["id"], content=row["content"], metadata=meta))
        return out

    def stats(self) -> Dict[str, Any]:
        return {"backend": self.backend_name, "path": self.db_path, "count": self.count()}

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
