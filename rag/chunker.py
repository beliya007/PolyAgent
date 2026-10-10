"""
文本分块器 (Chunker)

polyagent 第八章设计:
1. 基础 TextChunker - 按字符数切分,段落优先 + 滑窗重叠
2. MarkdownChunker    - 标题结构感知 + Token 估算(CJK + 英文)
   完整复现 polyagent _split_paragraphs_with_headings + _chunk_paragraphs
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Chunk:
    id: str
    content: str
    doc_id: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


# ============================================================
#  CJK / Token 工具
# ============================================================
def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF       # CJK Unified Ideographs
        or 0x3400 <= code <= 0x4DBF    # Extension A
        or 0x20000 <= code <= 0x2A6DF  # Extension B
        or 0x2A700 <= code <= 0x2B73F  # Extension C
        or 0x2B740 <= code <= 0x2B81F  # Extension D
        or 0x2B820 <= code <= 0x2CEAF  # Extension E
        or 0xF900 <= code <= 0xFAFF    # Compatibility Ideographs
    )


def _approx_token_len(text: str) -> int:
    """近似的 token 长度

    规则:
      - CJK 字符每个按 1 token 计
      - 非 CJK 文本按空白分词,每个词按 1 token 计
      - CJK 连续段不会被误算成一个 token(因为逐字符计数)
    """
    if not text:
        return 0
    cjk_count = 0
    non_cjk_buf: List[str] = []
    for ch in text:
        if _is_cjk(ch):
            cjk_count += 1
            # 累积的非 CJK 文本独立分词
            if non_cjk_buf:
                cjk_count += len([t for t in "".join(non_cjk_buf).split() if t])
                non_cjk_buf = []
        else:
            non_cjk_buf.append(ch)
    # 尾部非 CJK
    if non_cjk_buf:
        cjk_count += len([t for t in "".join(non_cjk_buf).split() if t])
    return cjk_count


# ============================================================
#  TextChunker - 字符级基础分块器
# ============================================================
class TextChunker:
    """基于字符数的简单分块器(段落优先 + 滑窗重叠)"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 80,
        separator: str = "\n",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def split(self, text: str, doc_id: str = "doc") -> List[Chunk]:
        text = text.strip()
        if not text:
            return []

        paragraphs = [p.strip() for p in text.split(self.separator) if p.strip()]
        chunks: List[Chunk] = []
        current = ""
        idx = 0

        for p in paragraphs:
            # 单段超长 → 硬切
            if len(p) > self.chunk_size:
                if current:
                    chunks.append(self._make_chunk(current, doc_id, idx))
                    idx += 1
                    current = ""
                for i in range(0, len(p), self.chunk_size - self.chunk_overlap):
                    pieces = p[i: i + self.chunk_size]
                    chunks.append(self._make_chunk(pieces, doc_id, idx))
                    idx += 1
                continue

            if not current:
                current = p
            elif len(current) + len(p) + 1 <= self.chunk_size:
                current = current + "\n" + p
            else:
                chunks.append(self._make_chunk(current, doc_id, idx))
                idx += 1
                overlap_text = (
                    current[-self.chunk_overlap:] if len(current) > self.chunk_overlap else current
                )
                current = (overlap_text + "\n" + p) if overlap_text else p

        if current:
            chunks.append(self._make_chunk(current, doc_id, idx))

        return chunks

    @staticmethod
    def _make_chunk(content: str, doc_id: str, idx: int) -> Chunk:
        return Chunk(
            id=f"{doc_id}_chunk_{idx}",
            content=content.strip(),
            doc_id=doc_id,
            chunk_index=idx,
        )


# ============================================================
#  MarkdownChunker - 标题感知 + Token 控制
# ============================================================
class MarkdownChunker:
    """Markdown 结构感知分块器

    完整复现 polyagent 第八章 _split_paragraphs_with_headings + _chunk_paragraphs:
      1. 解析标题层次 (#/##/###/...) 并维持栈
      2. 按空行分段,记录每段的 heading_path
      3. 按 token 数累积;超出阈值则切片,保留 overlap
      4. 重叠时优先保留后面的段落,维持主题连贯

    参数:
      chunk_tokens    - 单块最大 token 数(默认 500)
      overlap_tokens  - 块间重叠 token 数(默认 80)
    """

    def __init__(self, chunk_tokens: int = 500, overlap_tokens: int = 80) -> None:
        if chunk_tokens < 32:
            chunk_tokens = 32
        if overlap_tokens < 0:
            overlap_tokens = 0
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens

    # ------------------------------------------------------------------
    #  标题感知分段
    # ------------------------------------------------------------------
    @staticmethod
    def _split_paragraphs_with_headings(text: str) -> List[Dict]:
        """根据 Markdown 标题分割段落,保留 heading_path 元数据"""
        lines = text.splitlines()
        heading_stack: List[str] = []
        paragraphs: List[Dict] = []
        buf: List[str] = []
        char_pos = 0

        def flush_buf(end_pos: int) -> None:
            if not buf:
                return
            content = "\n".join(buf).strip()
            if not content:
                return
            paragraphs.append({
                "content": content,
                "heading_path": " > ".join(heading_stack) if heading_stack else None,
                "start": max(0, end_pos - len(content)),
                "end": end_pos,
            })

        for raw in lines:
            stripped = raw.strip()
            if stripped.startswith("#"):
                flush_buf(char_pos)
                level = len(stripped) - len(stripped.lstrip("#"))
                title = stripped.lstrip("#").strip()
                if level <= 0:
                    level = 1
                # 维护标题栈:同层或更深的弹出
                if level <= len(heading_stack):
                    heading_stack = heading_stack[: level - 1]
                heading_stack.append(title)
                char_pos += len(raw) + 1
                continue

            if stripped == "":
                flush_buf(char_pos)
                buf = []
            else:
                buf.append(raw)
            char_pos += len(raw) + 1

        flush_buf(char_pos)

        if not paragraphs:
            paragraphs = [{
                "content": text,
                "heading_path": None,
                "start": 0,
                "end": len(text),
            }]
        return paragraphs

    # ------------------------------------------------------------------
    #  Token 控制 + 重叠
    # ------------------------------------------------------------------
    def _chunk_paragraphs(self, paragraphs: List[Dict]) -> List[Dict]:
        chunks: List[Dict] = []
        cur: List[Dict] = []
        cur_tokens = 0
        i = 0

        while i < len(paragraphs):
            p = paragraphs[i]
            p_tokens = _approx_token_len(p["content"]) or 1

            if cur_tokens + p_tokens <= self.chunk_tokens or not cur:
                cur.append(p)
                cur_tokens += p_tokens
                i += 1
                continue

            chunks.append(self._emit(cur))
            # 重叠:从 cur 尾部回填一部分
            if self.overlap_tokens > 0 and cur:
                kept: List[Dict] = []
                kept_tokens = 0
                for x in reversed(cur):
                    t = _approx_token_len(x["content"]) or 1
                    if kept_tokens + t > self.overlap_tokens:
                        break
                    kept.append(x)
                    kept_tokens += t
                cur = list(reversed(kept))
                cur_tokens = kept_tokens
            else:
                cur = []
                cur_tokens = 0

        if cur:
            chunks.append(self._emit(cur))
        return chunks

    @staticmethod
    def _emit(group: List[Dict]) -> Dict:
        content = "\n\n".join(x["content"] for x in group)
        start = group[0]["start"]
        end = group[-1]["end"]
        heading_path = next(
            (x["heading_path"] for x in reversed(group) if x.get("heading_path")),
            None,
        )
        return {
            "content": content,
            "start": start,
            "end": end,
            "heading_path": heading_path,
        }

    # ------------------------------------------------------------------
    #  对外主入口
    # ------------------------------------------------------------------
    def split(self, text: str, doc_id: str = "doc") -> List[Chunk]:
        text = (text or "").strip()
        if not text:
            return []
        paragraphs = self._split_paragraphs_with_headings(text)
        raw_chunks = self._chunk_paragraphs(paragraphs)

        out: List[Chunk] = []
        for idx, c in enumerate(raw_chunks):
            content = (c.get("content") or "").strip()
            if not content:
                continue
            md = {
                "heading_path": c.get("heading_path"),
                "start": c.get("start"),
                "end": c.get("end"),
                "token_len": _approx_token_len(content),
            }
            out.append(
                Chunk(
                    id=f"{doc_id}_chunk_{idx}",
                    content=content,
                    doc_id=doc_id,
                    chunk_index=idx,
                    metadata=md,
                )
            )
        return out


__all__ = ["Chunk", "TextChunker", "MarkdownChunker", "_approx_token_len", "_is_cjk"]