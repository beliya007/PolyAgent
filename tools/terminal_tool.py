"""
终端工具 (TerminalTool) — polyagent 第九章 §9.5

设计要点:
- 默认基于 Python 内置实现(跨 Windows/Linux),不直接开 subprocess
- 命令白名单 + workspace 沙箱,只能访问允许根目录下的路径
- 支持 raw subprocess 模式(subprocess=True 显式开启,默认 False)
- ContextBuilder 通过 quick_context 命令拉取 JIT 上下文
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.base import BaseTool, ToolParameter


# 文档 §9.5 定义的逻辑命令(基于 Python 实现,跨平台)
ALLOWED_COMMANDS = {
    # 文件列表与信息
    "ls", "dir", "tree", "pwd", "stat", "du", "df",
    # 文件内容查看
    "cat", "head", "tail", "wc",
    # 文本处理
    "sort", "uniq", "cut", "echo",
    # 搜索
    "find", "grep",
    # 其他
    "which", "whereis", "path",
    # ContextBuilder JIT 集成
    "quick_context",
}

# 显式禁用的危险字符(防止路径穿越 / shell 注入)
_DENY_PATTERNS = [
    re.compile(r"\.\.[\\/]"),  # ../
    re.compile(r"[<>|&;`$]"),  # shell 元字符
]


def _resolve_safe(path: str, roots: List[Path]) -> Path:
    """把 path 解析为绝对路径,并校验它落在任一 root 下

    相对路径会优先相对 workspace 的根目录解析,而非 Python 进程 cwd
    """
    p = Path(path).expanduser()
    if not p.is_absolute():
        # 相对路径 → 相对第一个 root(workspace)
        p = roots[0] / p
    abs_path = p.resolve()
    for root in roots:
        try:
            abs_path.relative_to(root.resolve())
            return abs_path
        except ValueError:
            continue
    raise PermissionError(f"路径越界,不允许访问 {path}")


class TerminalTool(BaseTool):
    """即时文件系统访问工具(JIT 上下文)

    使用::

        term = TerminalTool(workspace="./project")
        term.run({"command": "ls"})                     # 列出 workspace
        term.run({"command": "cat README.md"})          # 读文件
        term.run({"command": "grep TODO src/"})         # 关键字搜索
        term.run({"command": "quick_context 重构"})     # ContextBuilder 用

    安全:
        - workspace 及 additional_roots 内的路径才允许访问
        - 默认走 Python 实现;enable_subprocess=True 时才走 shell(白名单仍生效)
    """

    name = "terminal"
    description = (
        "即时文件系统访问工具 — 安全地执行只读命令(ls/cat/head/grep/find 等)。"
        "支持 ContextBuilder 的 JIT 上下文(quick_context 命令)。"
    )
    parameters = [
        ToolParameter("command", "string", "要执行的命令", required=True),
        ToolParameter("path", "string", "可选路径参数", required=False),
        ToolParameter("pattern", "string", "grep/find 的关键字", required=False),
        ToolParameter("limit", "integer", "返回行/条目数限制(默认 50)", required=False, default=50),
        ToolParameter("query", "string", "quick_context 用查询词", required=False),
        ToolParameter("cwd", "string", "子进程工作目录(subprocess 模式)", required=False),
    ]

    def __init__(
        self,
        workspace: str = ".",
        additional_roots: Optional[List[str]] = None,
        enable_subprocess: bool = False,
        timeout_seconds: float = 5.0,
    ):
        super().__init__()
        ws = Path(workspace).expanduser().resolve()
        ws.mkdir(parents=True, exist_ok=True)
        self.workspace = ws
        self.allowed_roots: List[Path] = [ws]
        for r in additional_roots or []:
            p = Path(r).expanduser().resolve()
            if p.exists():
                self.allowed_roots.append(p)
        self.enable_subprocess = enable_subprocess
        self.timeout = timeout_seconds

    # ---------- 入口 ----------
    def execute(self, **kwargs) -> str:
        command = kwargs.get("command", "").strip()
        if not command:
            return "❌ command 不能为空"
        for pat in _DENY_PATTERNS:
            if pat.search(command):
                return f"❌ 包含非法字符({pat.pattern})"

        parts = shlex.split(command)
        if not parts:
            return "❌ 空命令"
        op, *args = parts
        if op not in ALLOWED_COMMANDS:
            return f"❌ 不支持的命令: {op};允许 {sorted(ALLOWED_COMMANDS)}"

        try:
            handler = getattr(self, f"_cmd_{op}", None) or getattr(self, "_cmd_default")
            result = handler(op, args, kwargs)
            return result if isinstance(result, str) else str(result)
        except PermissionError as e:
            return f"❌ 权限拒绝: {e}"
        except FileNotFoundError as e:
            return f"❌ 文件不存在: {e}"
        except Exception as e:  # noqa: BLE001
            return f"❌ 执行失败: {e}"

    def _cmd_default(self, op: str, args: List[str], kwargs: Dict[str, Any]) -> str:
        if not self.enable_subprocess:
            return f"❌ raw shell 模式未启用;{op} 需要 enable_subprocess=True"
        return self._run_subprocess([op, *args], kwargs.get("cwd"))

    # ---------- raw subprocess(显式开启) ----------
    def _run_subprocess(self, cmd: List[str], cwd: Optional[str]) -> str:
        proc = subprocess.run(
            cmd,
            shell=False,
            cwd=cwd or str(self.workspace),
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        out = proc.stdout or ""
        if proc.returncode != 0:
            out += f"\n[exit={proc.returncode}] {proc.stderr or ''}"
        return out[:5000]

    # ==================================================================
    #  各命令的 Python 实现
    # ==================================================================
    def _cmd_ls(self, op, args, kw):
        path = self._first(args) or "."
        target = _resolve_safe(path, self.allowed_roots)
        limit = int(kw.get("limit", 50))
        rows = []
        for p in sorted(target.iterdir()):
            rows.append(f"{'d' if p.is_dir() else 'f'}  {p.name}")
        return "\n".join(rows[:limit])

    def _cmd_dir(self, op, args, kw):
        return self._cmd_ls(op, args, kw)

    def _cmd_tree(self, op, args, kw):
        path = self._first(args) or "."
        target = _resolve_safe(path, self.allowed_roots)
        max_lines = int(kw.get("limit", 100))

        lines: List[str] = [str(target)]
        count = 0

        def _walk(d: Path, prefix: str) -> None:
            nonlocal count
            if count >= max_lines:
                return
            entries = sorted(d.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            for i, entry in enumerate(entries):
                if count >= max_lines:
                    lines.append(f"{prefix}... (截断)")
                    return
                is_last = i == len(entries) - 1
                connector = "└── " if is_last else "├── "
                lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
                count += 1
                if entry.is_dir():
                    extension = "    " if is_last else "│   "
                    _walk(entry, prefix + extension)

        _walk(target, "")
        return "\n".join(lines)

    def _cmd_pwd(self, op, args, kw):
        return str(self.workspace)

    def _cmd_cat(self, op, args, kw):
        if not args:
            return "❌ cat 需要文件路径"
        path = self._first(args)
        target = _resolve_safe(path, self.allowed_roots)
        if not target.is_file():
            return f"❌ 不是文件: {path}"
        limit = int(kw.get("limit", 200))
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        if len(lines) > limit:
            return "\n".join(lines[:limit]) + f"\n... (共 {len(lines)} 行,已截断)"
        return "\n".join(lines)

    def _cmd_head(self, op, args, kw):
        return self._tail_or_head(args, kw, head=True)

    def _cmd_tail(self, op, args, kw):
        return self._tail_or_head(args, kw, head=False)

    def _tail_or_head(self, args: List[str], kw: Dict[str, Any], head: bool) -> str:
        path = self._first(args)
        if not path:
            return "❌ 需要文件路径"
        n = int(kw.get("limit", 10))
        target = _resolve_safe(path, self.allowed_roots)
        if not target.is_file():
            return f"❌ 不是文件: {path}"
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        chosen = lines[:n] if head else lines[-n:]
        return "\n".join(chosen)

    def _cmd_wc(self, op, args, kw):
        if not args:
            return "❌ wc 需要文件路径"
        path = self._first(args)
        target = _resolve_safe(path, self.allowed_roots)
        if not target.is_file():
            return f"❌ 不是文件: {path}"
        text = target.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        words = text.split()
        return f"{len(lines):>6} {len(words):>6} {len(text):>6} {target.name}"

    def _cmd_grep(self, op, args, kw):
        pattern = kw.get("pattern") or (args[0] if args else "")
        path = kw.get("path") or (args[1] if len(args) > 1 else ".")
        if not pattern:
            return "❌ grep 需要 pattern"
        target = _resolve_safe(path, self.allowed_roots)
        if not target.exists():
            return f"❌ 路径不存在: {path}"
        rx = re.compile(pattern, re.IGNORECASE)
        limit = int(kw.get("limit", 50))
        out: List[str] = []
        files = [target] if target.is_file() else list(target.rglob("*"))
        for f in files:
            if not f.is_file():
                continue
            try:
                for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if rx.search(line):
                        out.append(f"{f.name}:{i}: {line.strip()}")
                        if len(out) >= limit:
                            return "\n".join(out) + f"\n... (截断 {limit})"
            except Exception:
                continue
        return "\n".join(out) if out else "(无匹配)"

    def _cmd_find(self, op, args, kw):
        path = kw.get("path") or self._first(args) or "."
        pattern = kw.get("pattern") or (args[1] if len(args) > 1 else None)
        target = _resolve_safe(path, self.allowed_roots)
        if not target.exists():
            return f"❌ 路径不存在: {path}"
        limit = int(kw.get("limit", 50))
        out: List[str] = []
        for f in target.rglob("*"):
            if pattern and pattern.lower() not in f.name.lower():
                continue
            out.append(str(f.relative_to(target)))
            if len(out) >= limit:
                return "\n".join(out) + f"\n... (截断 {limit})"
        return "\n".join(out) if out else "(无匹配)"

    def _cmd_stat(self, op, args, kw):
        path = self._first(args) or "."
        target = _resolve_safe(path, self.allowed_roots)
        st = target.stat()
        return (f"name: {target.name}\n"
                f"size: {st.st_size}\n"
                f"mtime: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime))}\n"
                f"is_dir: {target.is_dir()}")

    def _cmd_du(self, op, args, kw):
        path = self._first(args) or "."
        target = _resolve_safe(path, self.allowed_roots)
        total = 0
        n = 0
        if target.is_file():
            total = target.stat().st_size
            n = 1
        else:
            for f in target.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
                    n += 1
        return f"{total} bytes  ({n} files)"

    def _cmd_path(self, op, args, kw):
        return str(self.workspace)

    def _cmd_echo(self, op, args, kw):
        return " ".join(args)

    # ---------- ContextBuilder JIT 入口 ----------
    def _cmd_quick_context(self, op, args, kw):
        """ContextBuilder 调用的 JIT 上下文

        策略:
            1. 在 workspace 内 grep query 关键词
            2. 列出最近修改的文件
            3. 输出拼好的"项目速览"片段
        """
        query = kw.get("query") or (args[0] if args else "")
        if not query:
            return ""
        limit = int(kw.get("limit", 5))
        out: List[str] = [f"[JIT] query={query!r} workspace={self.workspace}"]

        # 1) 关键词命中
        try:
            hits = self._cmd_grep("grep", [query], {**kw, "limit": limit})
        except Exception:
            hits = ""
        if hits and hits != "(无匹配)":
            out.append("--- 关键词命中 ---")
            out.append(hits)

        # 2) 最近修改的文件
        try:
            files_with_mtime: List[tuple[float, Path]] = []
            for f in self.workspace.rglob("*"):
                if f.is_file() and "notes_index" not in f.name:
                    try:
                        files_with_mtime.append((f.stat().st_mtime, f))
                    except OSError:
                        continue
            files_with_mtime.sort(reverse=True)
            recent = files_with_mtime[:limit]
            if recent:
                out.append("--- 最近修改 ---")
                for mtime, f in recent:
                    rel = f.relative_to(self.workspace)
                    out.append(f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))}  {rel}")
        except Exception:
            pass

        return "\n".join(out)

    # ---------- 辅助 ----------
    @staticmethod
    def _first(args: List[str]) -> Optional[str]:
        return args[0] if args else None
