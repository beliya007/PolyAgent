"""
文档加载器 (Document Loader) - MarkItDown 风格完整实现

参考 polyagent 第八章 _convert_to_markdown 设计:

支持的格式 (统一转换为 Markdown 中间表示):
    - 文档    : PDF / DOCX / PPTX / XLSX
    - 文本    : TXT / Markdown / HTML / CSV / JSON / XML
    - 图片    : JPG / PNG / GIF / BMP / WEBP(可选 OCR)
    - 音频    : MP3 / WAV / M4A(可选转录)
    - 代码    : PY / JS / JAVA / GO / ...

设计要点:
    1. 转换器注册表 (Converter Registry):
       每个格式族一个 BaseConverter,运行时按文件扩展名匹配
       缺失相应可选库时,该 converter 优雅跳过,系统继续工作

    2. PDF 增强处理 (Enhanced PDF):
       按优先级尝试 pdfplumber → PyMuPDF(fitz) → pypdf → 标准库降级

    3. 标准库兜底 (Standard Library Fallback):
       所有可选库都缺失时,仍能用 zipfile / xml / csv / re 等纯标准库
       解析 DOCX/PPTX/XLSX/PDF 的可读部分

    4. 模态自动推断:
       根据扩展名 + 内容嗅探,返回 modality ∈
       {text, document, spreadsheet, presentation, image, audio, video, code, unknown}

    5. 缓存友好的设计:
       load_directory() 按 mtime 跳过未变更的文件,重复摄取零成本
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import struct
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

# ============================================================
#  Document 数据结构
# ============================================================
@dataclass
class Document:
    """统一文档对象"""
    id: str
    content: str
    source: str
    format: str = "text"           # 原始扩展名(去点)
    modality: str = "text"          # text/document/spreadsheet/presentation/image/audio/video/code
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "format": self.format,
            "modality": self.modality,
            "metadata": dict(self.metadata),
        }


# ============================================================
#  转换器抽象
# ============================================================
class BaseConverter(ABC):
    """格式转换器基类

    每个 converter 负责把一种 (或一族) 文件格式转换为 Markdown 文本。
    """
    name: str = "base"
    supported_exts: Tuple[str, ...] = ()
    priority: int = 100          # 数字越小优先级越高
    modality: str = "text"

    def can_handle(self, path: Path) -> bool:
        ext = path.suffix.lower().lstrip(".")
        return ext in self.supported_exts

    @abstractmethod
    def convert(self, path: Path) -> str:
        ...

    def is_available(self) -> bool:
        """返回该 converter 是否可用于当前环境(依赖是否就绪)"""
        return True


# ============================================================
#  文本类转换器(标准库,始终可用)
# ============================================================
class TextConverter(BaseConverter):
    """TXT / Markdown 等纯文本格式"""
    name = "text"
    supported_exts = ("txt", "md", "markdown", "rst", "log")
    priority = 10
    modality = "text"

    def convert(self, path: Path) -> str:
        return self._read_text_with_fallback(path)

    @staticmethod
    def _read_text_with_fallback(path: Path) -> str:
        """依次尝试 utf-8 / gbk / utf-8-sig / latin-1"""
        for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030", "latin-1"):
            try:
                return path.read_text(encoding=enc)
            except (UnicodeDecodeError, LookupError):
                continue
        # 最终兜底:忽略错误
        return path.read_text(encoding="utf-8", errors="ignore")


class CSVConverter(BaseConverter):
    """CSV → Markdown 表格"""
    name = "csv"
    supported_exts = ("csv",)
    priority = 10
    modality = "spreadsheet"

    def convert(self, path: Path) -> str:
        text = TextConverter._read_text_with_fallback(path)
        # 自动嗅探分隔符
        sample = text[:4096]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(io.StringIO(text), dialect)
        rows = [r for r in reader if any((c or "").strip() for c in r)]
        if not rows:
            return ""
        return CSVConverter._to_markdown_table(rows)

    @staticmethod
    def _to_markdown_table(rows: List[List[str]]) -> str:
        if not rows:
            return ""
        n_cols = max(len(r) for r in rows)
        # 补齐
        rows = [r + [""] * (n_cols - len(r)) for r in rows]
        header, *body = rows
        lines = [
            "| " + " | ".join(CSVConverter._escape(c) for c in header) + " |",
            "| " + " | ".join("---" for _ in header) + " |",
        ]
        for r in body:
            lines.append("| " + " | ".join(CSVConverter._escape(c) for c in r) + " |")
        return "\n".join(lines)

    @staticmethod
    def _escape(s: str) -> str:
        return (s or "").replace("|", "\\|").replace("\n", " ").strip()


class JSONConverter(BaseConverter):
    """JSON → Markdown 代码块 + 概览"""
    name = "json"
    supported_exts = ("json",)
    priority = 10
    modality = "text"

    def convert(self, path: Path) -> str:
        text = TextConverter._read_text_with_fallback(path)
        try:
            obj = json.loads(text)
            pretty = json.dumps(obj, ensure_ascii=False, indent=2)
        except Exception:
            pretty = text
        return f"```json\n{pretty}\n```"


class HTMLConverter(BaseConverter):
    """HTML → 清洗后 Markdown"""
    name = "html"
    supported_exts = ("html", "htm", "xhtml")
    priority = 20
    modality = "text"

    @staticmethod
    def _heading_repl(match: "re.Match") -> str:
        level = int(match.group(1)[1])  # h1 -> 1
        text = re.sub(r"<[^>]+>", "", match.group(2)).strip()
        return f"\n\n{'#' * level} {text}\n\n"

    @staticmethod
    def _inline_repl(match: "re.Match", tag: str) -> str:
        inner = re.sub(r"<[^>]+>", "", match.group(1)).strip()
        if tag in ("strong", "b"):
            return f"**{inner}**"
        if tag in ("em", "i"):
            return f"*{inner}*"
        if tag == "code":
            return f"`{inner}`"
        return inner

    def convert(self, path: Path) -> str:
        html = TextConverter._read_text_with_fallback(path)

        # 先剥 script / style / 注释
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)

        # 标题
        md = re.sub(r"<(h[1-6])[^>]*>(.*?)</\1>", self._heading_repl, html, flags=re.DOTALL | re.IGNORECASE)
        # 段落
        md = re.sub(r"<p[^>]*>(.*?)</p>", lambda m: f"\n\n{m.group(1).strip()}\n\n", md, flags=re.DOTALL | re.IGNORECASE)
        md = re.sub(r"<br\s*/?>", "\n", md, flags=re.IGNORECASE)
        md = re.sub(r"<li[^>]*>(.*?)</li>", lambda m: f"\n- {m.group(1).strip()}", md, flags=re.DOTALL | re.IGNORECASE)
        md = re.sub(r"<strong[^>]*>(.*?)</strong>", lambda m: f"**{m.group(1).strip()}**", md, flags=re.DOTALL | re.IGNORECASE)
        md = re.sub(r"<b[^>]*>(.*?)</b>", lambda m: f"**{m.group(1).strip()}**", md, flags=re.DOTALL | re.IGNORECASE)
        md = re.sub(r"<em[^>]*>(.*?)</em>", lambda m: f"*{m.group(1).strip()}*", md, flags=re.DOTALL | re.IGNORECASE)
        md = re.sub(r"<i[^>]*>(.*?)</i>", lambda m: f"*{m.group(1).strip()}*", md, flags=re.DOTALL | re.IGNORECASE)
        md = re.sub(r"<code[^>]*>(.*?)</code>", lambda m: f"`{m.group(1).strip()}`", md, flags=re.DOTALL | re.IGNORECASE)
        md = re.sub(r"<a[^>]*href=[\"'](.*?)[\"'][^>]*>(.*?)</a>", lambda m: f"[{m.group(2).strip()}]({m.group(1)})", md, flags=re.DOTALL | re.IGNORECASE)

        # 去掉剩余标签
        md = re.sub(r"<[^>]+>", " ", md)
        # 实体解码
        md = (
            md.replace("&nbsp;", " ")
              .replace("&amp;", "&")
              .replace("&lt;", "<")
              .replace("&gt;", ">")
              .replace("&quot;", '"')
              .replace("&#39;", "'")
        )
        # 合并多余空白
        md = re.sub(r"[ \t]+", " ", md)
        md = re.sub(r"\n{3,}", "\n\n", md)
        return md.strip()


class XMLConverter(BaseConverter):
    """XML / SVG 等 → 标签清理后的文本"""
    name = "xml"
    supported_exts = ("xml", "svg", "rss", "atom")
    priority = 20
    modality = "text"

    def convert(self, path: Path) -> str:
        text = TextConverter._read_text_with_fallback(path)
        # 简单 XML 解析(无需第三方库)
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(text)
            lines = [self._walk(root, depth=0)]
            return "\n".join(lines)
        except Exception:
            # 降级:仅去掉标签
            return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()

    def _walk(self, elem, depth: int) -> str:
        indent = "  " * depth
        text = (elem.text or "").strip()
        children = "\n".join(self._walk(c, depth + 1) for c in elem)
        if children:
            return f"{indent}{elem.tag}: {text}\n{children}"
        return f"{indent}{elem.tag}: {text}" if text else ""


class CodeConverter(BaseConverter):
    """源代码 → Markdown fenced code block"""
    name = "code"
    supported_exts = (
        "py", "js", "ts", "jsx", "tsx", "java", "go", "rs", "cpp", "c", "h",
        "cs", "rb", "php", "sh", "bash", "sql", "yaml", "yml", "toml", "ini",
        "conf", "kt", "swift", "scala", "r", "lua", "pl", "dart", "vue",
    )
    priority = 10
    modality = "code"

    _EXT_TO_LANG = {
        "py": "python", "js": "javascript", "ts": "typescript",
        "jsx": "jsx", "tsx": "tsx", "java": "java", "go": "go",
        "rs": "rust", "cpp": "cpp", "c": "c", "h": "c",
        "cs": "csharp", "rb": "ruby", "php": "php",
        "sh": "bash", "bash": "bash", "sql": "sql",
        "yaml": "yaml", "yml": "yaml", "toml": "toml",
        "ini": "ini", "conf": "ini", "kt": "kotlin",
        "swift": "swift", "scala": "scala", "r": "r",
        "lua": "lua", "pl": "perl", "dart": "dart", "vue": "vue",
    }

    def convert(self, path: Path) -> str:
        text = TextConverter._read_text_with_fallback(path)
        lang = self._EXT_TO_LANG.get(path.suffix.lower().lstrip("."), "")
        return f"```{lang}\n{text}\n```"


# ============================================================
#  Office 文档转换器
# ============================================================
class DocxConverter(BaseConverter):
    """DOCX → Markdown

    优先级链:
      1. python-docx (若安装)
      2. mammoth     (若安装)
      3. 标准库 zipfile + xml 解析降级
    """
    name = "docx"
    supported_exts = ("docx",)
    priority = 30
    modality = "document"

    def is_available(self) -> bool:
        return True  # 标准库降级永远可用

    def convert(self, path: Path) -> str:
        text = self._with_python_docx(path)
        if text is not None:
            return text
        text = self._with_mammoth(path)
        if text is not None:
            return text
        return self._stdlib_fallback(path)

    # ---------- 路径 1: python-docx ----------
    def _with_python_docx(self, path: Path) -> Optional[str]:
        try:
            from docx import Document as DocxDoc  # type: ignore
        except Exception:
            return None
        try:
            doc = DocxDoc(str(path))
            lines: List[str] = []
            for para in doc.paragraphs:
                txt = (para.text or "").strip()
                if not txt:
                    lines.append("")
                    continue
                style = (para.style.name or "").lower() if para.style else ""
                if "heading 1" in style:
                    lines.append(f"\n# {txt}\n")
                elif "heading 2" in style:
                    lines.append(f"\n## {txt}\n")
                elif "heading 3" in style:
                    lines.append(f"\n### {txt}\n")
                elif "heading" in style:
                    lines.append(f"\n#### {txt}\n")
                else:
                    lines.append(txt)
            # 表格
            for table in doc.tables:
                rows = [[(cell.text or "").strip() for cell in row.cells] for row in table.rows]
                if rows:
                    lines.append("")
                    lines.append(CSVConverter._to_markdown_table(rows))
            return "\n".join(lines).strip()
        except Exception as e:
            print(f"[DocxConverter] python-docx 失败: {e}")
            return None

    # ---------- 路径 2: mammoth ----------
    def _with_mammoth(self, path: Path) -> Optional[str]:
        try:
            import mammoth  # type: ignore
        except Exception:
            return None
        try:
            with open(path, "rb") as f:
                result = mammoth.convert_to_markdown(f)
            return (result.value or "").strip()
        except Exception as e:
            print(f"[DocxConverter] mammoth 失败: {e}")
            return None

    # ---------- 路径 3: 标准库降级 ----------
    def _stdlib_fallback(self, path: Path) -> str:
        """DOCX 本质是 ZIP,内含 word/document.xml,用 zipfile + 正则提取"""
        try:
            with zipfile.ZipFile(path, "r") as z:
                with z.open("word/document.xml") as f:
                    xml_bytes = f.read()
            xml_text = xml_bytes.decode("utf-8", errors="ignore")
            # 简单提取段落: <w:p>...</w:p>
            paragraphs = re.findall(r"<w:p[^>]*>(.*?)</w:p>", xml_text, flags=re.DOTALL)
            lines: List[str] = []
            for p in paragraphs:
                # 提取所有 <w:t> 文本节点
                texts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", p, flags=re.DOTALL)
                line = "".join(texts).strip()
                if line:
                    lines.append(line)
            return "\n".join(lines)
        except Exception as e:
            print(f"[DocxConverter] stdlib 降级失败: {e}")
            return ""


class PptxConverter(BaseConverter):
    """PPTX → Markdown(每页一张幻灯片)"""
    name = "pptx"
    supported_exts = ("pptx",)
    priority = 30
    modality = "presentation"

    def is_available(self) -> bool:
        return True

    def convert(self, path: Path) -> str:
        text = self._with_python_pptx(path)
        if text is not None:
            return text
        return self._stdlib_fallback(path)

    def _with_python_pptx(self, path: Path) -> Optional[str]:
        try:
            from pptx import Presentation  # type: ignore
        except Exception:
            return None
        try:
            prs = Presentation(str(path))
            slides_md: List[str] = []
            for i, slide in enumerate(prs.slides, 1):
                parts: List[str] = [f"\n# 幻灯片 {i}\n"]
                for shape in slide.shapes:
                    if not shape.has_text_frame:
                        if shape.shape_type == 19:  # TABLE
                            try:
                                tbl = shape.table
                                rows = [[(c.text or "").strip() for c in row.cells] for row in tbl.rows]
                                if rows:
                                    parts.append(CSVConverter._to_markdown_table(rows))
                            except Exception:
                                pass
                        continue
                    for para in shape.text_frame.paragraphs:
                        line = "".join(run.text for run in para.runs).strip()
                        if line:
                            lvl = (para.level or 0)
                            parts.append(("  " * lvl) + "- " + line if lvl > 0 else line)
                slides_md.append("\n".join(parts))
            return "\n".join(slides_md).strip()
        except Exception as e:
            print(f"[PptxConverter] python-pptx 失败: {e}")
            return None

    def _stdlib_fallback(self, path: Path) -> str:
        try:
            slides_md: List[str] = []
            with zipfile.ZipFile(path, "r") as z:
                slide_files = sorted(
                    [n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)],
                    key=lambda n: int(re.search(r"slide(\d+)\.xml", n).group(1)),
                )
                for i, name in enumerate(slide_files, 1):
                    with z.open(name) as f:
                        xml = f.read().decode("utf-8", errors="ignore")
                    texts = re.findall(r"<a:t[^>]*>(.*?)</a:t>", xml, flags=re.DOTALL)
                    lines = [t.strip() for t in texts if t.strip()]
                    slides_md.append(f"\n# 幻灯片 {i}\n" + "\n".join(lines))
            return "\n".join(slides_md).strip()
        except Exception as e:
            print(f"[PptxConverter] stdlib 降级失败: {e}")
            return ""


class XlsxConverter(BaseConverter):
    """XLSX → Markdown(每张工作表一张表)"""
    name = "xlsx"
    supported_exts = ("xlsx", "xlsm")
    priority = 30
    modality = "spreadsheet"

    def is_available(self) -> bool:
        return True

    def convert(self, path: Path) -> str:
        text = self._with_openpyxl(path)
        if text is not None:
            return text
        return self._stdlib_fallback(path)

    def _with_openpyxl(self, path: Path) -> Optional[str]:
        try:
            import openpyxl  # type: ignore
        except Exception:
            return None
        try:
            wb = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
            parts: List[str] = []
            for ws in wb.worksheets:
                parts.append(f"\n# 工作表:{ws.title}\n")
                rows: List[List[str]] = []
                for row in ws.iter_rows(values_only=True):
                    if row is None:
                        continue
                    rows.append([("" if v is None else str(v)) for v in row])
                if rows:
                    parts.append(CSVConverter._to_markdown_table(rows))
            return "\n".join(parts).strip()
        except Exception as e:
            print(f"[XlsxConverter] openpyxl 失败: {e}")
            return None

    def _stdlib_fallback(self, path: Path) -> str:
        try:
            parts: List[str] = []
            with zipfile.ZipFile(path, "r") as z:
                # 解析 sharedStrings.xml(可选)
                shared: List[str] = []
                if "xl/sharedStrings.xml" in z.namelist():
                    with z.open("xl/sharedStrings.xml") as f:
                        sxml = f.read().decode("utf-8", errors="ignore")
                    shared = re.findall(r"<t[^>]*>(.*?)</t>", sxml, flags=re.DOTALL)
                sheet_files = sorted(
                    [n for n in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", n)],
                )
                for name in sheet_files:
                    with z.open(name) as f:
                        sxml = f.read().decode("utf-8", errors="ignore")
                    rows_xml = re.findall(r"<row[^>]*>(.*?)</row>", sxml, flags=re.DOTALL)
                    rows: List[List[str]] = []
                    for row in rows_xml:
                        cells = re.findall(r"<c\s[^>]*?(?:t=\"(.*?)\")?[^>]*>(.*?)</c>", row, flags=re.DOTALL)
                        row_out: List[str] = []
                        for ctype, body in cells:
                            v_match = re.search(r"<v[^>]*>(.*?)</v>", body)
                            if not v_match:
                                is_match = re.search(r"<is><t[^>]*>(.*?)</t></is>", body)
                                v = is_match.group(1) if is_match else ""
                            else:
                                v = v_match.group(1)
                                if ctype == "s" and v.isdigit() and 0 <= int(v) < len(shared):
                                    v = shared[int(v)]
                            row_out.append(v)
                        rows.append(row_out)
                    title = Path(name).stem
                    parts.append(f"\n# 工作表:{title}\n")
                    if rows:
                        parts.append(CSVConverter._to_markdown_table(rows))
            return "\n".join(parts).strip()
        except Exception as e:
            print(f"[XlsxConverter] stdlib 降级失败: {e}")
            return ""


# ============================================================
#  PDF 转换器(增强版,多后端 fallback)
# ============================================================
class PdfConverter(BaseConverter):
    """PDF → Markdown

    优先级链:
      1. pdfplumber (文本 + 表格抽取质量最好)
      2. PyMuPDF / fitz  (速度快,内置 OCR 钩子)
      3. pypdf          (纯 Python,广泛可用)
      4. 标准库正则降级 (从压缩流里尝试提文本)
    """
    name = "pdf"
    supported_exts = ("pdf",)
    priority = 25
    modality = "document"

    def is_available(self) -> bool:
        return True

    def convert(self, path: Path) -> str:
        for fn in (
            self._with_pdfplumber,
            self._with_pymupdf,
            self._with_pypdf,
            self._stdlib_fallback,
        ):
            text = fn(path)
            if text and len(text.strip()) >= 20:
                return text
        return ""

    def _with_pdfplumber(self, path: Path) -> Optional[str]:
        try:
            import pdfplumber  # type: ignore
        except Exception:
            return None
        try:
            parts: List[str] = []
            with pdfplumber.open(str(path)) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    parts.append(f"\n## 第 {i} 页\n")
                    text = page.extract_text() or ""
                    parts.append(text)
                    # 尝试抽表格
                    try:
                        tables = page.extract_tables()
                        for tbl in tables or []:
                            rows = [[(c or "") for c in r] for r in tbl if r]
                            if rows:
                                parts.append(CSVConverter._to_markdown_table(rows))
                    except Exception:
                        pass
            return "\n".join(parts).strip()
        except Exception as e:
            print(f"[PdfConverter] pdfplumber 失败: {e}")
            return None

    def _with_pymupdf(self, path: Path) -> Optional[str]:
        try:
            import fitz  # type: ignore  # PyMuPDF
        except Exception:
            return None
        try:
            doc = fitz.open(str(path))
            parts: List[str] = []
            for i, page in enumerate(doc, 1):
                parts.append(f"\n## 第 {i} 页\n")
                parts.append(page.get_text("text"))
            doc.close()
            return "\n".join(parts).strip()
        except Exception as e:
            print(f"[PdfConverter] PyMuPDF 失败: {e}")
            return None

    def _with_pypdf(self, path: Path) -> Optional[str]:
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception:
            try:
                from PyPDF2 import PdfReader  # type: ignore  # 旧版兼容
            except Exception:
                return None
        try:
            reader = PdfReader(str(path))
            parts: List[str] = []
            for i, page in enumerate(reader.pages, 1):
                parts.append(f"\n## 第 {i} 页\n")
                try:
                    parts.append(page.extract_text() or "")
                except Exception:
                    parts.append("")
            return "\n".join(parts).strip()
        except Exception as e:
            print(f"[PdfConverter] pypdf 失败: {e}")
            return None

    def _stdlib_fallback(self, path: Path) -> str:
        """最后兜底:从 PDF 二进制流里正则提取可见文本"""
        try:
            data = path.read_bytes()
            # 粗略:提取 ( ... ) Tj / TJ 之间的字符串
            # 仅能拿到部分文字,但至少不会抛错
            text_chunks: List[str] = []
            for m in re.finditer(rb"\((.*?)\)\s*T[jJ]", data, flags=re.DOTALL):
                try:
                    s = m.group(1).decode("utf-8", errors="ignore")
                    if s.strip():
                        text_chunks.append(s)
                except Exception:
                    pass
            if not text_chunks:
                return ""
            # 简单按页分组(无法精确,直接合并)
            return "\n".join(text_chunks)
        except Exception as e:
            print(f"[PdfConverter] stdlib 降级失败: {e}")
            return ""


# ============================================================
#  图片 / 音频(可选)
# ============================================================
class ImageConverter(BaseConverter):
    """图片:若有 OCR(pytesseract)则提取文字,否则只记录元数据"""
    name = "image"
    supported_exts = ("png", "jpeg", "jpg", "gif", "bmp", "webp", "tiff")
    priority = 40
    modality = "image"

    def convert(self, path: Path) -> str:
        meta = self._extract_metadata(path)
        ocr_text = self._try_ocr(path)
        if ocr_text:
            return f"# 图片:{path.name}\n\n{ocr_text}\n\n{self._meta_to_md(meta)}"
        return f"# 图片:{path.name}\n\n(未提取到文字内容)\n\n{self._meta_to_md(meta)}"

    def _extract_metadata(self, path: Path) -> Dict:
        meta: Dict = {"filename": path.name, "size": path.stat().st_size}
        # PIL 可用时读 width/height
        try:
            from PIL import Image  # type: ignore
            with Image.open(str(path)) as im:
                meta["width"] = im.width
                meta["height"] = im.height
                meta["mode"] = im.mode
                meta["format"] = im.format
        except Exception:
            pass
        return meta

    def _try_ocr(self, path: Path) -> str:
        try:
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore
        except Exception:
            return ""
        try:
            with Image.open(str(path)) as im:
                return (pytesseract.image_to_string(im, lang="chi_sim+eng") or "").strip()
        except Exception as e:
            print(f"[ImageConverter] OCR 失败: {e}")
            return ""

    def _meta_to_md(self, meta: Dict) -> str:
        return "```\n" + json.dumps(meta, ensure_ascii=False, indent=2) + "\n```"


class AudioConverter(BaseConverter):
    """音频:若有转录工具则调用,否则只记录元数据"""
    name = "audio"
    supported_exts = ("mp3", "wav", "m4a", "flac", "ogg", "aac")
    priority = 40
    modality = "audio"

    def convert(self, path: Path) -> str:
        meta = self._extract_metadata(path)
        transcript = self._try_transcribe(path)
        if transcript:
            return f"# 音频:{path.name}\n\n{transcript}\n\n{self._meta_to_md(meta)}"
        return f"# 音频:{path.name}\n\n(未提取到转录内容)\n\n{self._meta_to_md(meta)}"

    def _extract_metadata(self, path: Path) -> Dict:
        meta: Dict = {"filename": path.name, "size": path.stat().st_size}
        # mutagen 可读 ID3 / Vorbis 标签
        try:
            import mutagen  # type: ignore
            mf = mutagen.File(str(path))
            if mf is not None:
                meta["duration"] = getattr(mf.info, "length", None)
                meta["bitrate"] = getattr(mf.info, "bitrate", None)
        except Exception:
            pass
        return meta

    def _try_transcribe(self, path: Path) -> str:
        # 1) faster-whisper(本地)
        try:
            from faster_whisper import WhisperModel  # type: ignore
            model = WhisperModel("base", device="cpu", compute_type="int8")
            segments, _ = model.transcribe(str(path), beam_size=5)
            return " ".join(seg.text for seg in segments).strip()
        except Exception:
            pass
        # 2) openai-whisper(本地)
        try:
            import whisper  # type: ignore
            model = whisper.load_model("base")
            result = model.transcribe(str(path))
            return (result.get("text") or "").strip()
        except Exception:
            pass
        # 3) speech_recognition + 第三方(网络)
        try:
            import speech_recognition as sr  # type: ignore
            r = sr.Recognizer()
            with sr.AudioFile(str(path)) as source:
                audio = r.record(source)
            return r.recognize_google(audio, language="zh-CN")
        except Exception:
            pass
        return ""

    def _meta_to_md(self, meta: Dict) -> str:
        return "```\n" + json.dumps(meta, ensure_ascii=False, indent=2) + "\n```"


# ============================================================
#  转换器注册表(运行时自动按可用性启用)
# ============================================================
class ConverterRegistry:
    """转换器注册表

    按 priority 升序排序,优先级高的 converter 优先匹配。
    缺失可选库的 converter 会被自动跳过。
    """

    def __init__(self, converters: Optional[List[BaseConverter]] = None) -> None:
        self._converters: List[BaseConverter] = []
        if converters is None:
            converters = self._default_converters()
        for c in converters:
            self.register(c)

    def register(self, converter: BaseConverter) -> None:
        if converter.is_available():
            self._converters.append(converter)
        self._converters.sort(key=lambda c: c.priority)

    @staticmethod
    def _default_converters() -> List[BaseConverter]:
        """返回默认转换器列表(纯 stdlib 的始终在前)"""
        return [
            TextConverter(),
            MarkdownConverter(),
            CodeConverter(),
            CSVConverter(),
            JSONConverter(),
            HTMLConverter(),
            XMLConverter(),
            DocxConverter(),
            PptxConverter(),
            XlsxConverter(),
            PdfConverter(),
            ImageConverter(),
            AudioConverter(),
        ]

    def find(self, path: Path) -> Optional[BaseConverter]:
        for c in self._converters:
            if c.can_handle(path):
                return c
        return None

    def list_supported(self) -> List[str]:
        out: List[str] = []
        for c in self._converters:
            out.extend(c.supported_exts)
        return sorted(set(out))

    def converters_for_modality(self, modality: str) -> List[BaseConverter]:
        return [c for c in self._converters if c.modality == modality]


# MarkdownConverter 别名(等同 TextConverter,语义化)
class MarkdownConverter(TextConverter):
    name = "markdown"
    supported_exts = ("md", "markdown")
    priority = 5
    modality = "text"

    def convert(self, path: Path) -> str:
        return TextConverter._read_text_with_fallback(path)


# ============================================================
#  DocumentLoader(主入口)
# ============================================================
class DocumentLoader:
    """多格式文档加载

    用法:
        loader = DocumentLoader("./data/knowledge_base")
        docs = loader.load_directory()              # 整库加载
        docs = loader.load_files([Path("a.pdf")])   # 指定文件
        doc = loader.load_file(path)                # 单文件
        doc = loader.load_text("...", source="raw") # 原始文本
    """

    # 知识库目录需要被跳过的临时/缓存文件
    _SKIP_GLOBS = ("*.vectors.json", "*.vectors", "__pycache__/*", ".*")

    def __init__(
        self,
        knowledge_base_path: Optional[Union[str, Path]] = None,
        registry: Optional[ConverterRegistry] = None,
        enable_cache: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.kb_path = Path(knowledge_base_path) if knowledge_base_path else None
        if self.kb_path:
            self.kb_path.mkdir(parents=True, exist_ok=True)
        self.registry = registry or ConverterRegistry()
        self.enable_cache = enable_cache
        self.cache_dir = Path(cache_dir) if cache_dir else (
            self.kb_path / ".loader_cache" if self.kb_path else Path("./.loader_cache")
        )
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    #  单文件 / 文本
    # ============================================================
    def load_text(
        self,
        content: str,
        doc_id: str = "raw",
        source: str = "inline",
        metadata: Optional[Dict] = None,
    ) -> Document:
        return Document(
            id=doc_id,
            content=content,
            source=source,
            format="text",
            modality="text",
            metadata=metadata or {},
        )

    def load_file(
        self,
        path: Union[str, Path],
        force: bool = False,
    ) -> Document:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {p}")
        if not p.is_file():
            raise ValueError(f"不是文件: {p}")

        ext = p.suffix.lower().lstrip(".") or "unknown"
        cache_key = self._cache_key(p)

        # 缓存命中
        if self.enable_cache and not force and self._is_cached(p, cache_key):
            cached = self._read_cache(cache_key)
            if cached is not None:
                return cached

        # 选 converter
        converter = self.registry.find(p)
        if converter is None:
            # 没有任何 converter 接管 → 兜底按纯文本读
            content = TextConverter._read_text_with_fallback(p)
            modality = "text"
            converter_name = "fallback-text"
        else:
            try:
                content = converter.convert(p) or ""
            except Exception as e:
                print(f"⚠️ [{converter.name}] {p.name} 转换失败,降级纯文本: {e}")
                content = TextConverter._read_text_with_fallback(p)
                converter_name = f"{converter.name} (degraded)"
            else:
                converter_name = converter.name
            modality = converter.modality

        if not content.strip():
            print(f"⚠️ {p.name}: 提取内容为空")

        doc = Document(
            id=p.stem,
            content=content,
            source=str(p.resolve()),
            format=ext,
            modality=modality,
            metadata={
                "filename": p.name,
                "size": p.stat().st_size,
                "mtime": p.stat().st_mtime,
                "converter": converter_name,
            },
        )

        if self.enable_cache:
            self._write_cache(cache_key, doc)
        return doc

    # ============================================================
    #  批量
    # ============================================================
    def load_files(
        self,
        paths: Iterable[Union[str, Path]],
        skip_errors: bool = True,
    ) -> List[Document]:
        out: List[Document] = []
        for p in paths:
            try:
                out.append(self.load_file(p))
            except Exception as e:
                if skip_errors:
                    print(f"⚠️ 跳过 {p}: {e}")
                else:
                    raise
        return out

    def load_directory(
        self,
        pattern: str = "**/*.*",
        extra_skip: Optional[List[str]] = None,
    ) -> List[Document]:
        """加载整个知识库目录(递归)"""
        if not self.kb_path or not self.kb_path.exists():
            return []
        docs: List[Document] = []
        skip_set = set(self._SKIP_GLOBS) | set(extra_skip or [])
        for f in self.kb_path.glob(pattern):
            if not f.is_file():
                continue
            if any(f.match(g) for g in skip_set):
                continue
            try:
                docs.append(self.load_file(f))
            except Exception as e:
                print(f"⚠️ 跳过 {f.name}: {e}")
        return docs

    # ============================================================
    #  缓存(基于文件 mtime + size)
    # ============================================================
    def _cache_key(self, path: Path) -> str:
        st = path.stat()
        raw = f"{path.resolve()}|{st.st_size}|{st.st_mtime}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:16]

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _is_cached(self, path: Path, key: str) -> bool:
        cp = self._cache_path(key)
        if not cp.exists():
            return False
        # 缓存的元数据里应当记录源文件 mtime
        try:
            data = json.loads(cp.read_text(encoding="utf-8"))
            return float(data.get("source_mtime", -1)) == float(path.stat().st_mtime)
        except Exception:
            return False

    def _read_cache(self, key: str) -> Optional[Document]:
        try:
            data = json.loads(self._cache_path(key).read_text(encoding="utf-8"))
            return Document(
                id=data["id"],
                content=data["content"],
                source=data["source"],
                format=data.get("format", "text"),
                modality=data.get("modality", "text"),
                metadata=data.get("metadata", {}),
            )
        except Exception:
            return None

    def _write_cache(self, key: str, doc: Document) -> None:
        try:
            payload = {
                "id": doc.id,
                "content": doc.content,
                "source": doc.source,
                "format": doc.format,
                "modality": doc.modality,
                "metadata": {
                    **doc.metadata,
                    "source_mtime": doc.metadata.get("mtime", 0),
                },
            }
            self._cache_path(key).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"⚠️ 写缓存失败: {e}")

    def clear_cache(self) -> int:
        n = 0
        if not self.cache_dir.exists():
            return 0
        for p in self.cache_dir.glob("*.json"):
            try:
                p.unlink()
                n += 1
            except Exception:
                pass
        return n

    # ============================================================
    #  辅助:模态推断(独立函数,便于外部使用)
    # ============================================================
    @staticmethod
    def infer_modality(path: Union[str, Path]) -> str:
        ext = Path(path).suffix.lower().lstrip(".")
        for c in ConverterRegistry._default_converters():
            if ext in c.supported_exts:
                return c.modality
        return "unknown"

    @staticmethod
    def supported_extensions() -> List[str]:
        return ConverterRegistry().list_supported()


# ============================================================
#  MarkItDown 风格的高阶转换函数(便捷导出)
# ============================================================
_default_registry: Optional[ConverterRegistry] = None


def get_default_registry() -> ConverterRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = ConverterRegistry()
    return _default_registry


def convert_to_markdown(path: Union[str, Path]) -> str:
    """MarkItDown 风格的统一入口:任何文件 → Markdown 字符串

    与 polyagent 第八章 _convert_to_markdown 等价:
        - 文件不存在返回 ""
        - 各 converter 失败时降级到下一个,最终降级纯文本
    """
    p = Path(path)
    if not p.exists():
        return ""
    converter = get_default_registry().find(p)
    if converter is None:
        return TextConverter._read_text_with_fallback(p)
    try:
        return (converter.convert(p) or "").strip()
    except Exception as e:
        print(f"[convert_to_markdown] {converter.name} 失败,降级: {e}")
        return TextConverter._read_text_with_fallback(p)


__all__ = [
    "Document",
    "BaseConverter",
    "ConverterRegistry",
    "TextConverter",
    "MarkdownConverter",
    "CodeConverter",
    "CSVConverter",
    "JSONConverter",
    "HTMLConverter",
    "XMLConverter",
    "DocxConverter",
    "PptxConverter",
    "XlsxConverter",
    "PdfConverter",
    "ImageConverter",
    "AudioConverter",
    "DocumentLoader",
    "convert_to_markdown",
    "get_default_registry",
]