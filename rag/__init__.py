"""rag 包 — RAG 检索增强 + 混合检索 + 重排序 + MarkItDown 多模态载入"""
from rag.chunker import Chunk, TextChunker, MarkdownChunker
from rag.embedder import Embedder
from rag.hybrid_search import HybridSearch
from rag.loader import (
    AudioConverter,
    BaseConverter,
    ConverterRegistry,
    CodeConverter,
    CSVConverter,
    DocxConverter,
    Document,
    DocumentLoader,
    HTMLConverter,
    ImageConverter,
    JSONConverter,
    MarkdownConverter,
    PdfConverter,
    PptxConverter,
    TextConverter,
    XlsxConverter,
    XMLConverter,
    convert_to_markdown,
    get_default_registry,
)
from rag.rag_tool import RAGTool
from rag.reranker import LLMReranker
from rag.vector_store import VectorRecord, VectorStore

__all__ = [
    # 主入口
    "Document", "DocumentLoader", "convert_to_markdown", "get_default_registry",
    # 转换器
    "BaseConverter", "ConverterRegistry",
    "TextConverter", "MarkdownConverter", "CodeConverter", "CSVConverter",
    "JSONConverter", "HTMLConverter", "XMLConverter",
    "DocxConverter", "PptxConverter", "XlsxConverter",
    "PdfConverter", "ImageConverter", "AudioConverter",
    # 分块
    "Chunk", "TextChunker", "MarkdownChunker",
    # 嵌入 / 向量
    "Embedder", "VectorRecord", "VectorStore",
    # 检索
    "HybridSearch", "LLMReranker",
    # 工具
    "RAGTool",
]