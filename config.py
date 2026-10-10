"""
Unified configuration loader.

Reads ``.env`` and provides typed, layered configuration:
- LLMConfig      API key, model, base URL, timeout, ...
- BusinessConfig data paths, memory capacity, RAG parameters
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


# Project root = directory containing this file
AGENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = AGENT_ROOT.parent

# Load .env: prefer agent/.env, then project root .env
for env_path in [AGENT_ROOT / ".env", PROJECT_ROOT / ".env"]:
    if env_path.exists():
        load_dotenv(env_path)
        break
else:
    try:
        load_dotenv()
    except Exception:  # noqa: BLE001
        pass


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(key, default)
    return v.strip() if isinstance(v, str) else v


def _env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key, "")
    if not v:
        return default
    return v.lower() in ("true", "1", "yes", "on")


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


# ============================================================
#  LLM configuration
# ============================================================
@dataclass
class LLMConfig:
    api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", "") or "")
    base_url: str = field(
        default_factory=lambda: _env(
            "LLM_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        or ""
    )
    model: str = field(
        default_factory=lambda: _env("LLM_MODEL_ID", "qwen-plus") or "qwen-plus"
    )
    timeout: int = field(default_factory=lambda: _env_int("LLM_TIMEOUT", 120))
    embedding_model: str = field(
        default_factory=lambda: _env(
            "LLM_EMBEDDING_MODEL", "text-embedding-v3"
        )
        or "text-embedding-v3"
    )
    enable_thinking: bool = field(
        default_factory=lambda: _env_bool("ENABLE_THINKING", False)
    )
    default_temperature: float = field(
        default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.7)
    )


# ============================================================
#  Application / storage configuration
# ============================================================
@dataclass
class BusinessConfig:
    """Application-side configuration (domain-neutral)."""

    # Optional external backend (HTTP). When empty, all built-in tools run
    # in standalone / mock mode.
    java_backend_url: Optional[str] = field(
        default_factory=lambda: _env("JAVA_BACKEND_URL") or None
    )

    # RAG knowledge base directory (relative to the agent/ folder)
    knowledge_base_path: str = field(
        default_factory=lambda: _env(
            "KNOWLEDGE_BASE_PATH", "./data/knowledge_base"
        )
        or "./data/knowledge_base"
    )

    # Memory storage directory
    memory_storage_path: str = field(
        default_factory=lambda: _env("MEMORY_STORAGE_PATH", "./data/memory")
        or "./data/memory"
    )

    # Working-memory capacity and TTL
    working_memory_capacity: int = field(
        default_factory=lambda: _env_int("WORKING_MEMORY_CAPACITY", 50)
    )
    working_memory_ttl_min: int = field(
        default_factory=lambda: _env_int("WORKING_MEMORY_TTL", 60)
    )

    # RAG retrieval parameters
    rag_top_k: int = field(default_factory=lambda: _env_int("RAG_TOP_K", 5))
    rag_chunk_size: int = field(default_factory=lambda: _env_int("RAG_CHUNK_SIZE", 500))
    rag_chunk_overlap: int = field(
        default_factory=lambda: _env_int("RAG_CHUNK_OVERLAP", 80)
    )


# ============================================================
#  Top-level Config
# ============================================================
@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    business: BusinessConfig = field(default_factory=BusinessConfig)


# ============================================================
#  Global singleton
# ============================================================
_config: Optional[Config] = None


def get_config(reload: bool = False) -> Config:
    """Get the global config singleton."""
    global _config
    if _config is None or reload:
        _config = Config()
    return _config


if __name__ == "__main__":
    c = get_config()
    print(f"LLM:    {c.llm.model} @ {c.llm.base_url[:50]}...")
    print(f"Key:    {c.llm.api_key[:10]}...")
    print(f"KB:     {c.business.knowledge_base_path}")
    print(f"Think:  {c.llm.enable_thinking}")
