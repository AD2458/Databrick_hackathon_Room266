"""Lazy import faiss so pip install in a notebook works without kernel restart."""

from __future__ import annotations
from typing import Any

def get_faiss() -> Any:
    try:
        import faiss  # type: ignore[import-untyped]
        return faiss
    except ImportError as e:
        raise ImportError(
            "faiss (faiss-cpu) is not importable. Install with:\n"
            "  pip install 'faiss-cpu>=1.7.0,<1.8' 'numpy>=1.24,<2' 'pandas>=2.0,<3'\n"
        ) from e