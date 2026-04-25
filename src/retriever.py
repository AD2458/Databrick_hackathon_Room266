"""Unified retriever interface for local NCERT FAISS index."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from retrieval import CorpusIndex
from embedder import get_embedder
from ncert_filter import filter_by_metadata

logger = logging.getLogger(__name__)

class NcertRetriever:
    """Wraps `CorpusIndex`, `SentenceEmbedder`, and `filter_by_metadata` into one interface."""

    def __init__(self, index_dir: str | Path) -> None:
        self._index_dir = str(index_dir)
        self._ci = None
        self._embedder = None

    def _load(self) -> None:
        if self._ci is not None:
            return
        
        logger.info("NcertRetriever: loading index from %s", self._index_dir)
        self._ci = CorpusIndex.load(self._index_dir)
        m = self._ci.manifest
        
        self._embedder = get_embedder(
            model_name=m.embedding_model,
            normalize=m.normalize_embeddings,
        )
        logger.info("NcertRetriever: loaded %d vectors, model %s", m.num_vectors, m.embedding_model)

    def search(self, query: str, subject: str, student_class: str, k: int = 3) -> pd.DataFrame:
        """Search the entire FAISS index and strictly filter by subject/class."""
        self._load()
        assert self._ci is not None and self._embedder is not None
        
        # 1. Embed the query
        emb = self._embedder.encode([query.strip()])
        
        # 2. Retrieve a wide net of semantic results (k=100) 
        # because the hard filter will drop anything outside the selected subject/class
        raw_semantic_df = self._ci.search(emb, k=100)

        # 3. Apply the strict metadata filter
        filtered_df = filter_by_metadata(
            semantic_results=raw_semantic_df,
            target_subject=subject,
            target_class=student_class,
            k=k
        )
        
        return filtered_df

def get_retriever() -> NcertRetriever:
    """Instantiate the configured retriever backend."""
    # Read the directory from environment, fallback to a standard Databricks/Linux temp dir
    index_dir = os.environ.get("NYAYA_INDEX_DIR", "/tmp/ncert_index").strip()
    
    logger.info("Using NcertRetriever (index_dir=%s)", index_dir)
    return NcertRetriever(index_dir)