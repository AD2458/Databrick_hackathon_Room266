"""Build FAISS index + chunk Parquet + manifest from embedded corpus rows."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from faiss_compat import get_faiss
from manifest import RAGManifest, utc_now_iso

if TYPE_CHECKING:
    from numpy.typing import NDArray

def _parquet_safe_str(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, (str, int, bool, np.integer, np.bool_)):
        return str(x)
    if isinstance(x, (float, np.floating)):
        return "" if np.isnan(float(x)) else str(float(x))
    return str(x)

def _sanitize_chunks_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    for col in out.columns:
        out[col] = [_parquet_safe_str(v) for v in out[col].tolist()]
    return out

def build_flat_ip_index(embeddings: "NDArray[np.float32]") -> Any:
    faiss = get_faiss()
    n, d = embeddings.shape
    index = faiss.IndexFlatIP(d)
    vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    return index

def save_rag_artifacts(
    output_dir: str | Path,
    embeddings: "NDArray[np.float32]",
    chunks_df: pd.DataFrame,
    embedding_model: str,
    catalog: str,
    schema: str,
    source_table: str,
    normalize_embeddings: bool = True,
) -> RAGManifest:
    
    faiss = get_faiss()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n, d = embeddings.shape
    d_int = int(d)
    n_int = int(n)
    if len(chunks_df) != n_int:
        raise ValueError(f"chunks_df rows ({len(chunks_df)}) != embeddings rows ({n_int})")

    chunks_df = _sanitize_chunks_df(chunks_df)

    index = build_flat_ip_index(embeddings)
    faiss_path = output_dir / "corpus.faiss"
    faiss.write_index(index, str(faiss_path))

    parquet_path = output_dir / "chunks.parquet"
    chunks_df.insert(0, "faiss_id", range(n_int))
    chunks_df.to_parquet(parquet_path, index=False)

    manifest = RAGManifest(
        embedding_model=str(embedding_model),
        embedding_dim=d_int,
        faiss_index_file=faiss_path.name,
        chunks_parquet_file=parquet_path.name,
        num_vectors=n_int,
        catalog=str(catalog),
        schema=str(schema),
        source_table=str(source_table),
        created_at_utc=utc_now_iso(),
        normalize_embeddings=normalize_embeddings,
        metric="inner_product",
    )
    mp = output_dir / "manifest.json"
    mp.write_text(manifest.to_json(), encoding="utf-8")
    return manifest