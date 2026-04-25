"""Build FAISS index from chunks.parquet at app startup by computing embeddings."""
import os
import json
from pathlib import Path
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def build_faiss_index_from_parquet(index_dir: Path):
    """Build FAISS index by computing embeddings from text in chunks.parquet."""
    print("[Startup] Building FAISS index from chunks.parquet...")
    
    chunks_path = index_dir / "chunks.parquet"
    corpus_path = index_dir / "corpus.faiss"
    manifest_path = index_dir / "manifest.json"
    
    # Check if already built
    if corpus_path.exists():
        print(f"[Startup] FAISS index already exists at {corpus_path}")
        return
    
    # Load manifest for configuration
    print(f"[Startup] Loading configuration from {manifest_path}...")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    embedding_model_name = manifest.get("embedding_model", "distiluse-base-multilingual-cased-v2")
    embedding_dim = manifest.get("embedding_dim", 512)
    normalize = manifest.get("normalize_embeddings", True)
    
    # Load embedding model
    print(f"[Startup] Loading embedding model: {embedding_model_name}...")
    model = SentenceTransformer(embedding_model_name)
    
    # Load chunks
    print(f"[Startup] Loading chunks from {chunks_path}...")
    df = pd.read_parquet(chunks_path)
    texts = df['text'].tolist()
    print(f"[Startup] Loaded {len(texts)} text chunks")
    
    # Compute embeddings (batch processing for speed)
    print(f"[Startup] Computing embeddings for {len(texts)} chunks...")
    batch_size = 32
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
        convert_to_numpy=True
    )
    
    embeddings = embeddings.astype('float32')
    print(f"[Startup] Computed embeddings with shape {embeddings.shape}")
    
    # Build FAISS index
    print("[Startup] Building FAISS index...")
    if normalize:
        # For normalized embeddings, use inner product (cosine similarity)
        index = faiss.IndexFlatIP(embedding_dim)
    else:
        index = faiss.IndexFlatL2(embedding_dim)
    
    index.add(embeddings)
    
    # Save index
    print(f"[Startup] Saving FAISS index to {corpus_path}...")
    faiss.write_index(index, str(corpus_path))
    
    print(f"[Startup] ✅ FAISS index built successfully with {index.ntotal} vectors")

if __name__ == "__main__":
    index_dir = Path(__file__).parent / "index"
    build_faiss_index_from_parquet(index_dir)
