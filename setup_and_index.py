# Databricks notebook source
# MAGIC %pip install faiss-cpu sentence-transformers gradio requests numpy pandas
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys, os, pandas as pd, numpy as np, pickle, faiss
from pathlib import Path


current_dir = os.getcwd()
sys.path.insert(0, os.path.join(current_dir, "src"))

from manifest import RAGManifest, utc_now_iso

# Load embeddings
with open("/Volumes/workspace/default/ncert-tutor/embeddings.pkl", "rb") as f:
    data = pickle.load(f)
embeddings = np.ascontiguousarray(data["embeddings"], dtype="float32")
print(f"✅ {len(embeddings)} embeddings loaded")

# Load chunks
df_chunks = spark.read.format("delta").load("/Volumes/workspace/default/ncert-tutor/delta/chunks")
chunks_pd = df_chunks.toPandas()
chunks_pd = chunks_pd.rename(columns={"class_num": "class", "content": "text"})
chunks_pd.insert(0, "faiss_id", range(len(chunks_pd)))

# Create a clean copy without any Spark metadata
cols = list(chunks_pd.columns)
chunks_pd = pd.DataFrame({col: chunks_pd[col].values for col in cols})
print(f"✅ {len(chunks_pd)} chunks loaded")

# Build FAISS index - using persistent Volume storage
OUTPUT_DIR = "/Volumes/workspace/default/ncert-tutor/index"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

faiss.normalize_L2(embeddings)
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

# Save all 3 files
faiss.write_index(index, f"{OUTPUT_DIR}/corpus.faiss")
print("✅ corpus.faiss saved")

chunks_pd.to_parquet(f"{OUTPUT_DIR}/chunks.parquet", index=False)
print("✅ chunks.parquet saved")

manifest = RAGManifest(
    embedding_model="distiluse-base-multilingual-cased-v2",
    embedding_dim=int(dim),
    faiss_index_file="corpus.faiss",
    chunks_parquet_file="chunks.parquet",
    num_vectors=len(embeddings),
    catalog="workspace",
    schema="default",
    source_table="ncert_chunks",
    created_at_utc=utc_now_iso(),
    normalize_embeddings=True,
    metric="inner_product",
)
Path(f"{OUTPUT_DIR}/manifest.json").write_text(manifest.to_json(), encoding="utf-8")
print("✅ manifest.json saved")
print(f"\n✅ Index built! {index.ntotal} vectors ready")