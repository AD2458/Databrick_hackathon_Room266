# Databricks notebook source
# MAGIC %pip install pdfplumber sentence-transformers faiss-cpu langchain-text-splitters openai streamlit langdetect requests

# COMMAND ----------



from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

# Load lighter model for Free Edition
model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

# Load chunks
df_chunks = spark.read.format("delta").load("/Volumes/workspace/default/ncert-tutor/delta/chunks")
chunks_list = df_chunks.select("chunk_id", "content").collect()

chunk_ids = [row["chunk_id"] for row in chunks_list]
texts = [row["content"] for row in chunks_list]

print(f"Embedding {len(texts)} chunks...")
embeddings = model.encode(texts, batch_size=16, show_progress_bar=True)

# Save embeddings
data = {"chunk_ids": chunk_ids, "embeddings": embeddings}
with open("/Volumes/workspace/default/ncert-tutor/embeddings.pkl", "wb") as f:
    pickle.dump(data, f)

print(f"✅ Saved {len(embeddings)} embeddings, shape: {embeddings.shape}")


# COMMAND ----------

import faiss, pickle
import numpy as np

with open("/Volumes/workspace/default/ncert-tutor/embeddings.pkl", "rb") as f:
    data = pickle.load(f)

embeddings = np.array(data["embeddings"]).astype("float32")
chunk_ids = data["chunk_ids"]

faiss.normalize_L2(embeddings)
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

faiss.write_index(index, "/Volumes/workspace/default/ncert-tutor/ncert.index")
with open("/Volumes/workspace/default/ncert-tutor/chunk_id_order.pkl", "wb") as f:
    pickle.dump(chunk_ids, f)

print(f"✅ FAISS index built: {index.ntotal} vectors")


# COMMAND ----------

files = dbutils.fs.ls("/Volumes/workspace/default/ncert-tutor/")
print("✅ Files in ncert-tutor Volume:")
for f in sorted([f.name for f in files]):
    print(f"  {f}")
