# Databricks notebook source
# MAGIC %pip install pdfplumber sentence-transformers faiss-cpu langchain-text-splitters openai streamlit langdetect requests
# MAGIC

# COMMAND ----------

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pyspark.sql.functions import *
from pyspark.sql.types import *

df_pages = spark.read.format("delta").load("/Volumes/workspace/default/ncert-tutor/delta/raw_pages")

chapter_texts = df_pages.groupBy("class_num", "subject", "chapter", "filename") \
    .agg(collect_list(struct("page_num", "page_text")).alias("pages"))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "],
)

chunk_rows = []
for row in chapter_texts.collect():
    sorted_pages = sorted(row["pages"], key=lambda x: x["page_num"])
    full_text = "\n\n".join([p["page_text"] for p in sorted_pages])
    chunks = splitter.split_text(full_text)
    
    for i, chunk in enumerate(chunks):
        chunk_id = f"c{row['class_num']}_{row['subject']}_ch{row['chapter']}_{i}"
        chunk_rows.append({
            "chunk_id": chunk_id,
            "content": chunk,
            "class_num": row["class_num"],
            "subject": row["subject"],
            "chapter": row["chapter"],
            "chunk_sequence": i,
            "source_file": row["filename"],
            "word_count": len(chunk.split()),
        })

chunk_schema = StructType([
    StructField("chunk_id", StringType()),
    StructField("content", StringType()),
    StructField("class_num", StringType()),
    StructField("subject", StringType()),
    StructField("chapter", StringType()),
    StructField("chunk_sequence", IntegerType()),
    StructField("source_file", StringType()),
    StructField("word_count", IntegerType()),
])

df_chunks = spark.createDataFrame(chunk_rows, chunk_schema)
df_chunks.write.format("delta").mode("overwrite") \
    .partitionBy("class_num", "subject") \
    .save("/Volumes/workspace/default/ncert-tutor/delta/chunks")

print(f"✅ Created {df_chunks.count()} chunks")
