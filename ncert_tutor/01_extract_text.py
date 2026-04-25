# Databricks notebook source
# MAGIC %pip install pdfplumber sentence-transformers faiss-cpu langchain-text-splitters openai streamlit langdetect requests
# MAGIC

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pdfplumber, re
from pyspark.sql.types import *

def extract_metadata_from_filename(filename):
    parts = filename.replace(".pdf", "").split("_")
    class_num = parts[0].replace("class", "")
    subject = "_".join(parts[1:-1])
    chapter = parts[-1].replace("ch", "")
    return class_num, subject, chapter

def extract_text_from_pdf(volume_path):
    local_path = volume_path.replace("dbfs:", "")  # This converts to /Volumes/...
    text_pages = []
    try:
        with pdfplumber.open(local_path) as pdf:  # pdfplumber can now read it
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                text = re.sub(r'\n{3,}', '\n\n', text)
                text = re.sub(r'[ \t]+', ' ', text)
                text = text.strip()
                if len(text) > 100:
                    text_pages.append((page_num + 1, text))
    except Exception as e:
        print(f"Error: {e}")
    return text_pages


schema = StructType([
    StructField("filename", StringType()),
    StructField("class_num", StringType()),
    StructField("subject", StringType()),
    StructField("chapter", StringType()),
    StructField("page_num", IntegerType()),
    StructField("page_text", StringType()),
])

rows = []
pdf_files = dbutils.fs.ls("/Volumes/workspace/default/ncert-tutor/")

for f in pdf_files:
    if not f.name.endswith(".pdf"):
        continue
    class_num, subject, chapter = extract_metadata_from_filename(f.name)
    pages = extract_text_from_pdf(f.path)
    for page_num, text in pages:
        rows.append((f.name, class_num, subject, chapter, page_num, text))

df_pages = spark.createDataFrame(rows, schema)
df_pages.write.format("delta").mode("overwrite").save("/Volumes/workspace/default/ncert-tutor/delta/raw_pages")

print(f"✅ Extracted {df_pages.count()} pages")


# COMMAND ----------

df_pages = spark.read.format("delta").load("/Volumes/workspace/default/ncert-tutor/delta/raw_pages")
print(f"✅ Total pages: {df_pages.count()}")
print("\nFirst 5 pages:")
df_pages.show(5, truncate=False)


# COMMAND ----------

databricks secrets put-acl \
  --scope ncert-tutor \
  --principal ch23b006@smail.iitm.ac.in \
  --permission READ