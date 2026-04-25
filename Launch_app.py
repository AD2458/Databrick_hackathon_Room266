# Databricks notebook source
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install faiss-cpu sentence-transformers gradio requests numpy pandas

# COMMAND ----------

from sarvam_client import chat_completions, translate_text

# COMMAND ----------

# MAGIC %pip install gradio

# COMMAND ----------

# DBTITLE 1,Cell 5
import sys
import os
import gradio as gr
import re
from openai import OpenAI

# Point Python to your src folder
current_dir = os.getcwd()
sys.path.insert(0, os.path.join(current_dir, "src"))

# Clear cached modules to ensure fresh imports
for mod in ['llm_client', 'sarvam_client', 'retriever', 'ncert_filter']:
    if mod in sys.modules:
        del sys.modules[mod]

# ── Secrets (safe for GitHub) ─────────────────────────────────────────────────
os.environ["SARVAM_API_KEY"] = dbutils.secrets.get(scope="ncert-tutor", key="sarvam-api-key")
os.environ["NYAYA_INDEX_DIR"] = "/Volumes/workspace/default/ncert-tutor/index"

# Databricks token for OpenAI client
os.environ["DATABRICKS_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

from retriever import get_retriever
from sarvam_client import chat_completions as _sarvam_chat

# ── OpenAI client for Databricks GPT-5.4 ──────────────────────────────────────
client = OpenAI(
    api_key=os.environ["DATABRICKS_TOKEN"],
    base_url="https://7474657381434754.ai-gateway.cloud.databricks.com/mlflow/v1"
)

# ── Retriever ─────────────────────────────────────────────────────────────────
retriever = get_retriever()


# ── LLM with fallback (returns answer + model name) ───────────────────────────
def llm_chat(messages: list) -> tuple[str, str]:
    """Try Databricks GPT-5.4 first, fall back to Sarvam.
    Returns: (answer, model_name)
    """
    try:
        response = client.chat.completions.create(
            model="databricks-gpt-5-4",
            messages=messages,
            max_tokens=5000,
            temperature=0.2
        )
        return (response.choices[0].message.content, "Databricks GPT-5.4")
    except Exception as db_err:
        print(f"[Databricks LLM failed, falling back to Sarvam] {db_err}")
        try:
            response = _sarvam_chat(messages=messages)
            return (response["choices"][0]["message"]["content"], "Sarvam AI")
        except Exception as sarvam_err:
            raise RuntimeError(f"Both LLMs failed. Databricks: {db_err} | Sarvam: {sarvam_err}")


# ── Response parser (handles <think> tags + shows model name) ─────────────────
def parse_response(raw_response: str, model_name: str) -> str:
    """Parse response with thinking tags and add model badge."""
    # Add model badge at the top
    model_badge = f'<div style="font-size: 0.85em; color: #6b7280; margin-bottom: 10px;">⚡ Powered by <b>{model_name}</b></div>\n\n'
    
    think_match = re.search(r'<think>(.*?)</think>', raw_response, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
        return model_badge + f"""<details>
<summary><b>💭 Show Thinking Process</b></summary>
<div style="padding: 12px; border-left: 3px solid #3b82f6; margin-top: 10px; opacity: 0.85;">
<em>{thinking}</em>
</div>
</details>

---

{answer}"""
    return model_badge + raw_response


# ── Core pipeline ─────────────────────────────────────────────────────────────
def process_query(message, history, subject, student_class):
    results_df = retriever.search(query=message, subject=subject, student_class=student_class, k=3)

    if results_df.empty:
        return f"I couldn't find any information in the Class {student_class} {subject} textbook for that."

    context_text = "\n\n".join(results_df["text"].tolist())

    system_prompt = (
        "You are Shiksha Sathi, a helpful Indian school tutor. "
        "Answer the student's question using ONLY the provided textbook context. "
        "If the answer isn't in the context, say so clearly."
    )
    user_prompt = f"Textbook Context:\n{context_text}\n\nStudent Question: {message}"

    try:
        raw_answer, model_used = llm_chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        return parse_response(raw_answer, model_used)
    except Exception as e:
        return f"Error: {e}"


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📚 Shiksha Sathi · शिक्षा साथी")
    gr.Markdown("Select your subject and class, then ask a question!")

    with gr.Row():
        subject_dropdown = gr.Dropdown(
            choices=["social_science", "science", "english"],
            value="social_science",
            label="Subject"
        )
        class_dropdown = gr.Dropdown(
            choices=["5", "6", "7", "8"],
            value="6",
            label="Class"
        )

    chat = gr.ChatInterface(
        fn=process_query,
        additional_inputs=[subject_dropdown, class_dropdown],
    )

demo.launch(share=True, debug=True)