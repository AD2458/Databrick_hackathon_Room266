import sys
import os
import gradio as gr
import re
from pathlib import Path

# Point Python to src folder
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

# ── Config ────────────────────────────────────────────────────────────────────
# Try to use Unity Catalog Volume for index (persistent across deployments)
# Fall back to local if volume not accessible
VOLUME_INDEX_DIR = "/Volumes/workspace/default/ncert-tutor/index"
LOCAL_INDEX_DIR = str(_ROOT / "index")

# Check which index location is available
if Path(VOLUME_INDEX_DIR).exists() and (Path(VOLUME_INDEX_DIR) / "corpus.faiss").exists():
    INDEX_DIR = VOLUME_INDEX_DIR
    print(f"[App Startup] ✅ Using volume index: {INDEX_DIR}")
else:
    INDEX_DIR = LOCAL_INDEX_DIR
    print(f"[App Startup] ⚠️ Volume not accessible, using local index: {INDEX_DIR}")

os.environ.setdefault("NYAYA_INDEX_DIR", INDEX_DIR)

# Debug: Check if SARVAM_API_KEY is set
sarvam_key = os.environ.get("SARVAM_API_KEY", "")
if sarvam_key:
    print(f"[App Startup] ✅ SARVAM_API_KEY is set (length: {len(sarvam_key)}, starts with: {sarvam_key[:10]}...)")
else:
    print("[App Startup] ❌ WARNING: SARVAM_API_KEY is NOT set!")

# ── Build FAISS index if needed ───────────────────────────────────────────────
print(f"[App Startup] Checking FAISS index in: {INDEX_DIR}")
corpus_path = Path(INDEX_DIR) / "corpus.faiss"

if not corpus_path.exists():
    print("[App Startup] FAISS index not found. Building from chunks.parquet...")
    from build_index_startup import build_faiss_index_from_parquet
    build_faiss_index_from_parquet(Path(INDEX_DIR))
else:
    print(f"[App Startup] ✅ FAISS index found at {corpus_path}")

# ── Import modules ────────────────────────────────────────────────────────────
from retriever import get_retriever
from sarvam_client import chat_completions as _sarvam_chat
from quiz_generator import generate_quiz, check_answers

# ── Retriever (loaded once at startup) ───────────────────────────────────────
retriever = get_retriever()
print("[App Startup] ✅ Retriever loaded successfully")


# ── LLM using Sarvam AI only ──────────────────────────────────────────────────
def llm_chat(messages: list) -> tuple[str, str]:
    """Use Sarvam AI for chat completions. Returns (answer, model_name)."""
    try:
        print(f"[LLM] Calling Sarvam AI with {len(messages)} messages...")
        response = _sarvam_chat(messages=messages)
        print(f"[LLM] ✅ Sarvam AI responded successfully")
        return (response["choices"][0]["message"]["content"], "Sarvam AI")
    except Exception as sarvam_err:
        print(f"[LLM] ❌ Sarvam AI error: {sarvam_err}")
        raise RuntimeError(f"Sarvam AI failed: {sarvam_err}")


# ── Response parser ───────────────────────────────────────────────────────────
def parse_response(raw_response: str, model_name: str) -> str:
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
    print(f"[Query] Received: '{message}' (subject={subject}, class={student_class})")
    results_df = retriever.search(query=message, subject=subject, student_class=student_class, k=3)
    print(f"[Query] Retrieved {len(results_df)} results")
    
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
        print(f"[Query] ❌ Error: {e}")
        return f"Error: {e}"


# ── Quiz Functions ────────────────────────────────────────────────────────────
# Global state for current quiz
current_quiz = {"data": None}

def generate_quiz_ui(subject, student_class, topic, difficulty, num_questions):
    """Generate quiz and return formatted HTML with proper dark mode contrast"""
    if not topic.strip():
        return "<div style='color: #ff4444; padding: 10px;'>⚠️ Please enter a topic for the quiz.</div>", gr.update(visible=False), gr.update(visible=False)
    
    quiz_result = generate_quiz(
        retriever=retriever,
        subject=subject,
        student_class=student_class,
        topic=topic,
        difficulty=difficulty,
        num_questions=int(num_questions)
    )
    
    if not quiz_result.get("success"):
        error_msg = quiz_result.get("error", "Unknown error")
        return f"<div style='color: #ff4444; padding: 10px;'>❌ {error_msg}</div>", gr.update(visible=False), gr.update(visible=False)
    
    # Store current quiz in global state
    current_quiz["data"] = quiz_result
    
    # Format quiz as HTML with explicit colors for readability in dark mode
    quiz_html = f"""
    <div style="font-family: system-ui; max-width: 800px; color: var(--body-text-color);">
        <h2 style="color: var(--body-text-color);">📝 Quiz: {topic}</h2>
        <p style="color: var(--body-text-color);"><b>Class {student_class}</b> • <b>{subject.replace('_', ' ').title()}</b> • <b>Difficulty:</b> {difficulty}</p>
        <hr style="border-color: var(--border-color-primary);">
    """
    
    questions = quiz_result["questions"]
    for i, q in enumerate(questions):
        # Use contrasting colors: darker background with light text OR lighter background with dark text
        quiz_html += f"""
        <div style="margin: 25px 0; padding: 20px; background: rgba(100, 100, 255, 0.1); border: 1px solid rgba(100, 100, 255, 0.3); border-radius: 10px;">
            <p style="font-weight: bold; font-size: 1.15em; margin-bottom: 15px; color: var(--body-text-color);">
                {i+1}. {q['question']}
            </p>
        """
        
        # Create a unique ID for each question's radio group
        radio_name = f"question_{i}"
        for option in q['options']:
            option_letter = option[0]  # Get A, B, C, or D
            quiz_html += f"""
            <div style="margin: 10px 0; padding: 10px; background: rgba(255, 255, 255, 0.05); border-radius: 6px; transition: background 0.2s;">
                <label style="display: flex; align-items: center; cursor: pointer; color: var(--body-text-color);">
                    <input type="radio" name="{radio_name}" value="{option_letter}" 
                           style="margin-right: 12px; transform: scale(1.3); cursor: pointer;">
                    <span style="font-size: 1.05em;">{option}</span>
                </label>
            </div>
            """
        
        quiz_html += "</div>"
    
    quiz_html += "</div>"
    
    return quiz_html, gr.update(visible=True), gr.update(visible=True)


def submit_quiz_ui():
    """Check answers and display results with proper styling"""
    if current_quiz["data"] is None:
        return "<div style='color: #ff4444; padding: 10px;'>⚠️ No quiz available. Please generate a quiz first.</div>"
    
    result_html = """
    <div style="font-family: system-ui; max-width: 800px; padding: 20px; background: rgba(255, 193, 7, 0.15); border: 2px solid rgba(255, 193, 7, 0.4); border-radius: 10px;">
        <h3 style="color: var(--body-text-color); margin-top: 0;">📊 Quiz Answers</h3>
        <p style="color: var(--body-text-color);">Check your answers against the correct answers below:</p>
        <hr style="border-color: rgba(255, 193, 7, 0.4); margin: 15px 0;">
    """
    
    questions = current_quiz["data"]["questions"]
    for i, q in enumerate(questions):
        correct_answer = q["correct_answer"]
        explanation = q.get("explanation", "")
        result_html += f"""
        <div style="margin: 15px 0; padding: 15px; background: rgba(40, 167, 69, 0.1); border-left: 4px solid #28a745; border-radius: 6px;">
            <p style="color: var(--body-text-color); margin-bottom: 8px;"><b>Question {i+1}:</b> {q['question']}</p>
            <p style="color: #28a745; font-weight: bold; margin: 8px 0;">✓ Correct Answer: {correct_answer}</p>
            {f'<p style="color: var(--body-text-color); opacity: 0.85; font-style: italic; margin-top: 8px;">{explanation}</p>' if explanation else ''}
        </div>
        """
    
    result_html += "</div>"
    return result_html


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📚 Shiksha Sathi · शिक्षा साथी")
    gr.Markdown("Your AI tutor for Indian school textbooks - Ask questions or take a quiz!")
    
    with gr.Tabs():
        # Tab 1: Chat Interface
        with gr.Tab("💬 Ask Questions"):
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
        
        # Tab 2: Quiz Generator
        with gr.Tab("🎯 Quiz Generator"):
            gr.Markdown("### Generate a quiz on any topic from your textbook!")
            
            with gr.Row():
                quiz_subject = gr.Dropdown(
                    choices=["social_science", "science", "english"],
                    value="social_science",
                    label="Subject"
                )
                quiz_class = gr.Dropdown(
                    choices=["5", "6", "7", "8"],
                    value="6",
                    label="Class"
                )
            
            with gr.Row():
                quiz_topic = gr.Textbox(
                    label="Topic",
                    placeholder="e.g., Solar System, Indian Independence, Photosynthesis",
                    scale=2
                )
                quiz_difficulty = gr.Dropdown(
                    choices=["Easy", "Medium", "Hard"],
                    value="Medium",
                    label="Difficulty",
                    scale=1
                )
                quiz_num_questions = gr.Dropdown(
                    choices=["5", "10", "15"],
                    value="5",
                    label="Questions",
                    scale=1
                )
            
            generate_btn = gr.Button("🎲 Generate Quiz", variant="primary", size="lg")
            
            quiz_display = gr.HTML(label="Quiz", visible=True)
            
            with gr.Row(visible=False) as submit_row:
                submit_btn = gr.Button("✅ Show Answers", variant="secondary", size="lg")
            
            result_display = gr.HTML(visible=False)
            
            # Event handlers
            generate_btn.click(
                fn=generate_quiz_ui,
                inputs=[quiz_subject, quiz_class, quiz_topic, quiz_difficulty, quiz_num_questions],
                outputs=[quiz_display, submit_row, result_display]
            )
            
            submit_btn.click(
                fn=submit_quiz_ui,
                inputs=[],
                outputs=[result_display]
            ).then(
                lambda: gr.update(visible=True),
                outputs=[result_display]
            )

print("[App Startup] ✅ Gradio UI initialized")
# No share=True, no debug=True — Databricks Apps handles the server
demo.launch()
print("[App Startup] ✅ App launched successfully")
