# Databricks notebook source
# MAGIC %pip install streamlit==1.31.0

# COMMAND ----------

# DBTITLE 1,Launch Gradio App
import sys
import os
import gradio as gr
import re
from pathlib import Path

# Point Python to src folder - use explicit path for notebook environment
_ROOT = Path("/Workspace/Users/ch23b006@smail.iitm.ac.in/R266")
sys.path.insert(0, str(_ROOT / "src"))

# ── Config ────────────────────────────────────────────────────────────────────
# Use local index directory
os.environ.setdefault("NYAYA_INDEX_DIR", str(_ROOT / "index"))

# Debug: Check if SARVAM_API_KEY is set
sarvam_key = os.environ.get("SARVAM_API_KEY", "")
if sarvam_key:
    print(f"[App Startup] ✅ SARVAM_API_KEY is set (length: {len(sarvam_key)}, starts with: {sarvam_key[:10]}...)")
else:
    print("[App Startup] ❌ WARNING: SARVAM_API_KEY is NOT set!")

# ── Build FAISS index from chunks.parquet at startup ─────────────────────────
print("[App Startup] Checking FAISS index...")
index_dir = _ROOT / "index"
corpus_path = index_dir / "corpus.faiss"

if not corpus_path.exists():
    print("[App Startup] FAISS index not found. Building from chunks.parquet...")
    from build_index_startup import build_faiss_index_from_parquet
    build_faiss_index_from_parquet(index_dir)
else:
    print(f"[App Startup] FAISS index found at {corpus_path}")

# ── Import modules after index is built ───────────────────────────────────────
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
    """Generate quiz and return formatted HTML"""
    if not topic.strip():
        return "<div style='color: red;'>⚠️ Please enter a topic for the quiz.</div>", gr.update(visible=False), gr.update(visible=False)
    
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
        return f"<div style='color: red;'>❌ {error_msg}</div>", gr.update(visible=False), gr.update(visible=False)
    
    # Store current quiz in global state
    current_quiz["data"] = quiz_result
    
    # Format quiz as HTML
    quiz_html = f"""
    <div style="font-family: system-ui; max-width: 800px;">
        <h2>📝 Quiz: {topic}</h2>
        <p><b>Class {student_class}</b> • <b>{subject.replace('_', ' ').title()}</b> • <b>Difficulty:</b> {difficulty}</p>
        <hr>
    """
    
    questions = quiz_result["questions"]
    for i, q in enumerate(questions):
        quiz_html += f"""
        <div style="margin: 25px 0; padding: 15px; background: #f8f9fa; border-radius: 8px;">
            <p style="font-weight: bold; font-size: 1.1em; margin-bottom: 12px;">
                {i+1}. {q['question']}
            </p>
        """
        
        # Create a unique ID for each question's radio group
        radio_name = f"question_{i}"
        for option in q['options']:
            option_letter = option[0]  # Get A, B, C, or D
            quiz_html += f"""
            <div style="margin: 8px 0;">
                <label style="display: flex; align-items: center; cursor: pointer;">
                    <input type="radio" name="{radio_name}" value="{option_letter}" 
                           style="margin-right: 10px; transform: scale(1.2);">
                    <span>{option}</span>
                </label>
            </div>
            """
        
        quiz_html += "</div>"
    
    quiz_html += "</div>"
    
    return quiz_html, gr.update(visible=True), gr.update(visible=True)


def submit_quiz_ui():
    """Check answers and display results"""
    if current_quiz["data"] is None:
        return "<div style='color: red;'>⚠️ No quiz available. Please generate a quiz first.</div>"
    
    # Extract user answers from JavaScript (we'll use a simple approach)
    # Since we can't directly access form data in Gradio, we'll create a simpler approach
    # For now, let's show instructions to improve this
    
    result_html = """
    <div style="font-family: system-ui; max-width: 800px; padding: 20px; background: #fff3cd; border-radius: 8px;">
        <h3>ℹ️ Note about Quiz Submission</h3>
        <p>Due to Gradio limitations, answer checking will be enhanced in the next version.</p>
        <p>For now, check your answers manually against these correct answers:</p>
        <hr>
    """
    
    questions = current_quiz["data"]["questions"]
    for i, q in enumerate(questions):
        correct_answer = q["correct_answer"]
        explanation = q.get("explanation", "")
        result_html += f"""
        <div style="margin: 15px 0; padding: 12px; background: white; border-left: 4px solid #28a745; border-radius: 4px;">
            <p><b>Question {i+1}:</b> {q['question']}</p>
            <p style="color: #28a745;"><b>✓ Correct Answer: {correct_answer}</b></p>
            {f'<p style="color: #666;"><i>{explanation}</i></p>' if explanation else ''}
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
                submit_btn = gr.Button("✅ Submit Answers", variant="secondary", size="lg")
            
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

# COMMAND ----------

"""Shiksha Sathi — Sovereign Voice Educational Tutor
Sarvam AI + Databricks | 22 Indian Languages | 100% India-Hosted

Architecture: STT (Sarvam Saaras, India) -> LLM (Sarvam-M/30B, India) -> TTS (Sarvam Bulbul, India)
"""

import os, re, json, base64, logging, sys, requests, io

# Disable Gradio analytics and external connections
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"

import gradio as gr

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.info(f"Python: {sys.version}, Gradio: {gr.__version__}")

# ── Secrets & Config ──────────────────────────────────────────────────────────
try:
    os.environ["SARVAM_API_KEY"] = dbutils.secrets.get(scope="ncert-tutor", key="sarvam-api-key")
    os.environ["DATABRICKS_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    # Get the Databricks host from the current context
    os.environ["DATABRICKS_HOST"] = "https://" + dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
except Exception as e:
    logger.warning(f"Could not load secrets: {e}. Using environment variables if set.")

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST", "")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
SARVAM_ENDPOINT = os.environ.get("SARVAM_ENDPOINT_NAME", "sarvam-30b-serving")

logger.info(f"Sarvam API Key: {'✓ Set' if SARVAM_API_KEY else '✗ Missing'}")
logger.info(f"Databricks Host: {DATABRICKS_HOST if DATABRICKS_HOST else '✗ Missing'}")
logger.info(f"Databricks Token: {'✓ Set' if DATABRICKS_TOKEN else '✗ Missing'}")

SYSTEM_PROMPT = """You are Shiksha Sathi, a friendly, encouraging, and highly accurate educational tutor for Indian school students.
Respond in the SAME language the user speaks (Hindi, English, Tamil, Telugu, Hinglish etc.).
Be concise but highly instructive (3-4 sentences max for voice-friendly responses).
Help with: explaining textbook concepts (Science, Math, History, Geography), answering curriculum questions, and providing study tips.
Break down complex topics into simple, easy-to-understand parts. Do not just give the answer; explain the 'why'."""

LANG_MAP = {
    "hi": "hi-IN", "en": "en-IN", "ta": "ta-IN", "te": "te-IN",
    "kn": "kn-IN", "ml": "ml-IN", "bn": "bn-IN", "gu": "gu-IN",
    "mr": "mr-IN", "pa": "pa-IN", "od": "od-IN",
}

LANG_NAMES = {
    "hi": "Hindi", "en": "English", "ta": "Tamil", "te": "Telugu",
    "kn": "Kannada", "ml": "Malayalam", "bn": "Bengali", "gu": "Gujarati",
    "mr": "Marathi", "pa": "Punjabi", "od": "Odia",
}

conversation = []


def call_llm(user_message):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, b in conversation:
        messages.append({"role": "user", "content": u})
        if b:
            messages.append({"role": "assistant", "content": b})
    messages.append({"role": "user", "content": user_message})

    if DATABRICKS_HOST and DATABRICKS_TOKEN:
        try:
            resp = requests.post(
                f"{DATABRICKS_HOST}/serving-endpoints/{SARVAM_ENDPOINT}/invocations",
                headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"},
                json={"messages": messages, "max_tokens": 400, "temperature": 0.3}, timeout=30)
            if resp.status_code == 200:
                c = resp.json()["choices"][0]["message"]["content"]
                return re.sub(r"<think>.*?</think>", "", c, flags=re.DOTALL).strip()
            else:
                logger.warning(f"Databricks endpoint returned {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"Databricks endpoint fallback: {e}")

    # Fallback to Sarvam
    if not SARVAM_API_KEY:
        return "Error: No API keys configured. Please set SARVAM_API_KEY."
    
    resp = requests.post("https://api.sarvam.ai/v1/chat/completions", headers={
        "api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json",
    }, json={"model": "sarvam-m", "messages": messages, "max_tokens": 400, "temperature": 0.3}, timeout=30)
    
    if resp.status_code != 200:
        raise Exception(f"LLM Error {resp.status_code}: {resp.text}")
    c = resp.json()["choices"][0]["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", c, flags=re.DOTALL).strip()


def clean_for_tts(text):
    """Strip markdown/HTML so TTS doesn't read formatting characters aloud."""
    t = re.sub(r'<[^>]+>', '', text)
    t = re.sub(r'\*\*(.+?)\*\*', r'\1', t)
    t = re.sub(r'\*(.+?)\*', r'\1', t)
    t = re.sub(r'__(.+?)__', r'\1', t)
    t = re.sub(r'_(.+?)_', r'\1', t)
    t = re.sub(r'~~(.+?)~~', r'\1', t)
    t = re.sub(r'`(.+?)`', r'\1', t)
    t = re.sub(r'^#{1,6}\s+', '', t, flags=re.MULTILINE)
    t = re.sub(r'^\s*[-*+]\s+', '', t, flags=re.MULTILINE)
    t = re.sub(r'^\s*\d+\.\s+', '', t, flags=re.MULTILINE)
    t = re.sub(r'^\s*>\s*', '', t, flags=re.MULTILINE)
    t = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', t)
    t = re.sub(r'[|]', ' ', t)
    t = re.sub(r'-{3,}', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def get_tts_html(text, lang="en"):
    if not SARVAM_API_KEY:
        return "<p style='color:#999'><em>TTS unavailable: No API key</em></p>"
    
    lang_code = LANG_MAP.get(lang[:2] if lang else "en", "en-IN")
    clean_text = clean_for_tts(text)
    try:
        resp = requests.post("https://api.sarvam.ai/text-to-speech", headers={
            "api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json",
        }, json={
            "inputs": [clean_text[:500]], "target_language_code": lang_code,
            "speaker": "anushka", "model": "bulbul:v2",
            "pitch": 0, "pace": 1.0, "loudness": 1.5, "enable_preprocessing": True,
        }, timeout=30)
        if resp.status_code == 200:
            ab = resp.json().get("audios", [None])[0]
            if ab:
                return f'<audio controls autoplay style="width:100%" src="data:audio/wav;base64,{ab}"></audio>'
    except Exception as e:
        logger.error(f"TTS error: {e}")
    return "<p style='color:#999'><em>Audio generation failed</em></p>"


def do_stt(audio_b64, filename="audio.webm"):
    if not SARVAM_API_KEY:
        raise Exception("STT Error: No API key configured")
    
    headers = {"api-subscription-key": SARVAM_API_KEY}
    audio_bytes = base64.b64decode(audio_b64)
    files = {"file": (filename, io.BytesIO(audio_bytes), "audio/webm")}
    data = {"model": "saaras:v3", "language_code": "unknown", "with_timestamps": "false"}
    resp = requests.post("https://api.sarvam.ai/speech-to-text", headers=headers, files=files, data=data, timeout=30)
    if resp.status_code != 200:
        raise Exception(f"STT Error: {resp.text}")
    r = resp.json()
    return r.get("transcript", ""), r.get("language_code", "en")


def detect_lang(text):
    if any('\u0900' <= c <= '\u097F' for c in text):
        return "hi"
    if any('\u0B80' <= c <= '\u0BFF' for c in text):
        return "ta"
    if any('\u0C00' <= c <= '\u0C7F' for c in text):
        return "te"
    if any('\u0C80' <= c <= '\u0CFF' for c in text):
        return "kn"
    if any('\u0D00' <= c <= '\u0D7F' for c in text):
        return "ml"
    if any('\u0980' <= c <= '\u09FF' for c in text):
        return "bn"
    return "en"


def format_chat():
    if not conversation:
        return """*Start learning! Try asking a question in Hindi, English, Tamil, or any Indian language.*

**Example queries:**
- "Explain the process of photosynthesis."
- "What were the main causes of the 1857 revolt?"
- "Solve this: If a car travels 60km in 2 hours, what is its speed?"
"""
    lines = []
    for u, b in conversation:
        lines.append(f'> **Student:** {u}')
        lines.append(f'**Shiksha Sathi:** {b}')
        lines.append('')
    return '\n\n'.join(lines)


def handle_text(user_text):
    global conversation
    if not user_text.strip():
        return "", format_chat(), "", ""
    try:
        reply = call_llm(user_text)
        lang = detect_lang(user_text)
        audio = get_tts_html(reply, lang)
        conversation.append((user_text, reply))
        lang_name = LANG_NAMES.get(lang, lang)
        return "", format_chat(), audio, f"Language: {lang_name}"
    except Exception as e:
        logger.error(f"Text error: {e}")
        conversation.append((user_text, f"Sorry, error occurred: {e}"))
        return "", format_chat(), "", f"Error: {e}"


def handle_voice_b64(audio_data_str):
    global conversation
    if not audio_data_str or not audio_data_str.strip():
        return format_chat(), "", "No audio data received. Please record first, then click Send Voice."
    try:
        if "base64," in audio_data_str:
            audio_data_str = audio_data_str.split("base64,")[1]
        user_text, lang = do_stt(audio_data_str)
        if not user_text.strip():
            return format_chat(), "", "Could not understand the audio. Please try again."
        reply = call_llm(user_text)
        audio = get_tts_html(reply, lang)
        conversation.append((user_text, reply))
        lang_name = LANG_NAMES.get(lang[:2], lang)
        return format_chat(), audio, f'Heard: "{user_text}" | Language: {lang_name}'
    except Exception as e:
        logger.error(f"Voice error: {e}")
        return format_chat(), "", f"Error: {e}"


def clear_all():
    global conversation
    conversation = []
    return format_chat(), "", "", "", "", ""


logger.info("Building UI...")

HEAD_JS = """
<script>
let sathiRecorder = null;
let sathiChunks = [];
let sathiRecording = false;
let sathiTimer = null;
let sathiSecs = 0;

async function sathiStartRec() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        sathiRecorder = new MediaRecorder(stream);
        sathiChunks = [];
        sathiRecorder.ondataavailable = (e) => { if (e.data.size > 0) sathiChunks.push(e.data); };
        sathiRecorder.onstop = () => {
            const blob = new Blob(sathiChunks, { type: 'audio/webm' });
            const reader = new FileReader();
            reader.onloadend = () => {
                const b64 = reader.result;
                const el = document.querySelector('#voice-data-box textarea');
                if (el) {
                    const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
                    setter.call(el, b64);
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                }
                document.getElementById('rec-status-text').textContent = 'Recording saved! Click "Send Voice" to process.';
            };
            reader.readAsDataURL(blob);
            stream.getTracks().forEach(t => t.stop());
        };
        sathiRecorder.start();
        sathiRecording = true;
        sathiSecs = 0;
        document.getElementById('rec-indicator').style.display = 'flex';
        document.getElementById('rec-status-text').textContent = 'Recording...';
        sathiTimer = setInterval(() => {
            sathiSecs++;
            const m = String(Math.floor(sathiSecs/60)).padStart(2,'0');
            const s = String(sathiSecs%60).padStart(2,'0');
            document.getElementById('rec-time').textContent = m + ':' + s;
        }, 1000);
    } catch (err) {
        document.getElementById('rec-status-text').textContent = 'Microphone access denied: ' + err.message;
    }
}

function sathiStopRec() {
    if (sathiRecorder && sathiRecorder.state !== 'inactive') {
        sathiRecorder.stop();
    }
    sathiRecording = false;
    clearInterval(sathiTimer);
    document.getElementById('rec-indicator').style.display = 'none';
}
</script>
"""

app = gr.Blocks(title="Shiksha Sathi Voice Tutor", head=HEAD_JS)
with app:
    gr.HTML("""<div style="text-align:center;margin-bottom:12px">
        <h1 style="color:#10B981;margin:0;font-size:2em">🎓 Shiksha Sathi</h1>
        <p style="color:#555;margin:2px 0;font-size:1.1em">Sovereign Voice Educational Tutor</p>
        <p style="font-size:12px;color:#888;margin:2px 0">Powered by <b>Sarvam AI</b> + <b>Databricks</b> | 22 Indian Languages</p>
        <div style="background:linear-gradient(90deg,#FF9933 33%,#FFFFFF 33%,#FFFFFF 66%,#138808 66%);padding:4px 16px;border-radius:20px;display:inline-block;font-size:11px;font-weight:bold;color:#333;margin-top:4px">
            100% India-Hosted &mdash; Full Data Sovereignty
        </div></div>""")

    chat_display = gr.Markdown(value=format_chat())

    with gr.Tab("Text"):
        with gr.Row():
            text_input = gr.Textbox(
                label="Ask Shiksha Sathi",
                placeholder="e.g. How does the heart pump blood? / Explain gravity in Hindi",
                scale=4, lines=1)
            text_btn = gr.Button("Send", variant="primary", scale=1)
        text_audio_html = gr.HTML()
        text_status = gr.Textbox(label="Status", interactive=False, max_lines=1)

    with gr.Tab("Voice"):
        gr.HTML("""<div style="text-align:center;padding:16px">
            <p style="color:#555;margin:0 0 8px 0">Use the buttons below to record your textbook question in any Indian language</p>
            <div id="rec-indicator" style="display:none;align-items:center;justify-content:center;gap:10px;margin:8px 0">
                <span style="display:inline-block;width:12px;height:12px;background:#dc3545;border-radius:50%;animation:blink 1s infinite"></span>
                <span id="rec-time" style="font-size:22px;font-weight:bold;color:#dc3545">00:00</span>
            </div>
            <p id="rec-status-text" style="color:#888;margin:4px 0;font-size:13px">Click "Record" to start</p>
            <style>@keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}</style>
        </div>""")
        with gr.Row():
            rec_btn = gr.Button("🎙 Record", variant="secondary", scale=1)
            stop_btn = gr.Button("⏹ Stop", variant="stop", scale=1)
        voice_data = gr.Textbox(label="Audio Data", visible=False, elem_id="voice-data-box")
        voice_btn = gr.Button("Send Voice", variant="primary")
        voice_audio_html = gr.HTML()
        voice_status = gr.Textbox(label="Status", interactive=False, max_lines=1)

    clear_btn = gr.Button("Clear Conversation", variant="secondary")

    gr.Markdown("""---
| Component | Provider | Location |
|-----------|----------|----------|
| **Speech-to-Text** | Sarvam Saaras v3 | India |
| **LLM Brain** | Sarvam-M / Sarvam-30B on Databricks | India |
| **Text-to-Speech** | Sarvam Bulbul v2 | India |
| **Platform** | Databricks | Azure Central India |

*Zero data leaves India. Supports: Hindi, English, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati, Marathi, Punjabi, Odia*""")

    text_btn.click(handle_text, [text_input], [text_input, chat_display, text_audio_html, text_status])
    text_input.submit(handle_text, [text_input], [text_input, chat_display, text_audio_html, text_status])
    rec_btn.click(fn=None, inputs=None, outputs=None, js="() => { sathiStartRec(); }")
    stop_btn.click(fn=None, inputs=None, outputs=None, js="() => { sathiStopRec(); }")
    voice_btn.click(handle_voice_b64, [voice_data], [chat_display, voice_audio_html, voice_status])
    clear_btn.click(clear_all, outputs=[chat_display, text_audio_html, voice_audio_html, text_status, voice_status, voice_data])

logger.info("UI built successfully")
print("\n" + "="*60)
print("🚀 Launching Shiksha Sathi on port 7001...")
print("="*60)

app.launch(share=True)

# COMMAND ----------

"""Shiksha Sathi — Sovereign Voice Educational Tutor
Sarvam AI + Databricks | 22 Indian Languages | 100% India-Hosted

Architecture: STT (Sarvam Saaras, India) -> LLM (Sarvam-M/30B, India) -> TTS (Sarvam Bulbul, India)
"""

import os, re, json, base64, logging, sys, requests, io
import shutil
import stat

# Disable Gradio analytics and external connections
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"

import gradio as gr

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.info(f"Python: {sys.version}, Gradio: {gr.__version__}")

# ── Secrets & Config ──────────────────────────────────────────────────────────
try:
    os.environ["SARVAM_API_KEY"] = dbutils.secrets.get(scope="ncert-tutor", key="sarvam-api-key")
    os.environ["DATABRICKS_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    # Get the Databricks host from the current context
    os.environ["DATABRICKS_HOST"] = "https://" + dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
except Exception as e:
    logger.warning(f"Could not load secrets: {e}. Using environment variables if set.")

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST", "")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
SARVAM_ENDPOINT = os.environ.get("SARVAM_ENDPOINT_NAME", "sarvam-30b-serving")

logger.info(f"Sarvam API Key: {'✓ Set' if SARVAM_API_KEY else '✗ Missing'}")
logger.info(f"Databricks Host: {DATABRICKS_HOST if DATABRICKS_HOST else '✗ Missing'}")
logger.info(f"Databricks Token: {'✓ Set' if DATABRICKS_TOKEN else '✗ Missing'}")

SYSTEM_PROMPT = """You are Shiksha Sathi, a friendly, encouraging, and highly accurate educational tutor for Indian school students.
Respond in the SAME language the user speaks (Hindi, English, Tamil, Telugu, Hinglish etc.).
Be concise but highly instructive (3-4 sentences max for voice-friendly responses).
Help with: explaining textbook concepts (Science, Math, History, Geography), answering curriculum questions, and providing study tips.
Break down complex topics into simple, easy-to-understand parts. Do not just give the answer; explain the 'why'."""

LANG_MAP = {
    "hi": "hi-IN", "en": "en-IN", "ta": "ta-IN", "te": "te-IN",
    "kn": "kn-IN", "ml": "ml-IN", "bn": "bn-IN", "gu": "gu-IN",
    "mr": "mr-IN", "pa": "pa-IN", "od": "od-IN",
}

LANG_NAMES = {
    "hi": "Hindi", "en": "English", "ta": "Tamil", "te": "Telugu",
    "kn": "Kannada", "ml": "Malayalam", "bn": "Bengali", "gu": "Gujarati",
    "mr": "Marathi", "pa": "Punjabi", "od": "Odia",
}

conversation = []


def call_llm(user_message):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, b in conversation:
        messages.append({"role": "user", "content": u})
        if b:
            messages.append({"role": "assistant", "content": b})
    messages.append({"role": "user", "content": user_message})

    if DATABRICKS_HOST and DATABRICKS_TOKEN:
        try:
            resp = requests.post(
                f"{DATABRICKS_HOST}/serving-endpoints/{SARVAM_ENDPOINT}/invocations",
                headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"},
                json={"messages": messages, "max_tokens": 400, "temperature": 0.3}, timeout=30)
            if resp.status_code == 200:
                c = resp.json()["choices"][0]["message"]["content"]
                return re.sub(r"<think>.*?</think>", "", c, flags=re.DOTALL).strip()
            else:
                logger.warning(f"Databricks endpoint returned {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"Databricks endpoint fallback: {e}")

    # Fallback to Sarvam
    if not SARVAM_API_KEY:
        return "Error: No API keys configured. Please set SARVAM_API_KEY."
    
    resp = requests.post("https://api.sarvam.ai/v1/chat/completions", headers={
        "api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json",
    }, json={"model": "sarvam-m", "messages": messages, "max_tokens": 400, "temperature": 0.3}, timeout=30)
    
    if resp.status_code != 200:
        raise Exception(f"LLM Error {resp.status_code}: {resp.text}")
    c = resp.json()["choices"][0]["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", c, flags=re.DOTALL).strip()


def clean_for_tts(text):
    """Strip markdown/HTML so TTS doesn't read formatting characters aloud."""
    t = re.sub(r'<[^>]+>', '', text)
    t = re.sub(r'\*\*(.+?)\*\*', r'\1', t)
    t = re.sub(r'\*(.+?)\*', r'\1', t)
    t = re.sub(r'__(.+?)__', r'\1', t)
    t = re.sub(r'_(.+?)_', r'\1', t)
    t = re.sub(r'~~(.+?)~~', r'\1', t)
    t = re.sub(r'`(.+?)`', r'\1', t)
    t = re.sub(r'^#{1,6}\s+', '', t, flags=re.MULTILINE)
    t = re.sub(r'^\s*[-*+]\s+', '', t, flags=re.MULTILINE)
    t = re.sub(r'^\s*\d+\.\s+', '', t, flags=re.MULTILINE)
    t = re.sub(r'^\s*>\s*', '', t, flags=re.MULTILINE)
    t = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', t)
    t = re.sub(r'[|]', ' ', t)
    t = re.sub(r'-{3,}', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def get_tts_html(text, lang="en"):
    if not SARVAM_API_KEY:
        return "<p style='color:#999'><em>TTS unavailable: No API key</em></p>"
    
    lang_code = LANG_MAP.get(lang[:2] if lang else "en", "en-IN")
    clean_text = clean_for_tts(text)
    try:
        resp = requests.post("https://api.sarvam.ai/text-to-speech", headers={
            "api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json",
        }, json={
            "inputs": [clean_text[:500]], "target_language_code": lang_code,
            "speaker": "anushka", "model": "bulbul:v2",
            "pitch": 0, "pace": 1.0, "loudness": 1.5, "enable_preprocessing": True,
        }, timeout=30)
        if resp.status_code == 200:
            ab = resp.json().get("audios", [None])[0]
            if ab:
                return f'<audio controls autoplay style="width:100%" src="data:audio/wav;base64,{ab}"></audio>'
    except Exception as e:
        logger.error(f"TTS error: {e}")
    return "<p style='color:#999'><em>Audio generation failed</em></p>"


def do_stt(audio_b64, filename="audio.webm"):
    if not SARVAM_API_KEY:
        raise Exception("STT Error: No API key configured")
    
    headers = {"api-subscription-key": SARVAM_API_KEY}
    audio_bytes = base64.b64decode(audio_b64)
    files = {"file": (filename, io.BytesIO(audio_bytes), "audio/webm")}
    data = {"model": "saaras:v3", "language_code": "unknown", "with_timestamps": "false"}
    resp = requests.post("https://api.sarvam.ai/speech-to-text", headers=headers, files=files, data=data, timeout=30)
    if resp.status_code != 200:
        raise Exception(f"STT Error: {resp.text}")
    r = resp.json()
    return r.get("transcript", ""), r.get("language_code", "en")


def detect_lang(text):
    if any('\u0900' <= c <= '\u097F' for c in text):
        return "hi"
    if any('\u0B80' <= c <= '\u0BFF' for c in text):
        return "ta"
    if any('\u0C00' <= c <= '\u0C7F' for c in text):
        return "te"
    if any('\u0C80' <= c <= '\u0CFF' for c in text):
        return "kn"
    if any('\u0D00' <= c <= '\u0D7F' for c in text):
        return "ml"
    if any('\u0980' <= c <= '\u09FF' for c in text):
        return "bn"
    return "en"


def format_chat():
    if not conversation:
        return """*Start learning! Try asking a question in Hindi, English, Tamil, or any Indian language.*

**Example queries:**
- "Explain the process of photosynthesis."
- "What were the main causes of the 1857 revolt?"
- "Solve this: If a car travels 60km in 2 hours, what is its speed?"
"""
    lines = []
    for u, b in conversation:
        lines.append(f'> **Student:** {u}')
        lines.append(f'**Shiksha Sathi:** {b}')
        lines.append('')
    return '\n\n'.join(lines)


def handle_text(user_text):
    global conversation
    if not user_text.strip():
        return "", format_chat(), "", ""
    try:
        reply = call_llm(user_text)
        lang = detect_lang(user_text)
        audio = get_tts_html(reply, lang)
        conversation.append((user_text, reply))
        lang_name = LANG_NAMES.get(lang, lang)
        return "", format_chat(), audio, f"Language: {lang_name}"
    except Exception as e:
        logger.error(f"Text error: {e}")
        conversation.append((user_text, f"Sorry, error occurred: {e}"))
        return "", format_chat(), "", f"Error: {e}"


def handle_voice_b64(audio_data_str):
    global conversation
    if not audio_data_str or not audio_data_str.strip():
        return format_chat(), "", "No audio data received. Please record first, then click Send Voice."
    try:
        if "base64," in audio_data_str:
            audio_data_str = audio_data_str.split("base64,")[1]
        user_text, lang = do_stt(audio_data_str)
        if not user_text.strip():
            return format_chat(), "", "Could not understand the audio. Please try again."
        reply = call_llm(user_text)
        audio = get_tts_html(reply, lang)
        conversation.append((user_text, reply))
        lang_name = LANG_NAMES.get(lang[:2], lang)
        return format_chat(), audio, f'Heard: "{user_text}" | Language: {lang_name}'
    except Exception as e:
        logger.error(f"Voice error: {e}")
        return format_chat(), "", f"Error: {e}"


def clear_all():
    global conversation
    conversation = []
    return format_chat(), "", "", "", "", ""


logger.info("Building UI...")

HEAD_JS = """
<script>
let sathiRecorder = null;
let sathiChunks = [];
let sathiRecording = false;
let sathiTimer = null;
let sathiSecs = 0;

async function sathiStartRec() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        sathiRecorder = new MediaRecorder(stream);
        sathiChunks = [];
        sathiRecorder.ondataavailable = (e) => { if (e.data.size > 0) sathiChunks.push(e.data); };
        sathiRecorder.onstop = () => {
            const blob = new Blob(sathiChunks, { type: 'audio/webm' });
            const reader = new FileReader();
            reader.onloadend = () => {
                const b64 = reader.result;
                const el = document.querySelector('#voice-data-box textarea');
                if (el) {
                    const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
                    setter.call(el, b64);
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                }
                document.getElementById('rec-status-text').textContent = 'Recording saved! Click "Send Voice" to process.';
            };
            reader.readAsDataURL(blob);
            stream.getTracks().forEach(t => t.stop());
        };
        sathiRecorder.start();
        sathiRecording = true;
        sathiSecs = 0;
        document.getElementById('rec-indicator').style.display = 'flex';
        document.getElementById('rec-status-text').textContent = 'Recording...';
        sathiTimer = setInterval(() => {
            sathiSecs++;
            const m = String(Math.floor(sathiSecs/60)).padStart(2,'0');
            const s = String(sathiSecs%60).padStart(2,'0');
            document.getElementById('rec-time').textContent = m + ':' + s;
        }, 1000);
    } catch (err) {
        document.getElementById('rec-status-text').textContent = 'Microphone access denied: ' + err.message;
    }
}

function sathiStopRec() {
    if (sathiRecorder && sathiRecorder.state !== 'inactive') {
        sathiRecorder.stop();
    }
    sathiRecording = false;
    clearInterval(sathiTimer);
    document.getElementById('rec-indicator').style.display = 'none';
}
</script>
"""

app = gr.Blocks(title="Shiksha Sathi Voice Tutor", head=HEAD_JS)
with app:
    gr.HTML("""<div style="text-align:center;margin-bottom:12px">
        <h1 style="color:#10B981;margin:0;font-size:2em">🎓 Shiksha Sathi</h1>
        <p style="color:#555;margin:2px 0;font-size:1.1em">Sovereign Voice Educational Tutor</p>
        <p style="font-size:12px;color:#888;margin:2px 0">Powered by <b>Sarvam AI</b> + <b>Databricks</b> | 22 Indian Languages</p>
        <div style="background:linear-gradient(90deg,#FF9933 33%,#FFFFFF 33%,#FFFFFF 66%,#138808 66%);padding:4px 16px;border-radius:20px;display:inline-block;font-size:11px;font-weight:bold;color:#333;margin-top:4px">
            100% India-Hosted &mdash; Full Data Sovereignty
        </div></div>""")

    chat_display = gr.Markdown(value=format_chat())

    with gr.Tab("Text"):
        with gr.Row():
            text_input = gr.Textbox(
                label="Ask Shiksha Sathi",
                placeholder="e.g. How does the heart pump blood? / Explain gravity in Hindi",
                scale=4, lines=1)
            text_btn = gr.Button("Send", variant="primary", scale=1)
        text_audio_html = gr.HTML()
        text_status = gr.Textbox(label="Status", interactive=False, max_lines=1)

    with gr.Tab("Voice"):
        gr.HTML("""<div style="text-align:center;padding:16px">
            <p style="color:#555;margin:0 0 8px 0">Use the buttons below to record your textbook question in any Indian language</p>
            <div id="rec-indicator" style="display:none;align-items:center;justify-content:center;gap:10px;margin:8px 0">
                <span style="display:inline-block;width:12px;height:12px;background:#dc3545;border-radius:50%;animation:blink 1s infinite"></span>
                <span id="rec-time" style="font-size:22px;font-weight:bold;color:#dc3545">00:00</span>
            </div>
            <p id="rec-status-text" style="color:#888;margin:4px 0;font-size:13px">Click "Record" to start</p>
            <style>@keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}</style>
        </div>""")
        with gr.Row():
            rec_btn = gr.Button("🎙 Record", variant="secondary", scale=1)
            stop_btn = gr.Button("⏹ Stop", variant="stop", scale=1)
        voice_data = gr.Textbox(label="Audio Data", visible=False, elem_id="voice-data-box")
        voice_btn = gr.Button("Send Voice", variant="primary")
        voice_audio_html = gr.HTML()
        voice_status = gr.Textbox(label="Status", interactive=False, max_lines=1)

    clear_btn = gr.Button("Clear Conversation", variant="secondary")

    gr.Markdown("""---
| Component | Provider | Location |
|-----------|----------|----------|
| **Speech-to-Text** | Sarvam Saaras v3 | India |
| **LLM Brain** | Sarvam-M / Sarvam-30B on Databricks | India |
| **Text-to-Speech** | Sarvam Bulbul v2 | India |
| **Platform** | Databricks | Azure Central India |

*Zero data leaves India. Supports: Hindi, English, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati, Marathi, Punjabi, Odia*""")

    text_btn.click(handle_text, [text_input], [text_input, chat_display, text_audio_html, text_status])
    text_input.submit(handle_text, [text_input], [text_input, chat_display, text_audio_html, text_status])
    rec_btn.click(fn=None, inputs=None, outputs=None, js="() => { sathiStartRec(); }")
    stop_btn.click(fn=None, inputs=None, outputs=None, js="() => { sathiStopRec(); }")
    voice_btn.click(handle_voice_b64, [voice_data], [chat_display, voice_audio_html, voice_status])
    clear_btn.click(clear_all, outputs=[chat_display, text_audio_html, voice_audio_html, text_status, voice_status, voice_data])

logger.info("UI built successfully")

# ── Local FRPC Tunnel Bypass ──────────────────────────────────────────────────
def apply_local_frpc():
    """Copies the frpc binary from your Workspace to the Gradio library."""
    # Explicitly define your workspace path
    workspace_dir = "/Workspace/Users/ch23b006@smail.iitm.ac.in/R266"
    uploaded_file = os.path.join(workspace_dir, "frpc_linux_aarch64_v0.2")
    
    gradio_dir = os.path.dirname(gr.__file__)
    target_path = os.path.join(gradio_dir, "frpc_linux_aarch64_v0.2")
    
    if os.path.exists(target_path):
        logger.info("FRPC binary already exists in Gradio directory.")
        return
        
    if os.path.exists(uploaded_file):
        logger.info(f"Found frpc at {uploaded_file}. Copying to Gradio...")
        shutil.copy(uploaded_file, target_path)
        st = os.stat(target_path)
        os.chmod(target_path, st.st_mode | stat.S_IEXEC)
        logger.info("✅ Tunnel is ready.")
    else:
        logger.warning(f"❌ Error: Could not find '{uploaded_file}'. Check if the file is uploaded to the R266 folder.")

# Apply the fix right before launch
apply_local_frpc()

print("\n" + "="*60)
print("🚀 Launching Shiksha Sathi with public tunnel enabled...")
print("="*60)

app.launch(share=True)

# COMMAND ----------

import os
import gradio as gr

# 1. The path where you were supposed to upload the file
workspace_path = "/Workspace/Users/ch23b006@smail.iitm.ac.in/R266/frpc_linux_aarch64_v0.2"

# 2. The temporary Gradio folder where the app looks for it
gradio_dir = os.path.dirname(gr.__file__)
gradio_path = os.path.join(gradio_dir, "frpc_linux_aarch64_v0.2")

print("--- 🔍 DIAGNOSTIC CHECK ---")
print(f"1. Checking your permanent Workspace:\n   Path: {workspace_path}")
if os.path.exists(workspace_path):
    print("   ✅ FOUND! The file is safely in your workspace.")
else:
    print("   ❌ MISSING! You have not uploaded the file here, or it has the wrong name.")

print(f"\n2. Checking the hidden Gradio folder:\n   Path: {gradio_path}")
if os.path.exists(gradio_path):
    print("   ✅ FOUND! Gradio sees the file and sharing should work.")
else:
    print("   ❌ MISSING! The file has not been copied to Gradio yet.")

# COMMAND ----------

# MAGIC %pip install "streamlit>=1.38.0" faiss-cpu sentence-transformers requests numpy pandas

# COMMAND ----------

# MAGIC %pip install faiss-cpu sentence-transformers gradio==4.44.0 gradio-client==1.3.0 requests numpy pandas

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install faiss-cpu sentence-transformers gradio==4.44.0 gradio-client==1.3.0 huggingface-hub==0.23.2 requests numpy pandas

# COMMAND ----------

databricks secrets put-acl \
  --scope ncert-tutor \
  --principal ch23b006@smail.iitm.ac.in \
  --permission READ

# COMMAND ----------

import sys
import os
import gradio as gr
import re
from openai import OpenAI

# ── Path setup ───────────────────────────────────────────────
current_dir = os.getcwd()
sys.path.insert(0, os.path.join(current_dir, "src"))

# Clear cached modules
for mod in ['llm_client', 'sarvam_client', 'retriever', 'ncert_filter']:
    if mod in sys.modules:
        del sys.modules[mod]

# ── Secrets ──────────────────────────────────────────────────
os.environ["SARVAM_API_KEY"] = dbutils.secrets.get(scope="ncert-tutor", key="sarvam-api-key")
os.environ["NYAYA_INDEX_DIR"] = "/Volumes/workspace/default/ncert-tutor/index"
os.environ["DATABRICKS_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# ── Imports from src ─────────────────────────────────────────
from retriever import get_retriever
from sarvam_client import (
    chat_completions as _sarvam_chat,
    speech_to_text_file,
    text_to_speech_wav_bytes,
    wav_bytes_to_numpy_float32,
    strip_markdown_for_tts
)

# ── OpenAI client (Databricks) ───────────────────────────────
client = OpenAI(
    api_key=os.environ["DATABRICKS_TOKEN"],
    base_url="https://7474657381434754.ai-gateway.cloud.databricks.com/mlflow/v1"
)

# ── Retriever ────────────────────────────────────────────────
retriever = get_retriever()

# ── LLM with fallback ────────────────────────────────────────
def llm_chat(messages: list) -> tuple[str, str]:
    try:
        response = client.chat.completions.create(
            model="databricks-gpt-5-4",
            messages=messages,
            max_tokens=2000,
            temperature=0.2
        )
        return (response.choices[0].message.content, "Databricks GPT-5.4")
    except Exception as db_err:
        print(f"[Databricks failed → Sarvam fallback] {db_err}")
        response = _sarvam_chat(messages=messages)
        return (response["choices"][0]["message"]["content"], "Sarvam AI")

# ── Response parser ──────────────────────────────────────────
def parse_response(raw_response: str, model_name: str) -> str:
    model_badge = f'⚡ Powered by {model_name}\n\n'

    think_match = re.search(r'<think>(.*?)</think>', raw_response, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
        return model_badge + f"[Thinking]\n{thinking}\n\n[Answer]\n{answer}"

    return model_badge + raw_response


# ── CORE PIPELINE (VOICE → TEXT → LLM → AUDIO) ───────────────
def process_query(audio_path, subject, student_class, language):
    if audio_path is None:
        return None, "Please provide audio input."

    # ── 1. Speech → Text ─────────────────────────────
    try:
        stt_result = sarvam_stt(audio_path)
        user_text = stt_result.get("text", "")
        print(f"[User said]: {user_text}")
    except Exception as e:
        return None, f"STT Error: {e}"

    if not user_text.strip():
        return None, "Could not understand audio."

    # ── 2. Retrieve context ──────────────────────────
    results_df = retriever.search(
        query=user_text,
        subject=subject,
        student_class=student_class,
        k=3
    )

    if results_df.empty:
        return None, "No relevant textbook content found."

    context_text = "\n\n".join(results_df["text"].tolist())

    system_prompt = (
        f"You are Shiksha Sathi, a helpful Indian school tutor. "
        f"Answer in {language}. Use ONLY the provided textbook context. "
        f"If not found, say you don't know."
    )

    user_prompt = f"Context:\n{context_text}\n\nQuestion: {user_text}"

    # ── 3. LLM ───────────────────────────────────────
    try:
        raw_answer, model_used = llm_chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
    except Exception as e:
        return None, f"LLM Error: {e}"

    answer_text = parse_response(raw_answer, model_used)

    # ── 4. Text → Speech ─────────────────────────────
    try:
        tts_result = sarvam_tts(
            text=answer_text,
            target_language=language
        )

        # Depending on API: adjust this
        audio_output_path = tts_result.get("audio_file") or tts_result.get("audio")

    except Exception as e:
        return None, f"TTS Error: {e}"

    return audio_output_path, answer_text


# ── UI ──────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 📚 Shiksha Sathi · Voice Tutor")
    gr.Markdown("🎤 Speak your question and get spoken answers!")

    with gr.Row():
        subject_dropdown = gr.Dropdown(
            ["social_science", "science", "english"],
            value="social_science",
            label="Subject"
        )

        class_dropdown = gr.Dropdown(
            ["5", "6", "7", "8"],
            value="6",
            label="Class"
        )

        language_dropdown = gr.Dropdown(
            ["en", "hi", "ta"],
            value="en",
            label="Response Language"
        )

    audio_input = gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="🎤 Speak your question"
    )

    submit_btn = gr.Button("Ask")

    audio_output = gr.Audio(label="🔊 Answer (Audio)")
    text_output = gr.Textbox(label="📝 Answer (Text)", lines=8)

    submit_btn.click(
        fn=process_query,
        inputs=[audio_input, subject_dropdown, class_dropdown, language_dropdown],
        outputs=[audio_output, text_output]
    )

# ── Launch ──────────────────────────────────────────────────
demo.launch(share=True, debug=True)

# COMMAND ----------

dbutils.secrets.list("ncert-tutor")

# COMMAND ----------

