# Databrick_hackathon_Room266
# 📚 Shiksha Sathi · शिक्षा साथी

**Your AI-Powered Educational Tutor for Indian School Textbooks**

Shiksha Sathi is an intelligent tutoring system designed to help Indian students (Classes 5-8) learn from their NCERT textbooks through natural language Q&A and interactive quizzes. Built with RAG (Retrieval-Augmented Generation) architecture, it provides accurate, context-aware answers directly from official textbooks.

[![Built with Databricks](https://img.shields.io/badge/Built%20with-Databricks-FF3621?logo=databricks)](https://databricks.com)
[![Powered by Sarvam AI](https://img.shields.io/badge/Powered%20by-Sarvam%20AI-4A90E2)](https://sarvam.ai)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange)](https://gradio.app)

---

## 🌟 Features

### 💬 Interactive Q&A Chat
- **Natural Language Questions**: Ask questions in plain language about any topic from your textbooks
- **Context-Aware Answers**: Responses are grounded in actual textbook content using RAG
- **Multi-Subject Support**: Social Science, Science, and English for Classes 5-8
- **Source Attribution**: See which textbook sections were used to answer your question

### 🎯 Quiz Generator
- **Custom Topic Quizzes**: Generate quizzes on any topic from your textbooks
- **Adaptive Difficulty**: Choose from Easy, Medium, or Hard difficulty levels
- **Multiple Choice Questions**: 5, 10, or 15 questions per quiz
- **Instant Feedback**: Get correct answers with explanations after completion

### 🎤 Voice Support (Experimental)
- **Speech-to-Text**: Ask questions by speaking (supports 22+ Indian languages)
- **Text-to-Speech**: Listen to answers read aloud
- **Multilingual**: Hindi, English, Tamil, Telugu, and more
- **Sovereign Architecture**: 100% India-hosted voice pipeline (Sarvam AI)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Shiksha Sathi App                       │
│                      (Gradio UI)                            │
└─────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┴─────────────────┐
            │                                   │
            ▼                                   ▼
┌───────────────────────┐           ┌───────────────────────┐
│   Retriever Module    │           │   LLM Module          │
│   (FAISS + Filters)   │           │   (Sarvam AI)         │
└───────────────────────┘           └───────────────────────┘
            │                                   │
            ▼                                   ▼
┌───────────────────────┐           ┌───────────────────────┐
│  Vector Database      │           │  Chat Completions     │
│  (FAISS Index)        │           │  (Sarvam M/30B)       │
│  + Metadata           │           │                       │
└───────────────────────┘           └───────────────────────┘
```

### Key Components

1. **Data Processing Pipeline**
   - Parquet files with chunked NCERT textbook content
   - Metadata: subject, class, chapter, chunk_id
   - FAISS vector index for semantic search

2. **Retrieval System**
   - Semantic search using FAISS
   - Metadata filtering by subject and class
   - Top-k retrieval (default k=3)

3. **LLM Integration**
   - Sarvam AI chat completions API
   - Context-aware prompt engineering
   - Response parsing with thinking tags support

4. **Voice Pipeline** (Optional)
   - STT: Sarvam Saaras v3
   - TTS: Sarvam Bulbul v2
   - Language detection and routing

---

## 📁 Project Structure

```
R266/
├── app.py                      # Main Gradio application
├── src/
│   ├── retriever.py            # FAISS retrieval logic
│   ├── sarvam_client.py        # Sarvam AI API client
│   ├── llm_client.py           # LLM wrapper (legacy)
│   ├── ncert_filter.py         # Metadata filtering utilities
│   ├── quiz_generator.py       # Quiz generation logic
│   └── build_index_startup.py  # FAISS index builder
├── index/                      # FAISS index files (local/volume)
│   ├── corpus.faiss            # Vector index
│   ├── chunks.parquet          # Text chunks + metadata
│   └── metadata.json           # Index configuration
├── notebooks/
│   └── new_launch.ipynb        # Development/testing notebook
└── README.md                   # This file
```

---

## 🚀 Setup & Installation

### Prerequisites

- **Python**: 3.10+
- **Databricks**: Workspace with Unity Catalog access (optional but recommended)
- **Sarvam AI API Key**: Sign up at [sarvam.ai](https://sarvam.ai)

### 1. Clone or Navigate to Project

```bash
cd /Workspace/Users/<your-username>/R266
```

### 2. Install Dependencies

```bash
pip install gradio faiss-cpu pandas pyarrow requests
```

### 3. Set Environment Variables

Create a `.env` file or use Databricks secrets:

```bash
# Required
export SARVAM_API_KEY="your-sarvam-api-key"

# Optional (auto-detected if in Databricks)
export NYAYA_INDEX_DIR="/Volumes/workspace/default/ncert-tutor/index"
```

**Using Databricks Secrets** (Recommended):

```python
# In notebook or app.py
os.environ["SARVAM_API_KEY"] = dbutils.secrets.get(scope="ncert-tutor", key="sarvam-api-key")
```

### 4. Prepare FAISS Index

**Option A: Use Existing Volume** (Recommended for Databricks Apps)
- Ensure `/Volumes/workspace/default/ncert-tutor/index` exists with `corpus.faiss` and `chunks.parquet`

**Option B: Build Index Locally**
```bash
# Place your chunks.parquet in R266/index/
python -c "from src.build_index_startup import build_faiss_index_from_parquet; from pathlib import Path; build_faiss_index_from_parquet(Path('index'))"
```

---

## 💻 Usage

### Running the App Locally

```bash
python app.py
```

The app will launch on `http://127.0.0.1:7860` by default.

### Running in Databricks Notebook

```python
# Cell 1: Install dependencies
%pip install gradio faiss-cpu

# Cell 2: Set secrets
import os
os.environ["SARVAM_API_KEY"] = dbutils.secrets.get(scope="ncert-tutor", key="sarvam-api-key")

# Cell 3: Run app
%run ./app.py
```

### Deploying as Databricks App

1. **Upload Files**: Ensure all files are in `/Users/<username>/R266/`
2. **Create App**: 
   - Go to Databricks Apps
   - Click "Create App"
   - Select `app.py` as entry point
   - Set environment variables in app configuration
3. **Launch**: Click "Run" to deploy

---

## 🎓 How to Use

### Q&A Chat

1. Select **Subject** (Social Science, Science, or English)
2. Select **Class** (5, 6, 7, or 8)
3. Type your question (e.g., "What is photosynthesis?")
4. Get an answer with sources from the textbook

### Quiz Generator

1. Navigate to **Quiz Generator** tab
2. Select subject, class, and topic
3. Choose difficulty and number of questions
4. Click **Generate Quiz**
5. Answer questions and click **Show Answers** to check

### Voice Mode (if enabled)

1. Navigate to **Voice** tab
2. Click **🎙 Record** to start recording
3. Ask your question (in any supported Indian language)
4. Click **⏹ Stop** when done
5. Click **Send Voice** to process
6. Listen to the audio response (auto-plays)

---

## 🔧 Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SARVAM_API_KEY` | ✅ Yes | - | Sarvam AI API key |
| `NYAYA_INDEX_DIR` | ❌ No | `./index` | Path to FAISS index directory |
| `DATABRICKS_TOKEN` | ❌ No | Auto | Databricks auth token (auto-detected) |
| `DATABRICKS_HOST` | ❌ No | Auto | Databricks workspace URL (auto-detected) |

### Supported Content

**Classes**: 5, 6, 7, 8  
**Subjects**: 
- Social Science (History, Geography, Civics)
- Science (Physics, Chemistry, Biology)
- English (Literature, Grammar)

**Languages** (Voice):
- Hindi, English, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati, Marathi, Punjabi, Odia

---

## 🛠️ Development

### Adding New Textbooks

1. **Process Textbook**: Convert PDF to text chunks with metadata
2. **Create Parquet**: Save as `chunks.parquet` with columns:
   - `text`: Chunk text
   - `subject`: Subject name
   - `student_class`: Class number (string)
   - `chapter_name`: Chapter title
   - `chunk_id`: Unique identifier
3. **Rebuild Index**: Run `build_index_startup.py`

### Customizing Retrieval

Edit `src/retriever.py`:
```python
# Change number of results
results = retriever.search(query="...", k=5)  # Default: k=3

# Adjust similarity threshold
# Modify search() method in Retriever class
```

### Customizing LLM Behavior

Edit `app.py` system prompt:
```python
system_prompt = (
    "You are Shiksha Sathi, a helpful Indian school tutor. "
    "Be encouraging and explain concepts step-by-step..."
)
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'gradio'"
```bash
pip install gradio
# Or in Databricks:
%pip install gradio
dbutils.library.restartPython()
```

### "SARVAM_API_KEY is NOT set"
```python
# Add to app.py before imports:
os.environ["SARVAM_API_KEY"] = "your-api-key-here"
```

### "FAISS index not found"
- Ensure `corpus.faiss` exists in index directory
- Check `NYAYA_INDEX_DIR` points to correct location
- Rebuild index using `build_index_startup.py`

### App not launching in Databricks
- **Compute Issue**: Cluster may be terminated. Start the cluster first.
- **Port Conflict**: Change port in `demo.launch(server_port=7860)`
- **Secrets**: Use Databricks secrets instead of hardcoded API keys

---

## 📊 Performance

- **Retrieval Latency**: ~50-200ms (FAISS search)
- **LLM Response Time**: ~2-5s (Sarvam AI)
- **Quiz Generation**: ~5-10s (depends on number of questions)
- **Voice Latency**: ~1-3s (STT + TTS combined)

---

## 🤝 Contributing

This is an educational project. Feel free to:
- Add support for more classes/subjects
- Improve retrieval algorithms
- Enhance UI/UX
- Add new features (flashcards, progress tracking, etc.)

---

## 📜 License

This project is for educational purposes. NCERT textbook content is copyrighted by NCERT (National Council of Educational Research and Training).

---

## 🙏 Acknowledgments

- **NCERT**: For high-quality educational content
- **Sarvam AI**: For India-first AI infrastructure
- **Databricks**: For scalable data platform
- **Gradio**: For rapid UI development

---

## 📞 Support

For issues or questions:
1. Check the **Troubleshooting** section above
2. Review logs in `[App Startup]` and `[LLM]` prefixed messages
3. Verify API keys and environment setup

---

**Built with ❤️ for Indian Students**
EOF

echo "✅ README.md created successfully"
