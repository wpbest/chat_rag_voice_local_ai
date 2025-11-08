# chat_rag_voice_local_ai
chat_rag_voice_local_ai — A single self-contained application running in VS Code: offline, voice-enabled RAG assistant that listens, thinks, and speaks using local embeddings and on-device LLMs. Private. Secure. Fast. Always available without the cloud.

--

## 🧠 Features
- 💬 Voice-driven natural conversation  
- 🧱 Local RAG memory using SQLite + `sqlite-vec`  
- 🗣️ Text-to-speech (TTS) responses  
- 🔒 Fully offline — no external API calls  
- ⚡ Fast startup and low resource use  
- 🧩 Designed for MS AI Toolkit / VS Code runtime  

---

## ⚙️ Setup Instructions (VS Code + Python 3.11.9)

### 1️⃣ Clone the repository
```bash
git clone https://github.com/wpbest/chat_rag_voice_local_ai.git
cd chat_rag_voice_local_ai

2️⃣ Create and activate a virtual envir

2️⃣ Create and activate a virtual environment with Python 3.11.9

Windows (PowerShell):

py -3.11 -m venv .venv
.venv\Scripts\activate


macOS / Linux:

python3.11 -m venv .venv
source .venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Open in VS Code

Open the project folder in VS Code

Press Ctrl + Shift + P → Python: Select Interpreter → choose the one from .venv

Check if the LLM is running
Invoke-RestMethod http://127.0.0.1:5272/v1/models   

Then open the terminal and run:

python chat_rag_voice_local_ai.py

5️⃣ Speak and interact

After the warm-up, AVA will say:

“Get Ready to Say something when I say I am Listening…”

Now you can talk to your offline assistant.