# chat_rag_voice_local_ai.py
import sys, time, struct, sqlite3, re, logging
from typing import List, Dict
import requests, speech_recognition as sr, pyttsx3, sqlite_vec
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ========= Logging Setup =========
LOG_FILE = "chat_memory.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)

# ========= RAG / Memory Config =========
DB_FILE, VEC_TABLE, META_TABLE, FACTS_TABLE = "chat_memory.db", "messages_vec", "messages_meta", "facts"
EMBEDDING_MODEL, EMBEDDING_DIMS, TOP_K, MAX_SNIPPET_CHARS = "all-MiniLM-L6-v2", 384, 5, 400

# ========= Microsoft AI Toolkit Config =========
AI_TOOLKIT_BASE_URL = "http://127.0.0.1:5272/v1/"
# AI_TOOLKIT_MODEL = "qwen2.5-coder-0.5b-instruct-cuda-gpu:3"
AI_TOOLKIT_MODEL = "gpt-oss-20b-cuda-gpu:1"
AI_TEMPERATURE, AI_MAX_TOKENS = 0.0, 80
_ai_client = OpenAI(base_url=AI_TOOLKIT_BASE_URL, api_key="unused")

# ========= Mic tuning =========
PHRASE_TIME_LIMIT, AMBIENT_NOISE_DURATION = 7, 0.4

# ========= Embedding Helpers =========
_model = None
def get_model():
    global _model
    if _model is None:
        logging.info("Loading embedding model (preload)...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logging.info("Embedding model loaded and ready.")
    return _model

def embed(text: str):
    return get_model().encode(text, normalize_embeddings=True).tolist()

def serialize_f32(vec):
    return struct.pack("%sf" % len(vec), *vec)

# ========= SQLite Setup =========
def ensure_db():
    conn = sqlite3.connect(DB_FILE)
    conn.enable_load_extension(True); sqlite_vec.load(conn); conn.enable_load_extension(False)
    conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS {VEC_TABLE} USING vec0(embedding float[{EMBEDDING_DIMS}])")
    conn.execute(f"CREATE TABLE IF NOT EXISTS {META_TABLE}(rowid INTEGER PRIMARY KEY,ts REAL NOT NULL,role TEXT NOT NULL,text TEXT NOT NULL)")
    conn.execute(f"CREATE TABLE IF NOT EXISTS {FACTS_TABLE}(key TEXT PRIMARY KEY,value TEXT NOT NULL,updated_at REAL NOT NULL)")
    conn.commit(); conn.close(); logging.info("Database initialized and verified.")

# ========= Warm-Up =========
def warmup_environment():
    start=time.time(); logging.info("Performing system warm-up...")
    ensure_db(); model=get_model(); _=model.encode("warmup test",normalize_embeddings=True)
    logging.info("Model embedding warm-up complete.")
    conn=sqlite3.connect(DB_FILE); conn.enable_load_extension(True); sqlite_vec.load(conn)
    conn.enable_load_extension(False); conn.execute(f"SELECT count(*) FROM {META_TABLE}"); conn.close()
    logging.info("SQLite vec0 warm-up complete."); logging.info(f"Warm-up finished in {time.time()-start:.2f} seconds.")

# ========= Memory & RAG =========
def remember(conn, role, text):
    vec=embed(text); cur=conn.execute(f"INSERT INTO {VEC_TABLE}(embedding) VALUES (?)",(serialize_f32(vec),))
    rid=cur.lastrowid
    conn.execute(f"INSERT INTO {META_TABLE}(rowid,ts,role,text) VALUES (?,?,?,?)",(rid,time.time(),role,text))
    conn.commit(); logging.info(f"Remembered message ({role}): {text[:80]}...")

def recall(conn, query, k=TOP_K):
    qv=embed(query)
    n=conn.execute(f"SELECT rowid,distance FROM {VEC_TABLE} WHERE embedding MATCH ? ORDER BY distance LIMIT ?",(serialize_f32(qv),k)).fetchall()
    if not n: logging.info("No similar memories recalled."); return []
    ids=",".join(str(rid) for rid,_ in n)
    meta=conn.execute(f"SELECT rowid,role,text FROM {META_TABLE} WHERE rowid IN ({ids})").fetchall()
    mb={r:(ro,tx) for r,ro,tx in meta}
    logging.info(f"Recalled {len(n)} memory snippets."); return [(d,*mb.get(r,('unknown',''))) for r,d in n]

def format_memory_snippets(h):
    if not h: return "None."
    return "\n".join([f"- ({ro}, d={d:.3f}) {(tx[:MAX_SNIPPET_CHARS]+'…') if len(tx)>MAX_SNIPPET_CHARS else tx}" for d,ro,tx in h])

# ========= Facts =========
def upsert_fact(conn,key,val):
    conn.execute(f"""
        INSERT INTO {FACTS_TABLE}(key,value,updated_at)
        VALUES(?,?,?)
        ON CONFLICT(key)
        DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
    """,(key,val,time.time()))
    conn.commit(); logging.info(f"Fact updated: {key} = {val}")

def get_all_facts(conn):
    return {k:v for k,v in conn.execute(f"SELECT key,value FROM {FACTS_TABLE}").fetchall()}

def extract_and_store_facts(conn, text):
    """
    Robust fact extraction compatible with gpt-oss-20b-cuda-gpu multi-channel output.
    Strips analysis chatter and captures only the final factual pairs.
    """
    try:
        prompt = (
            "You are AVA, an assistant that extracts factual self-descriptions the USER gives about themselves.\n"
            "Return them as plain 'key=value' pairs (for example: name=William, hair_color=brown).\n"
            "If no facts are found, return nothing. Do not include analysis, reasoning, or AVA-related info.\n\n"
            f"USER said: {text.strip()}\n\nFacts:\n"
        )

        chat = _ai_client.chat.completions.create(
            model=AI_TOOLKIT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.0,
        )

        raw = chat.choices[0].message.content.strip()

        # --- Clean up multi-channel markup ---
        # 1. Prefer final channel if present
        final_match = re.search(r"<\|channel\|>final<\|message\|>(.*?)<\|end\|>", raw, re.DOTALL)
        if final_match:
            content = final_match.group(1)
        else:
            # 2. Otherwise, take last message block after any <|channel|> markers
            parts = re.split(r"<\|channel\|>.*?<\|message\|>", raw)
            content = parts[-1] if len(parts) > 1 else raw

        # 3. Remove leftover <|...|> tags and analysis chatter
        cleaned = re.sub(r"<\|.*?\|>", "", content)
        cleaned = re.sub(r"(?i)analysis.*?:", "", cleaned).strip()

        # --- Parse key=value or key: value pairs only ---
        for line in cleaned.splitlines():
            if "=" in line or ":" in line:
                k, v = re.split(r"[:=]", line, 1)
                k, v = k.strip().lower(), v.strip()
                if k and v:
                    upsert_fact(conn, k, v)
                    logging.info(f'Fact updated: "{k}" = "{v}"')

        if not cleaned.strip():
            logging.info("No factual data extracted.")

    except Exception as e:
        logging.warning(f"Fact extraction skipped due to error: {e}")



# ========= Prompt builder =========
def build_rag_prompt(memory_block: str, user_text: str, facts: dict) -> str:
    """
    Builds a grounded prompt ensuring AVA relies on existing context and facts dynamically,
    without hardcoded phrasing or example prohibitions.
    """
    facts_block = "None."
    if facts:
        facts_block = "\n".join([f"- {k}: {v}" for k, v in facts.items()])

    return (
        "System role definition:\n"
        "You are AVA, an intelligent, self-consistent voice assistant speaking to the USER.\n"
        "The USER facts provided below are established context — assume they are accurate and current.\n"
        "Use those facts confidently to respond, even if they seem incomplete.\n"
        "If information is missing, respond gracefully without asking for it unless the USER requests clarification.\n"
        "Never adopt the USER’s identity or personal attributes.\n"
        "Use 'you' when referring to the USER and 'I' when speaking as AVA.\n"
        "Respond naturally, briefly, and conversationally.\n\n"
        f"USER facts:\n{facts_block}\n\n"
        f"Recalled conversation memory:\n{memory_block}\n\n"
        f"USER said: {user_text}\n\n"
        "Respond as AVA with confidence based on the given context."
    )


# ========= Main Loop =========
def listen_and_recognize():
    logging.info(f"Starting AVA Voice Assistant on Python {sys.version}"); ensure_db()
    r=sr.Recognizer(); tts=pyttsx3.init(); tts.setProperty("volume",1.0)
    with sr.Microphone() as s:
        logging.info("Calibrating ambient noise level...")
        r.adjust_for_ambient_noise(s,duration=AMBIENT_NOISE_DURATION)
        logging.info(f"Energy threshold set to: {r.energy_threshold:.2f}")
        print("Get Ready to Say something when I say I am Listening...")
        while True:
            try:
                print("Listening..."); a=r.listen(s,timeout=None,phrase_time_limit=PHRASE_TIME_LIMIT)
                if len(a.frame_data)==0: logging.warning("Captured empty audio; skipping."); continue
                text=r.recognize_google(a); logging.info(f"User said: {text}")
                conn=sqlite3.connect(DB_FILE); conn.enable_load_extension(True); sqlite_vec.load(conn); conn.enable_load_extension(False)
                extract_and_store_facts(conn,text)
                hits=recall(conn,text,TOP_K); facts=get_all_facts(conn); memory=format_memory_snippets(hits)
                prompt=build_rag_prompt(memory,text,facts)
                try:
                    chat=_ai_client.chat.completions.create(
                        model=AI_TOOLKIT_MODEL,
                        messages=[{"role":"user","content":prompt}],
                        max_tokens=AI_MAX_TOKENS,
                        temperature=AI_TEMPERATURE,
                    )
                    resp=chat.choices[0].message.content.strip()
                    logging.info(f"AI Toolkit response: {resp}")
                except Exception as e:
                    logging.error(f"AI Toolkit inference error: {e}"); resp="Sorry, there was an error from the AI Toolkit model."
                remember(conn,"user",text); remember(conn,"assistant",resp); conn.close()
                try:
                    tts=pyttsx3.init(); tts.setProperty('volume',1.0); tts.say(resp); tts.runAndWait(); tts.stop(); tts=None
                except Exception as e: logging.error(f"TTS playback error: {e}")
            except sr.UnknownValueError: logging.warning("Speech not recognized (low confidence)."); continue
            except sr.RequestError as e: logging.error(f"Speech Recognition API error: {e}"); continue
            except KeyboardInterrupt: logging.info("User interrupted. Exiting."); break
            except Exception as e: logging.exception(f"Unexpected error: {e}"); continue
    try: tts.stop()
    except Exception: pass

# ========= Entry =========
if __name__=="__main__": warmup_environment(); listen_and_recognize()
