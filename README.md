# chat_memory_rag_voice.py
import sys, time, struct, sqlite3, re, logging
from typing import List, Tuple, Dict
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
AI_TOOLKIT_MODEL = "qwen2.5-coder-0.5b-instruct-cuda-gpu:3"
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

def embed(text: str) -> List[float]:
    return get_model().encode(text, normalize_embeddings=True).tolist()

def serialize_f32(vec: List[float]) -> bytes:
    return struct.pack("%sf" % len(vec), *vec)

# ========= SQLite Setup =========
def ensure_db():
    conn = sqlite3.connect(DB_FILE)
    conn.enable_load_extension(True); sqlite_vec.load(conn); conn.enable_load_extension(False)
    conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS {VEC_TABLE} USING vec0(embedding float[{EMBEDDING_DIMS}])")
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {META_TABLE}(rowid INTEGER PRIMARY KEY,ts REAL NOT NULL,role TEXT NOT NULL,text TEXT NOT NULL)""")
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {FACTS_TABLE}(key TEXT PRIMARY KEY,value TEXT NOT NULL,updated_at REAL NOT NULL)""")
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
def remember(conn:sqlite3.Connection,role:str,text:str):
    vec=embed(text); cur=conn.execute(f"INSERT INTO {VEC_TABLE}(embedding) VALUES (?)",(serialize_f32(vec),))
    rid=cur.lastrowid
    conn.execute(f"INSERT INTO {META_TABLE}(rowid,ts,role,text) VALUES (?,?,?,?)",(rid,time.time(),role,text))
    conn.commit(); logging.info(f"Remembered message ({role}): {text[:80]}...")

def recall(conn:sqlite3.Connection,query:str,k:int=TOP_K):
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
    conn.execute(f"""INSERT INTO {FACTS_TABLE}(key,value,updated_at) VALUES(?,?,?) 
                     ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at""",(key,val,time.time()))
    conn.commit(); logging.info(f"Fact updated: {key} = {val}")

def get_all_facts(conn): 
    return {k:v for k,v in conn.execute(f"SELECT key,value FROM {FACTS_TABLE}").fetchall()}

def extract_and_store_facts(conn,text):
    t=text.lower()
    if m:=re.search(r"\bmy name is\s+([A-Z][a-zA-Z\-']+)",text): upsert_fact(conn,"name",m.group(1))
    if m:=re.search(r"\b(i am|i'?m)\s+(\d{1,3})\b",t): upsert_fact(conn,"age",m.group(2))
    if m:=re.search(r"\bmy hair is\s+(\w+)",t): upsert_fact(conn,"hair_color",m.group(1))
    if m:=re.search(r"\bmy eyes are\s+(\w+)",t): upsert_fact(conn,"eye_color",m.group(1))

# ========= Prompt builder (restored working version) =========
def build_rag_prompt(memory_block,user_text,facts):
    facts_block="None." if not facts else "\n".join([f"- {k}: {v}" for k,v in facts.items()])
    return (
        "System role definition:\n"
        "You are an AI assistant named AVA, speaking to the USER (the human).\n"
        "You are distinct from the USER and never share the USER’s attributes.\n"
        "Facts listed below describe the USER only.\n"
        "Maintain correct conversational perspective at all times:\n"
        " - When the USER says 'my', interpret that as referring to the USER.\n"
        " - When the USER says 'your', interpret that as referring to AVA.\n"
        "Do not copy or claim any USER fact as your own identity.\n"
        "Never respond with 'I am' for any USER fact — always refer to the USER.\n\n"
        f"USER facts:\n{facts_block}\n\n"
        f"Recalled conversation snippets:\n{memory_block}\n\n"
        f"USER says: {user_text}\n"
        "Respond naturally in one or two short sentences addressed to the USER."
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
                        messages=[
                            {"role":"system","content":build_rag_prompt(memory,text,facts)},
                            {"role":"user","content":text},
                        ],
                        max_tokens=AI_MAX_TOKENS,temperature=AI_TEMPERATURE,
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
