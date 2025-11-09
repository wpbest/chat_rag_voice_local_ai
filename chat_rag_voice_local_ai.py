# chat_voice_local_ai_rag_debug.py
import sys, os, time, struct, sqlite3, logging, re, traceback
import speech_recognition as sr, pyttsx3, sqlite_vec
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ========= Config =========
AI_TOOLKIT_BASE_URL = os.environ.get("AI_TOOLKIT_BASE_URL", "http://127.0.0.1:5272/v1/")
AI_TOOLKIT_MODEL    = os.environ.get("AI_TOOLKIT_MODEL", "deepseek-r1-distill-qwen-1.5b-cpu-int4-rtn-block-32-acc-level-4")
AI_TEMPERATURE      = float(os.environ.get("AI_TEMPERATURE", "0.3"))
AI_MAX_TOKENS       = int(os.environ.get("AI_MAX_TOKENS", "180"))
STRIP_THINK_FOR_TTS = True  # strip <think>...</think> before speaking, but keep it in logs/db

DB_FILE      = os.environ.get("CHAT_DB_FILE", "chat_memory.db")
VEC_TABLE    = "messages_vec"
META_TABLE   = "messages_meta"
EMB_MODEL    = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMB_DIMS     = int(os.environ.get("EMBEDDING_DIMS", "384"))
TOP_K        = int(os.environ.get("TOP_K", "5"))
MAX_SNIPPET  = int(os.environ.get("MAX_SNIPPET", "240"))

PHRASE_TIME_LIMIT        = int(os.environ.get("PHRASE_TIME_LIMIT", "7"))
AMBIENT_NOISE_DURATION   = float(os.environ.get("AMBIENT_NOISE_DURATION", "0.4"))

# ========= Logging =========
LOG_FILE = os.environ.get("CHAT_LOG_FILE", "chat_memory.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger("AVA-RAG")

# ========= Clients =========
_ai_client = OpenAI(base_url=AI_TOOLKIT_BASE_URL, api_key=os.environ.get("AI_DUMMY_KEY", "unused"))

# ========= Embeddings =========
_model = None
def get_model():
    global _model
    if _model is None:
        t0 = time.perf_counter()
        log.info(f"Loading embedding model: {EMB_MODEL}")
        _model = SentenceTransformer(EMB_MODEL)
        dt = (time.perf_counter() - t0) * 1000
        log.info(f"Embedding model ready in {dt:.1f} ms")
    return _model

def embed(text: str):
    t0 = time.perf_counter()
    vec = get_model().encode(text, normalize_embeddings=True).tolist()
    dt = (time.perf_counter() - t0) * 1000
    log.info(f"Embedded text len={len(text)} chars -> {len(vec)} dims in {dt:.1f} ms")
    return vec

def serialize_f32(vec):
    return struct.pack("%sf" % len(vec), *vec)

# ========= SQLite / Vector =========
def ensure_db():
    t0 = time.perf_counter()
    conn = sqlite3.connect(DB_FILE)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {VEC_TABLE} USING vec0(embedding float[{EMB_DIMS}])"
    )
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS {META_TABLE}(
            rowid INTEGER PRIMARY KEY,
            ts    REAL NOT NULL,
            role  TEXT NOT NULL,
            text  TEXT NOT NULL
        )"""
    )
    conn.commit()

    # Stats
    c1 = conn.execute(f"SELECT COUNT(*) FROM {META_TABLE}").fetchone()[0]
    c2 = conn.execute(f"SELECT COUNT(*) FROM {VEC_TABLE}").fetchone()[0]
    dt = (time.perf_counter() - t0) * 1000
    log.info(f"Database initialized and verified in {dt:.1f} ms. meta_rows={c1}, vec_rows={c2}")
    conn.close()

def db_connect():
    conn = sqlite3.connect(DB_FILE)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn

def remember(conn, role, text):
    t0 = time.perf_counter()
    vec = embed(text)
    cur = conn.execute(f"INSERT INTO {VEC_TABLE}(embedding) VALUES (?)", (serialize_f32(vec),))
    rid = cur.lastrowid
    conn.execute(
        f"INSERT INTO {META_TABLE}(rowid,ts,role,text) VALUES (?,?,?,?)",
        (rid, time.time(), role, text),
    )
    conn.commit()
    dt = (time.perf_counter() - t0) * 1000
    snippet = (text[:MAX_SNIPPET] + "…") if len(text) > MAX_SNIPPET else text
    log.info(f"Remembered rowid={rid} role={role} ({len(text)} chars) in {dt:.1f} ms :: {snippet!r}")

def recall(conn, query, k=TOP_K):
    t0 = time.perf_counter()
    qv = embed(query)
    rows = conn.execute(
        f"SELECT rowid,distance FROM {VEC_TABLE} WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        (serialize_f32(qv), k),
    ).fetchall()
    qdt = (time.perf_counter() - t0) * 1000
    log.info(f"Vector search: k={k}, hits={len(rows)} in {qdt:.1f} ms")
    if not rows:
        return []

    ids = ",".join(str(rid) for rid, _ in rows)
    meta = conn.execute(
        f"SELECT rowid,ts,role,text FROM {META_TABLE} WHERE rowid IN ({ids})"
    ).fetchall()

    # Map rowid -> meta
    lookup = {r: (ts, ro, tx) for r, ts, ro, tx in meta}
    # Compose detailed list with distances
    detailed = []
    for rid, dist in rows:
        ts, ro, tx = lookup.get(rid, (0.0, "unknown", ""))
        snip = (tx[:MAX_SNIPPET] + "…") if len(tx) > MAX_SNIPPET else tx
        detailed.append({
            "rowid": rid,
            "distance": float(dist),
            "ts": ts,
            "role": ro,
            "text": tx,
            "snippet": snip,
        })

    # Log the recalls in detail
    for i, d in enumerate(detailed, 1):
        log.info(
            f"Recall[{i}/{len(detailed)}]: rowid={d['rowid']} role={d['role']} "
            f"dist={d['distance']:.4f} ts={d['ts']:.3f} :: {d['snippet']!r}"
        )
    return detailed

# ========= Prompting =========
def build_messages(memory_items, user_text):
    # Memory section
    if memory_items:
        mem_lines = [f"- {m['role']}: {m['text']}" for m in memory_items]
        memory_block = "\n".join(mem_lines)
    else:
        memory_block = "None."

    user_block = (
        f"Relevant information you recall from prior conversation:\n"
        f"{memory_block}\n\n"
        f"User just said: {user_text}\n\n"
        f"Use any relevant context naturally in your reply."
    )

    # System defines tone/behavior clearly
    system_block = (
        "You are AVA, a highly courteous, emotionally intelligent, and engaging conversational assistant. "
        "Respond with warmth, professionalism, and natural curiosity. "
        "Acknowledge user details when relevant and keep dialogue flowing with brief, thoughtful follow-up questions. "
        "Never mention that you are an AI or voice assistant unless asked directly."
    )

    # Log the full blocks (verbatim)
    log.info("=== RAG MEMORY BLOCK BEGIN ===")
    log.info("\n" + memory_block)
    log.info("=== RAG MEMORY BLOCK END ===")

    log.info("=== USER BLOCK BEGIN ===")
    log.info("\n" + user_block)
    log.info("=== USER BLOCK END ===")

    return [
        {"role": "system", "content": system_block},
        {"role": "user",   "content": user_block},
    ]

def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# ========= Voice Loop =========
def listen_and_respond():
    log.info(f"Starting AVA (RAG + Deep Debug) on Python {sys.version}")
    ensure_db()
    r = sr.Recognizer()

    # Mic setup + calibration
    with sr.Microphone() as s:
        log.info("Calibrating ambient noise level...")
        t0 = time.perf_counter()
        r.adjust_for_ambient_noise(s, duration=AMBIENT_NOISE_DURATION)
        log.info(f"Energy threshold set to: {r.energy_threshold:.2f} (calib {(time.perf_counter()-t0)*1000:.1f} ms)")
        print("Ready. Speak when you hear 'Listening...'")

        while True:
            try:
                print("Listening...")
                t_listen0 = time.perf_counter()
                audio = r.listen(s, timeout=None, phrase_time_limit=PHRASE_TIME_LIMIT)
                listen_ms = (time.perf_counter() - t_listen0) * 1000
                log.info(f"Audio captured in {listen_ms:.1f} ms; frames={len(audio.frame_data)}")

                if len(audio.frame_data) == 0:
                    log.warning("Captured empty audio; skipping.")
                    continue

                t_sr0 = time.perf_counter()
                text = r.recognize_google(audio)
                sr_ms = (time.perf_counter() - t_sr0) * 1000
                log.info(f"ASR: {sr_ms:.1f} ms :: {text!r} (len={len(text)})")

                conn = db_connect()

                # DB stats pre
                pre_meta = conn.execute(f"SELECT COUNT(*) FROM {META_TABLE}").fetchone()[0]
                pre_vec  = conn.execute(f"SELECT COUNT(*) FROM {VEC_TABLE}").fetchone()[0]
                log.info(f"DB before recall: meta_rows={pre_meta}, vec_rows={pre_vec}")

                # === RAG Recall ===
                t_recall0 = time.perf_counter()
                memory_items = recall(conn, text, TOP_K)
                recall_ms = (time.perf_counter() - t_recall0) * 1000
                log.info(f"Recall pipeline finished in {recall_ms:.1f} ms")

                # === Build Messages ===
                messages = build_messages(memory_items, text)

                # === Model Call ===
                t_llm0 = time.perf_counter()
                chat = _ai_client.chat.completions.create(
                    model=AI_TOOLKIT_MODEL,
                    messages=messages,
                    max_tokens=AI_MAX_TOKENS,
                    temperature=AI_TEMPERATURE,
                )
                llm_ms = (time.perf_counter() - t_llm0) * 1000

                raw_resp = chat.choices[0].message.content or ""
                log.info(f"LLM call: {llm_ms:.1f} ms; resp_len={len(raw_resp)} chars")
                log.info("=== RAW LLM RESPONSE BEGIN ===")
                log.info("\n" + raw_resp)
                log.info("=== RAW LLM RESPONSE END ===")

                # === Save to DB ===
                t_save0 = time.perf_counter()
                remember(conn, "user", text)
                remember(conn, "assistant", raw_resp)
                post_meta = conn.execute(f"SELECT COUNT(*) FROM {META_TABLE}").fetchone()[0]
                post_vec  = conn.execute(f"SELECT COUNT(*) FROM {VEC_TABLE}").fetchone()[0]
                save_ms = (time.perf_counter() - t_save0) * 1000
                log.info(f"DB after save: meta_rows={post_meta}, vec_rows={post_vec} (save {save_ms:.1f} ms)")

                # === Speak (optionally strip <think>) ===
                spoken = strip_think(raw_resp) if STRIP_THINK_FOR_TTS else raw_resp
                log.info("=== SPOKEN TEXT BEGIN ===")
                log.info("\n" + spoken)
                log.info("=== SPOKEN TEXT END ===")

                try:
                    ttts0 = time.perf_counter()
                    tts = pyttsx3.init()
                    tts.setProperty("volume", 1.0)
                    tts.say(spoken)
                    tts.runAndWait()
                    tts.stop()
                    del tts
                    tts_ms = (time.perf_counter() - ttts0) * 1000
                    log.info(f"TTS playback finished in {tts_ms:.1f} ms")
                except Exception as e:
                    log.error(f"TTS playback error: {e}\n{traceback.format_exc()}")

                conn.close()

            except sr.UnknownValueError:
                log.warning("ASR: Speech not recognized (low confidence).")
                continue
            except sr.RequestError as e:
                log.error(f"ASR Request error: {e}")
                continue
            except KeyboardInterrupt:
                log.info("User interrupted. Exiting.")
                break
            except Exception as e:
                log.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
                continue

# ========= Main =========
if __name__ == "__main__":
    listen_and_respond()
