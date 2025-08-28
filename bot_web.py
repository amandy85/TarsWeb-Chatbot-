# bot_web.py
import os
import time
import asyncio
import logging
from collections import defaultdict
from dotenv import load_dotenv
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# OpenAI / OpenRouter client
import openai
from openai import OpenAI

# ---------- Configuration ----------
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemini-2.0-flash-exp:free")

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
logger = logging.getLogger("tarsweb")

if not OPENROUTER_API_KEY:
    logger.warning("OPENROUTER_API_KEY not found in environment (.env?). The app will still run but model calls will fail.")

# Instantiate OpenRouter/OpenAI client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "https://github.com/your-repo",
        "X-Title": "TarsWeb Chat",
    },
)

# ---------- App & CORS ----------
app = FastAPI(title="TarsWeb Chat API")

# For dev allow all; in production set allowed origins strictly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- In-memory state ----------
CONVERSATIONS: Dict[str, List[Dict[str, str]]] = defaultdict(list)
LAST_ACTIVITY: Dict[str, float] = defaultdict(float)

# Simple per-user request throttling (server-side)
USER_REQUEST_COUNT: Dict[str, int] = defaultdict(int)
LAST_REQUEST_TIME: Dict[str, float] = defaultdict(float)
REQUEST_LIMIT = 6  # number of allowed requests per minute per user
INACTIVITY_TIMEOUT = 30 * 60  # 30 minutes
MAX_HISTORY_ITEMS = 40

# ---------- Request models ----------
class ChatRequest(BaseModel):
    user_id: str
    message: str

class ResetRequest(BaseModel):
    user_id: str

# ---------- Helpers ----------
def _clean_response(text: str) -> str:
    """Minimal cleaning to remove weird escapes if present."""
    if text is None:
        return ""
    # add other cleaning rules if desired
    return text.strip()

async def _call_model(user_id: str) -> str:
    """
    Calls OpenRouter/OpenAI chat completions with exponential backoff for rate limit errors.
    Returns assistant text.
    Raises openai.RateLimitError on persistent rate-limiting (so caller can handle).
    """
    # Build messages: system first, then conversation history
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Keep replies concise and friendly."},
        *CONVERSATIONS.get(user_id, [])
    ]

    def sync_call():
        # This is a synchronous blocking call (the client lib uses blocking IO for now)
        return client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )

    max_attempts = 4
    backoff = 1.0
    last_exc = None

    for attempt in range(1, max_attempts + 1):
        try:
            resp = await asyncio.to_thread(sync_call)
            # safe extraction of content:
            bot_text = ""
            try:
                # library response can vary; preferred shape:
                bot_text = resp.choices[0].message.content
            except Exception:
                # fallback: try other common shapes
                if hasattr(resp, "choices") and len(resp.choices) > 0:
                    choice = resp.choices[0]
                    if hasattr(choice, "message") and getattr(choice.message, "content", None):
                        bot_text = choice.message.content
                    elif getattr(choice, "text", None):
                        bot_text = choice.text
                # as a last resort, stringify
                if not bot_text:
                    bot_text = str(resp)
            return _clean_response(bot_text)
        except openai.RateLimitError as e:
            # Upstream rate limit (HTTP 429). Retry with exponential backoff a few times.
            logger.warning(f"RateLimitError calling model (attempt {attempt}/{max_attempts}): {e}")
            last_exc = e
            if attempt == max_attempts:
                # give up and re-raise so caller returns 503
                raise
            await asyncio.sleep(backoff)
            backoff *= 2
        except Exception as e:
            # Non-rate-limit error: log and re-raise to surface as 500
            logger.exception("Unexpected error while calling model")
            raise

    if last_exc:
        raise last_exc
    raise RuntimeError("Failed to call model for unknown reasons")

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    index_path = os.path.join(os.path.dirname(__file__), "index3.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"detail": "index3.html not found. Place it next to bot_web.py or serve it separately."}

@app.post("/chat")
async def chat(req: ChatRequest):
    uid = req.user_id
    text = (req.message or "").strip()
    if not uid:
        raise HTTPException(status_code=400, detail="user_id is required")
    if not text:
        raise HTTPException(status_code=400, detail="message is required")

    now = time.time()

    # Inactivity-based reset
    if now - LAST_ACTIVITY.get(uid, 0) > INACTIVITY_TIMEOUT and CONVERSATIONS.get(uid):
        logger.info(f"[{uid}] Inactivity timeout: clearing conversation history.")
        CONVERSATIONS[uid] = []

    LAST_ACTIVITY[uid] = now

    # Simple per-minute request throttling
    if now - LAST_REQUEST_TIME.get(uid, 0) < 60:
        if USER_REQUEST_COUNT[uid] >= REQUEST_LIMIT:
            # politely tell client to slow down
            return JSONResponse(
                {"error": "Too many requests from this user. Please wait a minute and try again."},
                status_code=429
            )
    else:
        USER_REQUEST_COUNT[uid] = 0
        LAST_REQUEST_TIME[uid] = now

    USER_REQUEST_COUNT[uid] += 1

    # Append user message to conversation state
    CONVERSATIONS[uid].append({"role": "user", "content": text})
    # Trim if too long
    if len(CONVERSATIONS[uid]) > MAX_HISTORY_ITEMS:
        CONVERSATIONS[uid] = CONVERSATIONS[uid][-MAX_HISTORY_ITEMS:]

    logger.info(f"[{uid}] -> {text}")

    try:
        bot_response = await _call_model(uid)
        # store assistant response
        CONVERSATIONS[uid].append({"role": "assistant", "content": bot_response})
        # trim again if needed
        if len(CONVERSATIONS[uid]) > MAX_HISTORY_ITEMS:
            CONVERSATIONS[uid] = CONVERSATIONS[uid][-MAX_HISTORY_ITEMS:]
        return JSONResponse({"reply": bot_response})
    except openai.RateLimitError as e:
        # Upstream rate-limited; return 503 with Retry-After
        logger.warning(f"Rate-limited while handling /chat for {uid}: {e}")
        friendly = "Sorry — the model is temporarily busy. Please try again shortly."
        # We also append a polite assistant message into local history so UI can display something
        CONVERSATIONS[uid].append({"role": "assistant", "content": friendly})
        return JSONResponse(
            {"error": friendly},
            status_code=503,
            headers={"Retry-After": "60"}
        )
    except Exception as e:
        logger.exception("Error while handling /chat")
        raise HTTPException(status_code=500, detail="Server error: " + str(e))

@app.post("/reset")
async def reset(req: ResetRequest):
    uid = req.user_id
    if not uid:
        raise HTTPException(status_code=400, detail="user_id is required")
    if uid in CONVERSATIONS:
        CONVERSATIONS[uid] = []
    LAST_ACTIVITY[uid] = time.time()
    USER_REQUEST_COUNT[uid] = 0
    LAST_REQUEST_TIME[uid] = 0
    return {"detail": "Conversation reset"}

# ---------- Run helper ----------
if __name__ == "__main__":
    import uvicorn
    # note: using 0.0.0.0 to bind all interfaces; use 127.0.0.1 if you want local-only
    uvicorn.run("bot_web:app", host="0.0.0.0", port=8000, reload=True)
