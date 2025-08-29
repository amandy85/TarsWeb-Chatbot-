# bot_web.py
import os
import time
import asyncio
import logging
import re
import html
from collections import defaultdict
from dotenv import load_dotenv
from typing import Dict, List

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
def _strip_html_code_tags(text: str) -> str:
    """
    Remove common HTML code tags (<code>, <pre>, <span class="..."> that wrap code)
    while preserving inner text. Also unescape HTML entities.
    """
    if not text:
        return text
    # remove <code ...> and </code>, <pre ...> and </pre>
    text_no_code = re.sub(r"</?(code|pre)(?:\s[^>]*)?>", "", text, flags=re.IGNORECASE)
    # sometimes models wrap inline elements like <span class="inline-code">...</span>
    text_no_spans = re.sub(r"</?span(?:\s[^>]*)?>", "", text_no_code, flags=re.IGNORECASE)
    # remove other small markup but keep text
    text_no_tags = re.sub(r"<[^>]+>", "", text_no_spans)
    # Unescape HTML entities (&lt; &gt; &amp;)
    text_unescaped = html.unescape(text_no_tags)
    return text_unescaped

def _detect_language_hint(text: str) -> str:
    """
    Heuristic detection of code language for fenced code blocks.
    Returns a short language hint (e.g., 'python', 'java', 'cpp') or empty string.
    """
    t = text.strip()
    # Check for Java-ish
    if re.search(r"\b(public|private|protected)\s+(class|static|void)\b", t) or "System.out.println" in t or "public class " in t:
        return "java"
    # Python-ish
    if re.search(r"^\s*def\s+\w+\s*\(", t, flags=re.MULTILINE) or re.search(r"^\s*class\s+\w+\s*:", t, flags=re.MULTILINE) or re.search(r"import\s+\w+", t):
        return "python"
    # JavaScript / Node
    if re.search(r"\bconsole\.log\b", t) or re.search(r"function\s*\w*\s*\(", t) or re.search(r"=>", t):
        return "javascript"
    # C / C++
    if re.search(r"#include\s*<", t) or re.search(r"\bint\s+main\s*\(", t):
        return "cpp"
    # SQL-ish
    if re.search(r"\bSELECT\b|\bFROM\b|\bWHERE\b", t, flags=re.IGNORECASE):
        return "sql"
    # Shell
    if re.search(r"^#!/bin/(ba|sh)", t):
        return "bash"
    return ""

def _looks_like_code(text: str) -> bool:
    """
    Determine whether the assistant response looks like code or contains a code snippet.
    Use several heuristics: presence of multiple lines, semicolons/braces (for C-like languages),
    code keywords, or HTML <code> tags.
    """
    if not text:
        return False
    # If already contains fenced code, treat as code
    if "```" in text:
        return True
    # If contains HTML code tags, treat as code
    if re.search(r"<\s*(code|pre)(\s|>)", text, flags=re.IGNORECASE):
        return True
    # multiple lines and typical code symbols
    lines = text.splitlines()
    if len(lines) >= 2:
        # count lines that look like code (ending with ';', contain '{' or '}', or contain 'def ' etc.)
        code_like = 0
        for ln in lines[:30]:  # check first 30 lines only
            if ln.strip().endswith(";") or "{" in ln or "}" in ln or ln.strip().startswith(("def ", "class ", "import ", "#include")):
                code_like += 1
        if code_like >= 1:
            return True
    # single-line code patterns (like package names, function prototype)
    if re.search(r"\bfunc\b|\bdef\b|\bclass\b|\breturn\b", text):
        return True
    return False

def _wrap_in_fenced_block(text: str, lang_hint: str = "") -> str:
    """
    Wrap the provided text in a Markdown fenced code block.
    If lang_hint is present, it is appended to the opening fence.
    """
    # Ensure no accidental triple-backticks in inner text (escape by using ~~~ if needed)
    if "```" in text:
        # If the output already includes triple backticks inside, use ~~~ as the fence
        fence = "~~~"
    else:
        fence = "```"
    lang = lang_hint.strip()
    if lang:
        return f"{fence}{lang}\n{text.rstrip()}\n{fence}"
    else:
        return f"{fence}\n{text.rstrip()}\n{fence}"

def _clean_response(text: str) -> str:
    """
    Clean model output:
    - Strip weird whitespace
    - Remove HTML code/pre/span wrappers and unescape entities
    - If the text looks like code, wrap in fenced code block with a language hint when possible
    - If the text already contains a fenced block, leave it mostly untouched (but still strip HTML wrappers)
    """
    if text is None:
        return ""
    # Trim surrounding whitespace
    cleaned = text.strip()

    # If it contains HTML code tags (OpenAI sometimes returns <code>), strip them but still preserve the code text
    if re.search(r"<\s*(code|pre|span)(\s|>)", cleaned, flags=re.IGNORECASE):
        cleaned = _strip_html_code_tags(cleaned).strip()

    # If there are HTML entities like &lt; &gt; decode them
    cleaned = html.unescape(cleaned)

    # If text already contains Markdown fences, do minimal touching (but ensure no HTML leftover)
    if "```" in cleaned or "~~~" in cleaned:
        # Optionally ensure the fenced block has a language hint; attempt to add if missing.
        m = re.search(r"(^|\n)(?P<fence>```|~~~)(?P<lang>\w+)?\n?(?P<body>.*?)(?P=fence)", cleaned, flags=re.DOTALL)
        if m:
            if not m.group("lang"):
                lang = _detect_language_hint(m.group("body")) or ""
                if lang:
                    cleaned = cleaned.replace(m.group(0), f"\n{m.group('fence')}{lang}\n{m.group('body')}{m.group('fence')}", 1)
        return cleaned.strip()

    # If the assistant output looks like code, wrap it
    if _looks_like_code(cleaned):
        lang_hint = _detect_language_hint(cleaned)
        return _wrap_in_fenced_block(cleaned, lang_hint)

    # Otherwise return cleaned plain text
    return cleaned

async def _call_model(user_id: str) -> str:
    """
    Calls OpenRouter/OpenAI chat completions with exponential backoff for rate limit errors.
    Returns assistant text (cleaned).
    Raises openai.RateLimitError on persistent rate-limiting (so caller can handle).
    """
    # Build messages: system first, then conversation history
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Keep replies concise and friendly."},
        *CONVERSATIONS.get(user_id, [])
    ]

    def sync_call():
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
            bot_text = ""
            try:
                bot_text = resp.choices[0].message.content
            except Exception:
                if hasattr(resp, "choices") and len(resp.choices) > 0:
                    choice = resp.choices[0]
                    if hasattr(choice, "message") and getattr(choice.message, "content", None):
                        bot_text = choice.message.content
                    elif getattr(choice, "text", None):
                        bot_text = choice.text
                if not bot_text:
                    bot_text = str(resp)
            return _clean_response(bot_text)
        except openai.RateLimitError as e:
            logger.warning(f"RateLimitError calling model (attempt {attempt}/{max_attempts}): {e}")
            last_exc = e
            if attempt == max_attempts:
                raise
            await asyncio.sleep(backoff)
            backoff *= 2
        except Exception as e:
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
    if len(CONVERSATIONS[uid]) > MAX_HISTORY_ITEMS:
        CONVERSATIONS[uid] = CONVERSATIONS[uid][-MAX_HISTORY_ITEMS:]

    logger.info(f"[{uid}] -> {text}")

    try:
        bot_response = await _call_model(uid)
        # store assistant response (we store the cleaned text so history shows what was sent)
        CONVERSATIONS[uid].append({"role": "assistant", "content": bot_response})
        if len(CONVERSATIONS[uid]) > MAX_HISTORY_ITEMS:
            CONVERSATIONS[uid] = CONVERSATIONS[uid][-MAX_HISTORY_ITEMS:]
        return JSONResponse({"reply": bot_response})
    except openai.RateLimitError as e:
        logger.warning(f"Rate-limited while handling /chat for {uid}: {e}")
        friendly = "Sorry — the model is temporarily busy. Please try again shortly."
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
    uvicorn.run("bot_web:app", host="0.0.0.0", port=8000, reload=True)
