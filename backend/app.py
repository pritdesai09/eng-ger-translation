from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.translator import translate
import time

app = FastAPI(
    title="EN ↔ DE Translation API",
    description="Neural Machine Translation using Seq2Seq and Transformer",
    version="1.0.0"
)

# ── CORS (allow frontend to call this API) ────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response schemas ────────────────────────────────────────────────
class TranslateRequest(BaseModel):
    text:       str
    model_type: str  # 'seq2seq' | 'transformer'
    direction:  str  # 'en_de'   | 'de_en'

class TranslateResponse(BaseModel):
    translation:   str
    source_tokens: list
    model:         str
    direction:     str
    latency_ms:    float

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "message": "Translation API is running"}

@app.get("/models")
def list_models():
    return {
        "models": [
            {"id": "seq2seq",      "name": "Seq2Seq LSTM (No Attention)"},
            {"id": "transformer",  "name": "Transformer (Multi-Head Attention)"},
        ],
        "directions": [
            {"id": "en_de", "label": "English → German"},
            {"id": "de_en", "label": "German → English"},
        ]
    }

@app.post("/translate", response_model=TranslateResponse)
def translate_text(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if req.model_type not in ('seq2seq', 'transformer'):
        raise HTTPException(status_code=400, detail="Invalid model_type.")
    if req.direction not in ('en_de', 'de_en'):
        raise HTTPException(status_code=400, detail="Invalid direction.")

    t0     = time.time()
    result = translate(req.text, req.model_type, req.direction)
    latency = (time.time() - t0) * 1000

    return TranslateResponse(**result, latency_ms=round(latency, 2))