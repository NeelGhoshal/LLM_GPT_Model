"""Web server: a small page for writing stories with the trained model.

Run locally:
  pip install -r requirements-serve.txt
  uvicorn server:app --reload
Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import os
import threading
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

from model import GPT, GPTConfig

ROOT = Path(__file__).parent
WEIGHTS = Path(os.environ.get("MODEL_PATH", ROOT / "out" / "model.pt"))
MODEL_URL = os.environ.get("MODEL_URL")  # optional: download weights if not on disk

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
torch.set_grad_enabled(False)

if not WEIGHTS.exists():
    if not MODEL_URL:
        raise SystemExit(f"No weights at {WEIGHTS}; commit out/model.pt or set MODEL_URL")
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading weights from {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, WEIGHTS)

tok = Tokenizer.from_file(str(ROOT / "tokenizer.json"))
EOT = tok.token_to_id("<|endoftext|>")
ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
model = GPT(GPTConfig(**ckpt["config"]))
model.load_state_dict(ckpt["model"])
model.float().eval()
del ckpt

# One generation at a time: the CPU is small, and parallel runs only slow each other down.
lock = threading.Lock()

app = FastAPI(title="TinyStories GPT")


class GenerateRequest(BaseModel):
    prompt: str = Field("Once upon a time", max_length=1000)
    max_new_tokens: int = Field(250, ge=1, le=500)
    temperature: float = Field(0.8, ge=0.05, le=2.0)
    top_k: int = Field(50, ge=1, le=500)


def stream_tokens(req: GenerateRequest):
    """Yield server-sent events: one per chunk of new text, then a final 'done'."""
    ids = [EOT] + tok.encode(req.prompt).ids
    ids = ids[-model.cfg.block_size:]
    idx = torch.tensor([ids])
    new: list[int] = []
    sent = ""
    with lock:
        for _ in range(req.max_new_tokens):
            logits, _ = model(idx[:, -model.cfg.block_size:])
            logits = logits[:, -1] / req.temperature
            kth = torch.topk(logits, min(req.top_k, logits.size(-1))).values[:, -1:]
            logits[logits < kth] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            if nxt.item() == EOT:
                break
            idx = torch.cat([idx, nxt], dim=1)
            new.append(nxt.item())
            # Decode everything so far and send only the new part, so multi-token
            # characters and word spacing come out right.
            text = tok.decode(new)
            if len(text) > len(sent) and not text.endswith("�"):
                yield f"data: {json.dumps({'text': text[len(sent):]})}\n\n"
                sent = text
    yield f"data: {json.dumps({'done': True})}\n\n"


@app.post("/api/generate")
def generate(req: GenerateRequest):
    return StreamingResponse(stream_tokens(req), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/healthz")
def healthz():
    return {"ok": True, "params": model.num_params()}


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")
