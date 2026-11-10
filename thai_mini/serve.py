"""JSON stdin/stdout inference server helper for Next.js API route."""
import json
import os
import sys
from pathlib import Path
import torch

try:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from env import get_device
from tokenizer.char import CharTokenizer
from data.loader import load_dataset
from model.transformer import MiniTransformer
from generate import generate_text

DATA = Path(os.environ.get("THAI_MINI_DATA") or (ROOT.parent / "public" / "data" / "models" / "default" / "train.json"))
CKPT = Path(os.environ.get("THAI_MINI_CKPT") or (ROOT / "checkpoints" / "mini_transformer.pt"))
FALLBACK_DATA = [ROOT.parent / "public" / "data" / "models" / "default" / "train.json", ROOT.parent / "public" / "data" / "train.json"]
STOP_MARKERS = ("<|question|>", "<|answer|>", "<|sep|>", "\nคำถาม:", "\nผู้ใช้:", "\nUser:", "\nQ:", "\nคำตอบ:", "\nผู้ช่วย:", "\nBot:")


def out(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def fail(msg):
    out({"ok": False, "error": msg, "checkpoint": str(CKPT), "data": str(DATA)})
    raise SystemExit(0)


def load_tokenizer(ckpt, data_path):
    tok = CharTokenizer()
    tstate = ckpt.get("tokenizer") if isinstance(ckpt, dict) else None
    if tstate:
        tok.load_state(tstate)
        return tok, "checkpoint"
    for p in ([data_path] if data_path else []) + FALLBACK_DATA:
        if p and Path(p).exists():
            tok.fit(load_dataset(str(p)))
            return tok, f"data:{Path(p).name}"
    fail(f"tokenizer/data source not found: {DATA}")


def trim(gen):
    s = gen
    for m in STOP_MARKERS:
        i = s.find(m)
        if i >= 0:
            s = s[:i]
    return s.strip()


def main():
    try:
        raw = sys.stdin.buffer.read()
        req = json.loads(raw.decode("utf-8", errors="replace")) if raw else {}
    except Exception as e:
        fail(f"invalid json: {e}")
    prompt = str(req.get("prompt", ""))
    max_new = max(1, min(int(req.get("max_new_tokens", 80)), 400))
    temp = max(0.05, min(float(req.get("temperature", 0.7)), 2.0))
    top_k = max(1, min(int(req.get("top_k", 24)), 200))
    rep = max(1.0, min(float(req.get("repetition_penalty", 1.15)), 2.0))

    if not CKPT.exists():
        fail(f"checkpoint not found: {CKPT}")
    device = get_device()
    ckpt = torch.load(str(CKPT), map_location=device, weights_only=False)
    cfg = ckpt.get("config") or {}
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    data_path = DATA if DATA.exists() else next((p for p in FALLBACK_DATA if p.exists()), DATA)
    tok, tok_src = load_tokenizer(ckpt, data_path)
    w = state.get("tok_emb.weight")
    ckpt_vocab = int(w.shape[0]) if w is not None and hasattr(w, "shape") else tok.vocab_size
    if ckpt_vocab != tok.vocab_size and not (isinstance(ckpt, dict) and ckpt.get("tokenizer")):
        chars = list(getattr(tok, "chars", []) or [])
        if len(chars) < ckpt_vocab:
            chars += [f"<u{i}>" for i in range(len(chars), ckpt_vocab)]
        tok.load_state({"chars": chars[:ckpt_vocab], "special": getattr(tok, "special", [])})
        tok_src += "+adjusted"

    model = MiniTransformer(tok.vocab_size, seq_len=int(cfg.get("seq_len", 64)), n_embd=int(cfg.get("n_embd", 128)), n_head=int(cfg.get("n_head", 4)), n_layer=int(cfg.get("n_layer", 3)), dropout=float(cfg.get("dropout", 0.1)))
    model.load_state_dict(state, strict=True)
    model.to(device)
    full = generate_text(model, tok, prompt, device, max_new_tokens=max_new, temperature=temp, top_k=top_k, repetition_penalty=rep)
    gen = trim(full[len(prompt):] if full.startswith(prompt) else full)
    out({"ok": True, "output": gen, "full": full, "checkpoint": str(CKPT), "data": str(data_path), "device": str(device), "vocab_size": tok.vocab_size, "tokenizer": tok_src, "config": {"seq_len": cfg.get("seq_len"), "best_epoch": cfg.get("best_epoch")}})


if __name__ == "__main__":
    main()
