"""Train Mini-LLM (Mini Transformer) — per-model train/val/test data."""
import argparse
import json
from pathlib import Path
import torch

from env import get_device, info
from tokenizer.char import CharTokenizer
from data.loader import load_dataset, build_loaders, build_loader
from model.transformer import MiniTransformer
from training.engine import train
from eval.evaluate import evaluate
from generate import generate_text

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent
DEFAULT_TRAIN = PROJ / "public" / "data" / "models" / "default" / "train.json"


def resolve_data(path_str, model_id="default", kind="train"):
    if not path_str:
        return None
    p = Path(path_str)
    if p.exists():
        return p
    p2 = PROJ / path_str
    if p2.exists():
        return p2
    norm = path_str.replace("\\", "/")
    for old, mk in (("public/data/train.json", "train"), ("public/data/test.json", "test"), ("public/data/val.json", "val"),
                    ("public/data/train.txt", "train"), ("public/data/test.txt", "test"), ("public/data/val.txt", "val")):
        if norm.endswith(old):
            for ext in (".json", ".txt"):
                p3 = PROJ / "public" / "data" / "models" / model_id / f"{mk}{ext}"
                if p3.exists():
                    return p3
                p4 = PROJ / "public" / "data" / "models" / "default" / f"{mk}{ext}"
                if p4.exists():
                    return p4
    for ext in (".json", ".txt"):
        p5 = PROJ / "public" / "data" / "models" / model_id / f"{kind}{ext}"
        if p5.exists():
            return p5
        p5b = PROJ / "public" / "data" / "models" / "default" / f"{kind}{ext}"
        if p5b.exists():
            return p5b
    for alt in (["val", "test"] if kind in ("val", "test") else []):
        for ext in (".json", ".txt"):
            p6 = PROJ / "public" / "data" / "models" / model_id / f"{alt}{ext}"
            if p6.exists():
                return p6
    raise FileNotFoundError(f"Data not found: {path_str} (tried {p} and {p2})")


def eval_file(model, tok, path, seq_len, batch_size, device):
    text = load_dataset(str(path))
    loader, stats = build_loader(text, tok, seq_len=seq_len, batch_size=batch_size, shuffle=False)
    return {**evaluate(model, loader, device), **stats, "data": str(path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_TRAIN))
    ap.add_argument("--val-data", default=None)
    ap.add_argument("--test-data", default=None)
    ap.add_argument("--model-id", default="default")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--prompt", default="คำถาม: สวัสดีครับ\nคำตอบ:")
    ap.add_argument("--top-k", type=int, default=24, dest="top_k")
    ap.add_argument("--temp", type=float, default=0.7, dest="temperature")
    ap.add_argument("--rep-penalty", type=float, default=1.15, dest="repetition_penalty")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    train_path = resolve_data(args.data, args.model_id, "train")
    val_path = None
    try:
        val_path = resolve_data(args.val_data, args.model_id, "val") if args.val_data else resolve_data(str(PROJ / "public" / "data" / "models" / args.model_id / "val.json"), args.model_id, "val")
    except FileNotFoundError:
        val_path = None
    test_path = None
    try:
        test_path = resolve_data(args.test_data, args.model_id, "test") if args.test_data else resolve_data(str(PROJ / "public" / "data" / "models" / args.model_id / "test.json"), args.model_id, "test")
    except FileNotFoundError:
        test_path = None
    out = Path(args.out) if args.out else (ROOT / "checkpoints" / f"{args.model_id}.pt")
    print(json.dumps({"env": info(), "model_id": args.model_id, "train_data": str(train_path), "val_data": str(val_path) if val_path else None, "test_data": str(test_path) if test_path else None, "out": str(out)}, ensure_ascii=False))

    device = get_device()
    train_text = load_dataset(str(train_path))
    tok = CharTokenizer().fit(train_text)

    if val_path:
        train_loader, train_stats = build_loader(train_text, tok, seq_len=args.seq_len, batch_size=args.batch_size, shuffle=True)
        val_text = load_dataset(str(val_path))
        try:
            val_loader, val_stats = build_loader(val_text, tok, seq_len=args.seq_len, batch_size=args.batch_size, shuffle=False)
            stats = {"tokens": train_stats["tokens"], "train_samples": train_stats["samples"], "val_samples": val_stats["samples"], "val_tokens": val_stats["tokens"], "split": "file"}
        except ValueError:
            train_loader, val_loader, stats = build_loaders(train_text, tok, seq_len=args.seq_len, batch_size=args.batch_size)
            stats["split"] = "ratio"
            val_loader = None
    else:
        train_loader, val_loader, stats = build_loaders(train_text, tok, seq_len=args.seq_len, batch_size=args.batch_size)
        stats["split"] = "ratio"

    print(json.dumps({"vocab_size": tok.vocab_size, **stats}, ensure_ascii=False))

    model = MiniTransformer(tok.vocab_size, seq_len=args.seq_len, n_embd=args.embed, n_head=args.heads, n_layer=args.layers, dropout=args.dropout)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    extra = {"n_embd": args.embed, "n_head": args.heads, "n_layer": args.layers, "dropout": args.dropout, "train_data": str(train_path), "model_id": args.model_id}
    history, ckpt = train(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr, ckpt_dir=str(out.parent), ckpt_name=out.name, tokenizer=tok, extra_config=extra)
    result = {"model_id": args.model_id, "checkpoint": ckpt, "train_data": str(train_path), "val_data": str(val_path) if val_path else None, "test_data": str(test_path) if test_path else None, "history": history}
    if val_loader:
        result["validation"] = evaluate(model, val_loader, device)
        if val_path:
            result["validation"]["data"] = str(val_path)
    if test_path:
        try:
            result["test"] = eval_file(model, tok, test_path, args.seq_len, args.batch_size, device)
        except ValueError:
            result["test"] = {"error": "test data too short for seq_len", "data": str(test_path)}
    result["sample"] = generate_text(model, tok, args.prompt, device, max_new_tokens=80, temperature=args.temperature, top_k=args.top_k, repetition_penalty=args.repetition_penalty)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
