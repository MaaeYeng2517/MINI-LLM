"""Evaluate a saved checkpoint on train/val/test data."""
import argparse
import json
from pathlib import Path
import torch

from env import get_device, info
from tokenizer.thai_subword import ThaiCharTokenizer
from data.loader import load_dataset, build_loader
from model.transformer import MiniTransformer
from eval.evaluate import evaluate

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent


def resolve(p):
    p = Path(p)
    return p if p.exists() else (PROJ / p)


def eval_split(model, tok, path, seq_len, batch_size, device):
    loader, stats = build_loader(load_dataset(str(path)), tok, seq_len=seq_len, batch_size=batch_size)
    return {**evaluate(model, loader, device), **stats, "data": str(path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="default")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data", default=None, help="Tokenizer source (default: per-model train.txt)")
    ap.add_argument("--train-data", default=None)
    ap.add_argument("--val-data", default=None)
    ap.add_argument("--test-data", default=None)
    ap.add_argument("--seq-len", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    ckpt_path = resolve(args.ckpt or (ROOT / "checkpoints" / f"{args.model_id}.pt"))
    data_path = resolve(args.data or args.train_data or (PROJ / "public/data/models" / args.model_id / "train.json"))
    val_path = resolve(args.val_data or (PROJ / "public/data/models" / args.model_id / "val.json"))
    test_path = resolve(args.test_data or (PROJ / "public/data/models" / args.model_id / "test.json"))
    device = get_device()
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = ckpt.get("config") or {}
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    tok = ThaiCharTokenizer()
    if isinstance(ckpt, dict) and ckpt.get("tokenizer"):
        tok.load_state(ckpt["tokenizer"])
    else:
        tok.fit(load_dataset(str(data_path)))
    w = state.get("tok_emb.weight")
    if w is not None and int(w.shape[0]) != tok.vocab_size:
        n = int(w.shape[0]); vocab = list(tok.vocab)
        if len(vocab) < n: vocab += [f"<u{i}>" for i in range(len(vocab), n)]
        tok.load_state({"vocab": vocab[:n]})
    seq_len = int(cfg.get("seq_len", args.seq_len))
    model = MiniTransformer(tok.vocab_size, seq_len=seq_len, n_embd=int(cfg.get("n_embd", 96)), n_head=int(cfg.get("n_head", 4)), n_layer=int(cfg.get("n_layer", 2)), dropout=float(cfg.get("dropout", 0.1)))
    model.load_state_dict(state, strict=False)
    model.to(device)

    result = {"env": info(), "model_id": args.model_id, "checkpoint": str(ckpt_path), "tokenizer_data": str(data_path)}
    if data_path.exists():
        result["train_eval"] = eval_split(model, tok, data_path, seq_len, args.batch_size, device)
    if val_path.exists():
        result["val_eval"] = eval_split(model, tok, val_path, seq_len, args.batch_size, device)
    if test_path.exists():
        result["test_eval"] = eval_split(model, tok, test_path, seq_len, args.batch_size, device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
