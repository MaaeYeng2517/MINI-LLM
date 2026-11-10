"""Load checkpoint and generate text with Output Engine support."""
import argparse
import json
from pathlib import Path
import torch
from env import get_device, info
from tokenizer.thai_subword import ThaiCharTokenizer
from data.loader import load_text, load_dataset
from model.transformer import MiniTransformer
from generate import generate_text, generate_with_output_engine
from tokenizer.output_engine import OutputEngine, OutputFormat


ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN = ROOT.parent / "public" / "data" / "models" / "default" / "train.json"
DEFAULT_CKPT = ROOT / "checkpoints" / "mini_transformer.pt"


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
    for ext in (".json", ".txt"):
        p3 = PROJ / "public" / "data" / "models" / model_id / f"{kind}{ext}"
        if p3.exists():
            return p3
        p4 = PROJ / "public" / "data" / "models" / "default" / f"{kind}{ext}"
        if p4.exists():
            return p4
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_TRAIN), help="Tokenizer source text (used if checkpoint has no tokenizer)")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--prompt", default="สวัสดี")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=24)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--output-dir", default="outputs", help="Output directory for generated text")
    ap.add_argument("--output-format", choices=["txt", "json", "jsonl", "csv"], default="jsonl", help="Output format")
    ap.add_argument("--save-output", action="store_true", help="Save generated output to file")
    ap.add_argument("--no-unicode-validation", action="store_true", help="Disable Unicode validation")
    ap.add_argument("--no-normalize", action="store_true", help="Disable NFC normalization")
    ap.add_argument("--show-tokens", action="store_true", help="Show token IDs in output")
    ap.add_argument("--show-graphemes", action="store_true", help="Show grapheme clusters in output")
    args = ap.parse_args()

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt.get("config") or {}
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    tok = ThaiCharTokenizer()
    if isinstance(ckpt, dict) and ckpt.get("tokenizer"):
        tok.load_state(ckpt["tokenizer"])
    else:
        tok.fit(load_dataset(args.data))
    # align vocab to weights if needed
    w = state.get("tok_emb.weight")
    if w is not None and int(w.shape[0]) != tok.vocab_size:
        n = int(w.shape[0]); vocab = list(tok.vocab)
        if len(vocab) < n:
            vocab += [f"<u{i}>" for i in range(len(vocab), n)]
        tok.load_state({"vocab": vocab[:n]})
    model = MiniTransformer(tok.vocab_size, seq_len=int(cfg.get("seq_len", 48)), n_embd=int(cfg.get("n_embd", 96)), n_head=int(cfg.get("n_head", 4)), n_layer=int(cfg.get("n_layer", 2)), dropout=float(cfg.get("dropout", 0.1)))
    model.load_state_dict(state, strict=False)
    model.to(device)

    # Create output engine
    output_engine = OutputEngine(
        decoder=tok,  # ThaiCharTokenizer inherits from TuringEncoder which has decode
        output_dir=args.output_dir,
        validate_unicode=not args.no_unicode_validation,
        normalize_output=not args.no_normalize,
    )

    # Generate with output engine
    result = generate_with_output_engine(
        model=model,
        tokenizer=tok,
        prompt=args.prompt,
        device=device,
        output_engine=output_engine,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        save_output=args.save_output,
        output_format=OutputFormat(args.output_format),
    )

    # Display to terminal
    if args.show_tokens:
        print(f"Tokens: {result.token_ids}")
    if args.show_graphemes:
        print(f"Graphemes: {output_engine.display_to_terminal(result.generated_text, show_graphemes=True)}")
    else:
        print(f"Generated: {output_engine.display_to_terminal(result.generated_text)}")

    # Print full result as JSON
    output_dict = {
        "env": info(),
        "checkpoint": args.ckpt,
        "prompt": args.prompt,
        "output": result.generated_text,
        "token_ids": result.token_ids if args.show_tokens else None,
        "generation_params": result.generation_params,
        "timestamp": result.timestamp,
    }
    print(json.dumps({k: v for k, v in output_dict.items() if v is not None}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
