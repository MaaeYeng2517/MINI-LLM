"""Generate training/validation/test metric graphs for Mini-LLM models."""
import json
import os
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"
OUT_DIR = ROOT / "graphs"
MODEL_IDS = ["default", "thai-demo", "thai-v1-opto", "thai_mini_pt"]


def load_checkpoint(model_id):
    ckpt_path = CKPT_DIR / f"{model_id}.pt"
    if not ckpt_path.exists():
        # try default name
        alt = CKPT_DIR / "mini_transformer.pt"
        if alt.exists() and model_id == "thai_mini_pt":
            ckpt_path = alt
        else:
            return None
    return torch.load(str(ckpt_path), map_location="cpu", weights_only=False)


def plot_history(model_id, history, out_dir):
    if not history:
        return None
    epochs = [h["epoch"] for h in history]
    train_loss = [h.get("loss") for h in history if h.get("loss") is not None]
    val_loss = [h.get("val_loss") for h in history if h.get("val_loss") is not None]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Loss curve
    ax = axes[0]
    if train_loss:
        ax.plot(epochs[:len(train_loss)], train_loss, "o-", label="Train Loss", color="#1a1a1a")
    if val_loss:
        ax.plot(epochs[:len(val_loss)], val_loss, "s-", label="Val Loss", color="#d32f2f")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{model_id} — Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)

    # Perplexity curve
    ax = axes[1]
    if train_loss:
        train_ppl = [__import__("math").exp(min(l, 20)) for l in train_loss]
        ax.plot(epochs[:len(train_ppl)], train_ppl, "o-", label="Train Perplexity", color="#1a1a1a")
    if val_loss:
        val_ppl = [__import__("math").exp(min(l, 20)) for l in val_loss]
        ax.plot(epochs[:len(val_ppl)], val_ppl, "s-", label="Val Perplexity", color="#d32f2f")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Perplexity")
    ax.set_title(f"{model_id} — Perplexity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)

    plt.tight_layout()
    out_path = out_dir / f"{model_id}_history.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_model_comparison(all_results, out_dir):
    if not all_results:
        return None
    model_ids = [r["model_id"] for r in all_results]
    final_train = [r.get("final_train_loss") for r in all_results]
    final_val = [r.get("final_val_loss") for r in all_results]
    final_ppl = [r.get("final_perplexity") for r in all_results]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    x = range(len(model_ids))
    ax = axes[0]
    ax.bar(x, final_train, color="#1a1a1a", label="Train Loss", alpha=0.8)
    ax.bar([i + 0.4 for i in x], final_val, color="#d32f2f", label="Val Loss", alpha=0.8)
    ax.set_xticks([i + 0.2 for i in x])
    ax.set_xticklabels(model_ids, rotation=15)
    ax.set_ylabel("Loss")
    ax.set_title("Final Loss by Model")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(x, final_ppl, color="#7b1fa2", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(model_ids, rotation=15)
    ax.set_ylabel("Perplexity")
    ax.set_title("Final Perplexity by Model")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = out_dir / "model_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for model_id in MODEL_IDS:
        ckpt = load_checkpoint(model_id)
        if ckpt is None:
            print(f"⚠️  No checkpoint found for {model_id}")
            continue

        history = ckpt.get("history", [])
        config = ckpt.get("config", {})
        extra = config.get("extra_config", {}) if isinstance(config, dict) else {}

        result = {
            "model_id": model_id,
            "epochs": len(history),
            "final_train_loss": history[-1].get("loss") if history else None,
            "final_val_loss": history[-1].get("val_loss") if history else None,
            "final_perplexity": None,
            "vocab_size": config.get("vocab_size"),
            "seq_len": config.get("seq_len"),
            "params": ckpt.get("params"),
            "checkpoint": str(CKPT_DIR / f"{model_id}.pt"),
        }
        if result["final_val_loss"] is not None:
            import math
            result["final_perplexity"] = math.exp(min(result["final_val_loss"], 20))
        all_results.append(result)

        # Per-model graph
        png = plot_history(model_id, history, OUT_DIR)
        if png:
            print(f"📊 Graph saved: {png}")

    # Comparison graph
    comp = plot_model_comparison(all_results, OUT_DIR)
    if comp:
        print(f"📊 Comparison graph saved: {comp}")

    # Save summary JSON
    summary = {
        "models": all_results,
        "generated_at": __import__("datetime").datetime.now().isoformat(),
    }
    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"📋 Summary saved: {summary_path}")
    return summary


if __name__ == "__main__":
    main()
