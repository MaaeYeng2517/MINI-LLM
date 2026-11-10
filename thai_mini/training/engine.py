import os
import copy
import torch
from torch import nn


def save_checkpoint(model, optimizer, epoch, history, path, tokenizer=None, config=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "history": history}
    if tokenizer is not None and hasattr(tokenizer, "state"):
        obj["tokenizer"] = tokenizer.state()
    if config:
        obj["config"] = config
    torch.save(obj, path)
    return path


def _loss(model, x, y):
    out = model(x)
    # support models returning logits, (logits, loss), or (logits, ...)
    if isinstance(out, tuple):
        logits = out[0]
        if len(out) > 1 and out[1] is not None and hasattr(out[1], "ndim") and out[1].ndim == 0:
            return out[1], logits
    else:
        logits = out
    # full-seq logits [B,T,V] or next-token [B,V]
    if logits.ndim == 3:
        return nn.CrossEntropyLoss()(logits.reshape(-1, logits.size(-1)), y.reshape(-1)), logits
    if logits.ndim == 2 and y.ndim == 2:
        y = y[:, -1]
    return nn.CrossEntropyLoss()(logits.reshape(-1, logits.size(-1)), y.reshape(-1)), logits


def run_epoch(model, loader, optimizer, device, train=True):
    model.train(train)
    total = 0.0
    count = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss, _ = _loss(model, x, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item()
            count += 1
    return total / max(count, 1)


def train(model, train_loader, val_loader, device, epochs=10, lr=3e-4, ckpt_dir="checkpoints", ckpt_name="mini_transformer.pt", tokenizer=None, extra_config=None):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = []
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    seq_len = getattr(train_loader.dataset, "seq_len", 48) if hasattr(train_loader, "dataset") else 48
    config = {"vocab_size": getattr(model, "vocab_size", None), "n_embd": getattr(model, "n_embd", None), "n_head": getattr(model, "n_head", None), "n_layer": getattr(model, "n_layer", None), "seq_len": seq_len}
    if extra_config:
        config.update(extra_config)
    bad = 0
    for epoch in range(1, epochs + 1):
        loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False) if val_loader else None
        row = {"epoch": epoch, "loss": loss, "val_loss": val_loss}
        history.append(row)
        monitor = val_loss if val_loss is not None else loss
        if monitor < best_val - 1e-5:
            best_val = monitor
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            bad = 0
        else:
            bad += 1
        print({"epoch": epoch, "loss": loss, "val_loss": val_loss, "best_epoch": best_epoch})
        if val_loader and bad >= 12 and epoch >= 15:
            print({"early_stop": True, "best_epoch": best_epoch})
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        config["best_epoch"] = best_epoch
        config["best_val_loss"] = best_val
    save_checkpoint(model, optimizer, best_epoch or epoch, history, ckpt_path, tokenizer=tokenizer, config=config)
    return history, ckpt_path
