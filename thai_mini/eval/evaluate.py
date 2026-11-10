import math
import torch
from torch import nn


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    loss_fn = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        if logits.ndim == 3:
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        else:
            if y.ndim == 2:
                y = y[:, -1]
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total += loss.item()
        n += 1
    loss = total / max(n, 1)
    return {"loss": loss, "perplexity": math.exp(min(loss, 20)), "batches": n}
