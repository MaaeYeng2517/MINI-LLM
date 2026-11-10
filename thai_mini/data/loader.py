import json
import torch
from torch.utils.data import Dataset, DataLoader


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_dataset(path):
    """Load dataset from .txt (raw text) or .json (Q&A list or string list), UTF-8."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        data = json.loads(raw)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            # Q&A format: [{"question":..., "answer":...}, ...]
            parts = []
            for item in data:
                q = item.get("question", "")
                a = item.get("answer", "")
                if q or a:
                    parts.append(f"คำถาม: {q}\nคำตอบ: {a}")
            return "\n\n".join(parts)
        if isinstance(data, list):
            return "\n".join(x for x in data if isinstance(x, str) and x.strip())
        if isinstance(data, str):
            return data
    except json.JSONDecodeError:
        pass
    return raw


class NextTokenDataset(Dataset):
    """Full-sequence next-token targets for causal LM: x=t[0..n-1], y=t[1..n]."""
    def __init__(self, ids, seq_len):
        self.ids = ids
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.ids) - self.seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.ids[idx : idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def build_loaders(text, tokenizer, seq_len=32, batch_size=32, val_split=0.15, shuffle=True):
    ids = tokenizer.encode(text)
    if len(ids) <= seq_len + 2:
        raise ValueError("Training text too short")
    split = max(1, int(len(ids) * (1 - val_split)))
    train_ds = NextTokenDataset(ids[:split], seq_len)
    val_ds = NextTokenDataset(ids[split - seq_len :] if split > seq_len else ids, seq_len) if split < len(ids) - seq_len - 1 else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_ds, batch_size=batch_size) if val_ds and len(val_ds) else None
    return train_loader, val_loader, {
        "tokens": len(ids),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds) if val_ds else 0,
    }


def build_loader(text, tokenizer, seq_len=32, batch_size=32, shuffle=False):
    ids = tokenizer.encode(text)
    if len(ids) <= seq_len + 2:
        raise ValueError("Text too short")
    ds = NextTokenDataset(ids, seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return loader, {"tokens": len(ids), "samples": len(ds)}
