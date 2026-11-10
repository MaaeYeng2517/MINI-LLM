import torch
import torch.nn.functional as F

STOP_MARKERS = ("<|question|>", "<|answer|>", "<|sep|>", "\nคำถาม:", "\nผู้ใช้:", "\nUser:", "\nQ:", "\nคำตอบ:", "\nผู้ช่วย:", "\nBot:")


def _logits(out):
    return out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def generate_text(model, tokenizer, prompt, device, max_new_tokens=80, temperature=0.7, top_k=24, repetition_penalty=1.15):
    model.eval()
    try:
        ids = tokenizer.encode(prompt)
    except TypeError:
        ids = tokenizer.encode(prompt)
    if not ids:
        ids = [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    seq_len = int(getattr(model, "seq_len", idx.size(1)))
    generated = list(ids)
    for _ in range(max_new_tokens):
        cur = idx[:, -seq_len:]
        logits = _logits(model(cur))
        next_logits = (logits[:, -1, :] if logits.ndim == 3 else logits).to(torch.float32)
        next_logits = next_logits / max(float(temperature), 1e-5)
        if repetition_penalty and repetition_penalty != 1.0:
            seen = set(generated[-96:])
            for t in seen:
                if 0 <= t < next_logits.size(-1):
                    v = next_logits[0, t]
                    next_logits[0, t] = v / repetition_penalty if v > 0 else v * repetition_penalty
        if top_k and top_k > 0:
            k = min(int(top_k), next_logits.size(-1))
            vals, _ = torch.topk(next_logits, k)
            cutoff = vals[:, -1].unsqueeze(-1)
            next_logits = torch.where(next_logits < cutoff, torch.full_like(next_logits, float("-inf")), next_logits)
        probs = F.softmax(next_logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        nid = int(nxt.item())
        generated.append(nid)
        idx = torch.cat([idx, nxt], dim=1)
        if len(generated) % 4 == 0:
            try:
                tail = tokenizer.decode(generated[-80:], filter_special=False)
            except TypeError:
                tail = tokenizer.decode(generated[-80:])
            for m in STOP_MARKERS:
                if m in tail:
                    try:
                        return tokenizer.decode(generated, filter_special=False)
                    except TypeError:
                        return tokenizer.decode(generated)
    try:
        return tokenizer.decode(generated, filter_special=False)
    except TypeError:
        return tokenizer.decode(generated)
