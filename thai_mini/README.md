# Thai Mini Intelligence (Python / PyTorch)

## Milestone 1: Environment — พร้อมใช้งาน
- Python 3.x, PyTorch, MPS (Apple Silicon) / CUDA / CPU fallback

```bash
cd /Users/maaeyeng/mini-llm/thai_mini
python3 -m pip install -r requirements.txt   # ถ้ายังไม่มี torch
python3 env.py
```

## Milestones 2–7 (โครงสร้างเตรียมไว้)
- `tokenizer/` — Character Tokenizer → BPE
- `data/` — Dataset loader, train.txt, batches
- `model/` — Mini Transformer (Embedding, Attention, FFN, LayerNorm)
- `training/` — Forward, Loss, Backprop, Optimizer
- `eval/` — Validation/Test evaluation
- `checkpoints/` — saved checkpoints
