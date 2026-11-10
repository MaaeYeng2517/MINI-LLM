"""Data cleaning utilities for Thai Q&A datasets (JSON + TXT)."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "public" / "data"
MODELS_DIR = DATA_DIR / "models"

# Thai character ranges + common Latin/punctuation
_THAI_RE = re.compile(r"[\u0E00-\u0E7F]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_DIGIT_RE = re.compile(r"[0-9]")
_PUNCT_RE = re.compile(r"[.,!?\"'()\-\:;]")
_WHITESPACE_RE = re.compile(r"\s+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_REPEAT_RE = re.compile(r"(.)\1{4,}")
_NON_THAI_EN_NUM_PUNCT_RE = re.compile(
    r"[^\u0E00-\u0E7F\u0041-\u005A\u0061-\u007A\u0030-\u0039\s.,!?\"'()\-\:;\u00A0\u200B\u200C\u200D\uFE0F\u200D\u200E\u202A-\u202E\u2060-\u206F\u2028\u2029\u00AD\u200B\u200C\u200D\uFE0F]"
)


def normalize_text(text):
    """Normalize whitespace, strip BOM, remove control chars, collapse repeats."""
    if not text:
        return ""
    text = text.replace("\ufeff", "").replace("\u200B", "").replace("\u200C", "")
    text = _CONTROL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = text.strip()
    text = _REPEAT_RE.sub(lambda m: m.group(1) * 3, text)
    return text


def is_valid_qa(q, a, min_q=2, min_a=2, max_len=2000):
    """Check if a Q&A pair is valid after normalization."""
    if not q or not a:
        return False
    if len(q) < min_q or len(a) < min_a:
        return False
    if len(q) > max_len or len(a) > max_len:
        return False
    if q == a and len(q) < 5:
        return False
    return True


def clean_json_dataset(data, dedup=True, dedup_key=None):
    """Clean a list of {question, answer} dicts: normalize, validate, dedup."""
    seen = set()
    cleaned = []
    stats = {"total": len(data), "empty_removed": 0, "invalid_removed": 0, "dup_removed": 0}

    for item in data:
        if not isinstance(item, dict):
            stats["invalid_removed"] += 1
            continue
        q = normalize_text(item.get("question", ""))
        a = normalize_text(item.get("answer", ""))
        if not q and not a:
            stats["empty_removed"] += 1
            continue
        if not is_valid_qa(q, a):
            stats["invalid_removed"] += 1
            continue
        key = dedup_key(q, a) if dedup_key else (q, a)
        if dedup and key in seen:
            stats["dup_removed"] += 1
            continue
        seen.add(key)
        cleaned.append({"question": q, "answer": a})

    stats["kept"] = len(cleaned)
    return cleaned, stats


def clean_txt_lines(text):
    """Clean raw TXT lines: remove empty, normalize, collapse blank lines."""
    lines = text.split("\n")
    cleaned = []
    prev_empty = False
    for line in lines:
        norm = normalize_text(line)
        if not norm:
            if not prev_empty:
                cleaned.append("")
                prev_empty = True
            continue
        prev_empty = False
        cleaned.append(norm)
    # trim trailing empties
    while cleaned and not cleaned[-1]:
        cleaned.pop()
    return "\n".join(cleaned)


def clean_json_file(path, out_path=None, dedup=True):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cleaned, stats = clean_json_dataset(data, dedup=dedup)
    out = out_path or path
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    return stats


def clean_txt_file(path, out_path=None):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    cleaned = clean_txt_lines(text)
    out = out_path or path
    with open(out, "w", encoding="utf-8") as f:
        f.write(cleaned)
    return {"chars": len(text), "cleaned_chars": len(cleaned)}


def clean_all(model_ids=None, write=True):
    """Clean all model datasets (train/val/test JSON + TXT)."""
    if model_ids is None:
        model_ids = [d.name for d in MODELS_DIR.iterdir() if d.is_dir()]

    all_stats = {}
    for mid in model_ids:
        mdir = MODELS_DIR / mid
        if not mdir.is_dir():
            continue
        model_stats = {}
        for kind in ("train", "val", "test"):
            jpath = mdir / f"{kind}.json"
            tpath = mdir / f"{kind}.txt"
            if jpath.exists():
                stats = clean_json_file(jpath, jpath if write else None)
                model_stats[f"{kind}_json"] = stats
            if tpath.exists():
                stats = clean_txt_file(tpath, tpath if write else None)
                model_stats[f"{kind}_txt"] = stats
        all_stats[mid] = model_stats
        print(f"--- {mid} ---")
        for k, v in model_stats.items():
            print(f"  {k}: {v}")
    return all_stats


if __name__ == "__main__":
    import sys
    write = "--write" in sys.argv
    clean_all(write=write)