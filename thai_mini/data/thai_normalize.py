"""Thai text standardization for Q&A datasets.

Standardizes Thai text to support proper Thai tokenization:
- Normalize whitespace (collapse multiple spaces, trim)
- Remove duplicate particles (ครับ ครับ -> ครับ)
- Remove space before particles (ครับ -> ครับ)
- Remove zero-width/BOM/control characters
- Collapse repeated characters (>3 -> 3)
- Normalize line endings
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "public" / "data" / "models"

# Thai Unicode ranges
THAI_CONSONANTS = range(0x0E01, 0x0E2F)   # ก-ฮ
THAI_VOWELS = range(0x0E30, 0x0E3F)       # ็-ๆ
THAI_TONE_MARKS = range(0x0E48, 0x0E4F)    # ่-๊
THAI_DIGITS = range(0x0E50, 0x0E5F)        # ๐-๙

PARTICLES = r"(ครับ|ค่ะ|ไหม|จัง|น่ะ|สิ|จ้ะ|ค่ะ|ครับๆ|ไหมๆ|จังๆ)"
REPEAT_THRESHOLD = 3


def is_thai_char(c):
    return (
        0x0E01 <= ord(c) <= 0x0E3A or
        0x0E40 <= ord(c) <= 0x0E7F
    )


def normalize_text(text):
    """Standardize Thai text for tokenization."""
    if not text:
        return ""

    # Remove BOM and zero-width characters
    text = text.replace("\ufeff", "")
    text = re.sub(r"[\u200B\u200C\u200D\uFE0F\u200E\u200F\u202A-\u202E\u2060-\u206F\u2028\u2029\u00AD\u2009\u200A\u200B\u202F\u205F\u3000]", "", text)

    # Remove control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse multiple spaces into one
    text = re.sub(r"[ \t]+", " ", text)

    # Remove space before particles (ครับ ค่ะ ไหม จัง น่ะ สิ)
    text = re.sub(rf"\s+({PARTICLES})(?=[\s\u0E01-\u0E7F]|$)", r"\1", text)

    # Remove duplicate consecutive particles (ครับ ครับ -> ครับ)
    prev = None
    while prev != text:
        prev = text
        text = re.sub(rf"({PARTICLES})\s+\1", r"\1", text)

    # Collapse repeated characters (more than threshold -> threshold)
    text = re.sub(r"(.)\1{" + str(REPEAT_THRESHOLD) + r",}", lambda m: m.group(1) * REPEAT_THRESHOLD, text)

    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Trim leading/trailing whitespace
    text = text.strip()

    return text


def clean_qa_dataset(data):
    """Clean and standardize a list of {question, answer} dicts."""
    seen = set()
    cleaned = []
    for item in data:
        if not isinstance(item, dict):
            continue
        q = normalize_text(item.get("question", ""))
        a = normalize_text(item.get("answer", ""))
        if not q and not a:
            continue
        if len(q) < 2 or len(a) < 2:
            continue
        key = (q, a)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"question": q, "answer": a})
    return cleaned


def clean_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cleaned = clean_qa_dataset(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    return {"before": len(data), "after": len(cleaned)}


def clean_txt_file(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    cleaned = normalize_text(text)
    with open(path, "w", encoding="utf-8") as f:
        f.write(cleaned)
    return {"before": len(text), "after": len(cleaned)}


def clean_all():
    """Standardize Thai text in all model datasets."""
    results = {}
    for mdir in sorted(MODELS_DIR.iterdir()):
        if not mdir.is_dir():
            continue
        mid = mdir.name
        model_results = {}
        for kind in ("train", "val", "test"):
            jpath = mdir / f"{kind}.json"
            tpath = mdir / f"{kind}.txt"
            if jpath.exists():
                model_results[f"{kind}_json"] = clean_json_file(jpath)
            if tpath.exists():
                model_results[f"{kind}_txt"] = clean_txt_file(tpath)
        results[mid] = model_results
        print(f"--- {mid} ---")
        for k, v in model_results.items():
            print(f"  {k}: {v['before']} -> {v['after']} chars/items")
    return results


if __name__ == "__main__":
    clean_all()