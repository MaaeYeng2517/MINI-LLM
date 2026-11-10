import re


PAD = "<PAD>"
UNK = "<UNK>"
BOS = "<BOS>"
EOS = "<EOS>"
QUESTION = "<|question|>"
ANSWER = "<|answer|>"
SEP = "<|sep|>"
SPECIAL = [PAD, UNK, BOS, EOS, QUESTION, ANSWER, SEP]


def is_thai(ch):
    cp = ord(ch)
    return 0x0E00 <= cp <= 0x0E7F


def is_latin(ch):
    cp = ord(ch)
    return 0x0041 <= cp <= 0x005A or 0x0061 <= cp <= 0x007A


def script_tag(ch):
    if is_thai(ch):
        return "TH"
    if is_latin(ch):
        return "LA"
    return "OT"


def script_aware_split(text):
    parts = []
    cur = ""
    cur_script = None
    for ch in text:
        s = script_tag(ch)
        if cur_script is None:
            cur_script = s
            cur += ch
        elif s == cur_script or s == "OT" or cur_script == "OT":
            cur += ch
            if s != "OT":
                cur_script = s
        else:
            parts.append(cur)
            cur = ch
            cur_script = s
    if cur:
        parts.append(cur)
    return parts


class CharTokenizer:
    def __init__(self, script_aware=True, special_tokens=None):
        self.chars = []
        self.stoi = {}
        self.itos = {}
        self.script_aware = script_aware
        self.special = list(special_tokens) if special_tokens else list(SPECIAL)

    def fit(self, text):
        chars = set()
        for ch in text:
            chars.add(ch)
        self.chars = sorted(chars)
        vocab = self.special + self.chars
        self.stoi = {c: i for i, c in enumerate(vocab)}
        self.itos = {i: c for c, i in self.stoi.items()}
        return self

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, text):
        unk_id = self.stoi.get(UNK, 0)
        if self.script_aware:
            parts = script_aware_split(text)
            ids = []
            for p in parts:
                for ch in p:
                    ids.append(self.stoi.get(ch, unk_id))
            return ids
        return [self.stoi.get(c, unk_id) for c in text if c in self.stoi or unk_id is not None]

    def encode_with_special(self, text):
        return [self.stoi[BOS]] + self.encode(text) + [self.stoi[EOS]]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if i in self.itos and self.itos[i] not in self.special)

    def state(self):
        return {"chars": self.chars, "script_aware": self.script_aware, "special": self.special}

    def load_state(self, state):
        chars = state.get("chars") if isinstance(state, dict) else state
        self.chars = list(chars or [])
        self.script_aware = state.get("script_aware", True) if isinstance(state, dict) else True
        self.special = state.get("special", []) if isinstance(state, dict) else []
        vocab = self.special + self.chars
        self.stoi = {c: i for i, c in enumerate(vocab)}
        self.itos = {i: c for c, i in self.stoi.items()}
        return self


def format_qa(pairs):
    return " ".join(f"{QUESTION} {p['question']} {SEP} {ANSWER} {p['answer']}" for p in pairs)


def parse_qa(text):
    pairs = []
    pattern = re.compile(
        rf"{re.escape(QUESTION)}\s+(.*?)\s*{re.escape(SEP)}\s*{re.escape(ANSWER)}\s+",
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        question = m.group(1).strip()
        start = m.end()
        next_q = text.find(QUESTION, start)
        answer = text[start:next_q].strip() if next_q != -1 else text[start:].strip()
        pairs.append({"question": question, "answer": answer})
    return pairs
