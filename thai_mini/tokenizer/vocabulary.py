"""Vocabulary management for the Thai-aware tokenizer.

Holds the token <-> id mapping, special tokens, and (de)serialisation
logic so the encoder/decoder can stay focused on conversion.
"""

import json
from pathlib import Path

PAD = "<PAD>"
UNK = "<UNK>"
BOS = "<BOS>"
EOS = "<EOS>"
QUESTION = "<|question|>"
ANSWER = "<|answer|>"
SEP = "<|sep|>"
SPECIAL = [PAD, UNK, BOS, EOS, QUESTION, ANSWER, SEP]

# Default id assigned to UNK when present; otherwise 0 (PAD).
_DEFAULT_UNK_ID = 1


class Vocabulary:
    """Bidirectional mapping between characters (and special tokens) and ids."""

    def __init__(self, special_tokens=None):
        self.special = list(special_tokens) if special_tokens else list(SPECIAL)
        self.chars = []
        self.stoi = {}
        self.itos = {}
        self._unk_id = self.special.index(UNK) if UNK in self.special else _DEFAULT_UNK_ID

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------
    def build(self, chars, vocab_size=None):
        """Build the mapping from an iterable of characters.

        Special tokens always keep their leading positions; remaining
        slots are filled with ``chars`` (deduplicated, sorted).
        """
        seen = []
        seen_set = set()
        for c in chars:
            if c in seen_set or c in self.special:
                continue
            seen_set.add(c)
            seen.append(c)

        vocab = list(self.special) + seen
        if vocab_size is not None and len(vocab) > vocab_size:
            vocab = vocab[:vocab_size]

        self.chars = [c for c in vocab if c not in self.special]
        self.stoi = {c: i for i, c in enumerate(vocab)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self._unk_id = self.stoi.get(UNK, _DEFAULT_UNK_ID)
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def vocab_size(self):
        return len(self.stoi)

    @property
    def unk_id(self):
        return self._unk_id

    def __len__(self):
        return len(self.stoi)

    def __contains__(self, token):
        return token in self.stoi

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    def token_to_id(self, token, default=None):
        if default is None:
            default = self._unk_id
        return self.stoi.get(token, default)

    def id_to_token(self, id_, default=UNK):
        return self.itos.get(id_, default)

    def is_special(self, token):
        return token in self.special

    def is_special_id(self, id_):
        return id_ in self.itos and self.itos[id_] in self.special

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def state(self):
        return {
            "special": list(self.special),
            "chars": list(self.chars),
            "vocab_size": self.vocab_size,
        }

    def load_state(self, state):
        if isinstance(state, (list, tuple)):
            # Plain list of tokens: rebuild vocab directly.
            vocab = list(state)
            self.special = [t for t in vocab if t in SPECIAL] or list(SPECIAL)
            self.chars = [c for c in vocab if c not in self.special]
        else:
            self.special = list(state.get("special", list(SPECIAL)))
            chars = state.get("chars", state.get("vocab", []))
            self.chars = list(chars)
            if "vocab_size" in state:
                try:
                    limit = int(state["vocab_size"])
                    if limit and len(self.special) + len(self.chars) > limit:
                        self.chars = self.chars[: max(0, limit - len(self.special))]
                except (TypeError, ValueError):
                    pass

        vocab = list(self.special) + [c for c in self.chars if c not in self.special]
        self.stoi = {c: i for i, c in enumerate(vocab)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self._unk_id = self.stoi.get(UNK, _DEFAULT_UNK_ID)
        return self

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state(), f, ensure_ascii=False)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return self.load_state(json.load(f))