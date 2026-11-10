"""Character-level decoder for the Thai-aware tokenizer.

Reverses :class:`TuringEncoder`, with helpers for generation-style
streaming decode and Q&A structure parsing.
"""

import re

from .turing_encoder import (
    is_thai,
    is_latin,
    Vocabulary as _Vocab,
)
from .vocabulary import SPECIAL, PAD, UNK, BOS, EOS, QUESTION, ANSWER, SEP


class TuringDecoder:
    """Character-level decoder backed by a :class:`Vocabulary`."""

    def __init__(self, script_aware=True, special_tokens=None, normalize=True):
        self.script_aware = script_aware
        self.normalize = normalize
        self.vocab = _Vocab(special_tokens=special_tokens)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, text):
        """Build the vocabulary from ``text``.

        Must mirror :meth:`TuringEncoder.fit` exactly so ids map to the
        same characters. Order: Thai first, then Latin, then others,
        each group sorted.
        """
        if self.normalize:
            from .normalizer import normalize_for_encode
            text = normalize_for_encode(text)
        chars = set(text)
        thai = sorted(c for c in chars if is_thai(c))
        latin = sorted(c for c in chars if is_latin(c))
        other = sorted(c for c in chars if not is_thai(c) and not is_latin(c))
        self.vocab.build(thai + latin + other)
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def vocab_size(self):
        return self.vocab.vocab_size

    @property
    def special(self):
        return self.vocab.special

    @property
    def itos(self):
        return self.vocab.itos

    @property
    def stoi(self):
        return self.vocab.stoi

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------
    def decode(self, ids, filter_special=True):
        """Decode a list of token ids back into text."""
        chars = []
        for i in ids:
            token = self.vocab.id_to_token(i)
            if token is None:
                continue
            if filter_special and token in self.vocab.special:
                continue
            chars.append(token)
        return "".join(chars)

    def decode_with_special(self, ids):
        """Decode keeping special tokens in the output."""
        return self.decode(ids, filter_special=False)

    def decode_stream(self, ids, filter_special=True):
        """Decode incrementally (same result as :meth:`decode`)."""
        return self.decode(ids, filter_special=filter_special)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def state(self):
        return self.vocab.state()

    def load_state(self, state):
        self.vocab.load_state(state)
        return self

    # ------------------------------------------------------------------
    # Q&A helpers
    # ------------------------------------------------------------------
    def format_qa(self, pairs):
        """Format a list of ``{"question":..., "answer":...}`` dicts into a string."""
        return " ".join(
            f"{QUESTION} {p['question']} {SEP} {ANSWER} {p['answer']}"
            for p in pairs
        )

    def parse_qa(self, text):
        """Parse a Q&A formatted string back into a list of dicts."""
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

    def decode_qa(self, ids):
        """Decode ids and parse the Q&A structure from the result."""
        text = self.decode(ids, filter_special=False)
        return self.parse_qa(text)