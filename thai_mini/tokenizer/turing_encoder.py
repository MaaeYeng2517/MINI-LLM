"""Character-level encoder for the Thai-aware tokenizer.

Encodes text into token ids using a :class:`Vocabulary`. Thai, Latin and
other scripts are handled at character level, with optional script-aware
segmentation so the model can see script boundaries.
"""

import re

from .vocabulary import Vocabulary, SPECIAL
from .normalizer import normalize_for_encode

# Special-token regex used to keep special tokens atomic during encoding.
_SPECIAL_SEP = "|".join(re.escape(s) for s in SPECIAL)
_SPECIAL_RE = re.compile(f"({_SPECIAL_SEP})") if _SPECIAL_SEP else None


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
    """Split text into runs of the same script type."""
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


class TuringEncoder:
    """Character-level encoder backed by a :class:`Vocabulary`."""

    def __init__(self, vocab_size=2000, script_aware=True, special_tokens=None, normalize=True):
        self._vocab_size_limit = vocab_size
        self.script_aware = script_aware
        self.normalize = normalize
        self.vocab = Vocabulary(special_tokens=special_tokens)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, text):
        """Build the vocabulary from ``text``."""
        if self.normalize:
            text = normalize_for_encode(text)
        chars = set(text)
        # Order: Thai first, then Latin, then others (deterministic).
        thai = sorted(c for c in chars if is_thai(c))
        latin = sorted(c for c in chars if is_latin(c))
        other = sorted(c for c in chars if not is_thai(c) and not is_latin(c))
        self.vocab.build(thai + latin + other, vocab_size=self._vocab_size_limit)
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def vocab_size(self):
        return self.vocab.vocab_size

    @vocab_size.setter
    def vocab_size(self, value):
        self._vocab_size_limit = value

    @property
    def unk_id(self):
        return self.vocab.unk_id

    @property
    def special(self):
        return self.vocab.special

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode(self, text):
        """Encode ``text`` into a list of token ids."""
        if text is None:
            return []
        if self.normalize:
            text = normalize_for_encode(text)
        if not text:
            return []

        unk_id = self.vocab.unk_id
        ids = []

        if _SPECIAL_RE:
            parts = _SPECIAL_RE.split(text)
            for part in parts:
                if not part:
                    continue
                if part in self.vocab.stoi:
                    ids.append(self.vocab.stoi[part])
                else:
                    ids.extend(self._encode_segment(part, unk_id))
        else:
            ids.extend(self._encode_segment(text, unk_id))
        return ids

    def _encode_segment(self, segment, unk_id):
        if self.script_aware:
            sub_parts = script_aware_split(segment)
            out = []
            for sp in sub_parts:
                if not sp:
                    continue
                for ch in sp:
                    out.append(self.vocab.stoi.get(ch, unk_id))
            return out
        return [self.vocab.stoi.get(ch, unk_id) for ch in segment]

    def encode_with_special(self, text):
        ids = self.encode(text)
        bos = self.vocab.token_to_id("<BOS>", default=None)
        eos = self.vocab.token_to_id("<EOS>", default=None)
        if bos is not None:
            ids = [bos] + ids
        if eos is not None:
            ids = ids + [eos]
        return ids

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------
    def decode(self, ids, filter_special=True):
        chars = []
        for i in ids:
            token = self.vocab.id_to_token(i)
            if token is None:
                continue
            if filter_special and token in self.vocab.special:
                continue
            chars.append(token)
        return "".join(chars)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def state(self):
        return self.vocab.state()

    def load_state(self, state):
        self.vocab.load_state(state)
        return self