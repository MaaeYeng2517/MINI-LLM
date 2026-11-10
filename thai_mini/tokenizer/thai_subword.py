"""Thai-aware Character-level Tokenizer for Mini-LLM.

Compatibility shim: re-exports :class:`TuringEncoder` as
``ThaiCharTokenizer`` so existing code that imports from this module
continues to work unchanged.
"""

import re

from .turing_encoder import (
    TuringEncoder,
    is_thai,
    is_latin,
    script_tag,
    script_aware_split,
    normalize_for_encode,
)
from .turing_decoder import TuringDecoder
from .vocabulary import (
    Vocabulary,
    SPECIAL,
    PAD,
    UNK,
    BOS,
    EOS,
    QUESTION,
    ANSWER,
    SEP,
)


class ThaiCharTokenizer(TuringEncoder):
    """Character-level tokenizer with script-aware segmentation.

    Thai script is tokenized at character level (appropriate since Thai lacks
    explicit word boundaries). Latin and other scripts are also character-level.
    Special tokens for Q&A format are included.
    """

    def __init__(self, vocab_size=2000, script_aware=True, special_tokens=None, normalize=True):
        super().__init__(
            vocab_size=vocab_size,
            script_aware=script_aware,
            special_tokens=special_tokens,
            normalize=normalize,
        )

    # Convenience aliases matching the old API surface.
    def encode_with_special(self, text):
        return super().encode_with_special(text)

    def decode(self, ids, filter_special=True):
        return super().decode(ids, filter_special=filter_special)

    def state(self):
        return super().state()

    def load_state(self, state):
        return super().load_state(state)


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


__all__ = [
    "ThaiCharTokenizer",
    "TuringEncoder",
    "TuringDecoder",
    "Vocabulary",
    "SPECIAL",
    "PAD",
    "UNK",
    "BOS",
    "EOS",
    "QUESTION",
    "ANSWER",
    "SEP",
    "is_thai",
    "is_latin",
    "script_tag",
    "script_aware_split",
    "normalize_for_encode",
    "format_qa",
    "parse_qa",
]