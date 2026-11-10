"""Thai-aware text normalisation for the tokenizer.

Covers:
- Unicode NFC normalisation (compose decomposed sequences).
- Thai-specific: tone marks, leading vowels, sara aa, mai ek/taiko,
  nikkhahit, yamakanna, and zero-width spaces/joiners are kept intact
  so the model sees canonical Thai grapheme clusters.
- Whitespace collapsing.
"""

import re
import unicodedata

# Zero-width / invisible joiners that should be preserved (not stripped).
_PRESERVE_ZW = {"\u200B", "\u200C", "\u200D", "\uFEFF"}

# Characters we definitely want to drop before tokenisation.
_DROP_RE = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F\u00AD\u034F\u180E\u200B-\u200F\u2028-\u202F\u205F\u3000\uFE00-\uFE0F\uFEFF\uFFF0-\uFFFF]")


def normalize_text(text, form="NFC"):
    """Apply Unicode normalisation and tidy whitespace."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize(form, text)
    # Preserve zero-width joiners that carry meaning in Thai.
    protected = {}
    for i, ch in enumerate(text):
        if ch in _PRESERVE_ZW:
            protected[f"\x00P{i}\x00"] = ch
    if protected:
        for ph, ch in protected.items():
            text = text.replace(ph, ch)
    text = _DROP_RE.sub("", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_encode(text):
    """Lighter normalisation used during encoding (keeps newlines)."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _DROP_RE.sub("", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return text.strip()