"""Thai Unicode Engine for Mini-LLM TH.

Provides comprehensive Unicode handling for Thai text including:
- UTF-8 encoding/decoding
- Unicode NFC/NFD normalization
- Thai character validation (U+0E00–U+0E7F)
- Thai combining character handling (vowels, tone marks, combining chars)
- Grapheme cluster detection and segmentation
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set, Dict
from enum import Enum


class ThaiCharCategory(Enum):
    """Thai character categories based on Unicode block U+0E00–U+0E7F."""
    CONSONANT = "consonant"
    VOWEL = "vowel"
    TONE_MARK = "tone_mark"
    SIGN = "sign"
    DIGIT = "digit"
    PUNCTUATION = "punctuation"
    OTHER = "other"


@dataclass(frozen=True)
class ThaiCharInfo:
    """Information about a Thai character."""
    char: str
    codepoint: int
    name: str
    category: ThaiCharCategory
    is_combining: bool
    combining_class: int


THAI_BLOCK_START = 0x0E00
THAI_BLOCK_END = 0x0E7F

THAI_CONSONANTS = {
    0x0E01: "THAI CHARACTER KO KAI",
    0x0E02: "THAI CHARACTER KHO KHAI",
    0x0E03: "THAI CHARACTER KHO KHUAT",
    0x0E04: "THAI CHARACTER KHO KHWAI",
    0x0E05: "THAI CHARACTER KHO KHON",
    0x0E06: "THAI CHARACTER KHO RAKHANG",
    0x0E07: "THAI CHARACTER NGO NGU",
    0x0E08: "THAI CHARACTER CHO CHAN",
    0x0E09: "THAI CHARACTER CHO CHING",
    0x0E0A: "THAI CHARACTER CHO CHANG",
    0x0E0B: "THAI CHARACTER SO SO",
    0x0E0C: "THAI CHARACTER CHO CHOE",
    0x0E0D: "THAI CHARACTER YO YING",
    0x0E0E: "THAI CHARACTER DO CHADA",
    0x0E0F: "THAI CHARACTER TO PATAK",
    0x0E10: "THAI CHARACTER THO THAN",
    0x0E11: "THAI CHARACTER THO NANGMONTHO",
    0x0E12: "THAI CHARACTER THO PHUTHAO",
    0x0E13: "THAI CHARACTER NO NEN",
    0x0E14: "THAI CHARACTER DO DEK",
    0x0E15: "THAI CHARACTER TO TAO",
    0x0E16: "THAI CHARACTER THO THUNG",
    0x0E17: "THAI CHARACTER THO THAHAN",
    0x0E18: "THAI CHARACTER THO THONG",
    0x0E19: "THAI CHARACTER NO NU",
    0x0E1A: "THAI CHARACTER BO BAIMAI",
    0x0E1B: "THAI CHARACTER PO PLA",
    0x0E1C: "THAI CHARACTER PHO PHUNG",
    0x0E1D: "THAI CHARACTER FO FA",
    0x0E1E: "THAI CHARACTER PHO PHAN",
    0x0E1F: "THAI CHARACTER FO FAN",
    0x0E20: "THAI CHARACTER PHO SAMPHAO",
    0x0E21: "THAI CHARACTER MO MA",
    0x0E22: "THAI CHARACTER YO YAK",
    0x0E23: "THAI CHARACTER RO RUA",
    0x0E24: "THAI CHARACTER RU",
    0x0E25: "THAI CHARACTER LO LING",
    0x0E26: "THAI CHARACTER LU",
    0x0E27: "THAI CHARACTER WO WAEN",
    0x0E28: "THAI CHARACTER SO SALA",
    0x0E29: "THAI CHARACTER SO RUSI",
    0x0E2A: "THAI CHARACTER SO SUA",
    0x0E2B: "THAI CHARACTER HO HIP",
    0x0E2C: "THAI CHARACTER LO CHULA",
    0x0E2D: "THAI CHARACTER O ANG",
    0x0E2E: "THAI CHARACTER HO NOKHUK",
}

THAI_VOWELS = {
    0x0E30: "THAI CHARACTER SARA A",
    0x0E31: "THAI CHARACTER SARA AA",
    0x0E32: "THAI CHARACTER SARA AM",
    0x0E33: "THAI CHARACTER SARA I",
    0x0E34: "THAI CHARACTER SARA II",
    0x0E35: "THAI CHARACTER SARA UE",
    0x0E36: "THAI CHARACTER SARA UEE",
    0x0E37: "THAI CHARACTER SARA U",
    0x0E38: "THAI CHARACTER SARA UU",
    0x0E39: "THAI CHARACTER SARA UUE",
    0x0E3A: "THAI CHARACTER PHINTHU",
    0x0E40: "THAI CHARACTER SARA E",
    0x0E41: "THAI CHARACTER SARA AE",
    0x0E42: "THAI CHARACTER SARA O",
    0x0E43: "THAI CHARACTER SARA AI MAIMUAN",
    0x0E44: "THAI CHARACTER SARA AI MAIMALAI",
}

THAI_TONE_MARKS = {
    0x0E48: "THAI CHARACTER MAI EK",
    0x0E49: "THAI CHARACTER MAI THO",
    0x0E4A: "THAI CHARACTER MAI TRI",
    0x0E4B: "THAI CHARACTER MAI CHATTAWA",
}

THAI_SIGNS = {
    0x0E4C: "THAI CHARACTER THANTHAKHAT",
    0x0E4D: "THAI CHARACTER NIKKHAHIT",
    0x0E4E: "THAI CHARACTER YAMAKKAN",
}

THAI_DIGITS = {
    0x0E50: "THAI DIGIT ZERO",
    0x0E51: "THAI DIGIT ONE",
    0x0E52: "THAI DIGIT TWO",
    0x0E53: "THAI DIGIT THREE",
    0x0E54: "THAI DIGIT FOUR",
    0x0E55: "THAI DIGIT FIVE",
    0x0E56: "THAI DIGIT SIX",
    0x0E57: "THAI DIGIT SEVEN",
    0x0E58: "THAI DIGIT EIGHT",
    0x0E59: "THAI DIGIT NINE",
}

THAI_PUNCTUATION = {
    0x0E3F: "THAI CURRENCY SYMBOL BAHT",
    0x0E4F: "THAI CHARACTER FONGMAN",
    0x0E5A: "THAI CHARACTER ANGKHANKHU",
    0x0E5B: "THAI CHARACTER KHOMUT",
}

ALL_THAI_CHARS = {
    **THAI_CONSONANTS,
    **THAI_VOWELS,
    **THAI_TONE_MARKS,
    **THAI_SIGNS,
    **THAI_DIGITS,
    **THAI_PUNCTUATION,
}

THAI_COMBINING_CLASSES = {
    0x0E31: 103,  # SARA AA
    0x0E34: 103,  # SARA II
    0x0E35: 103,  # SARA UE
    0x0E36: 103,  # SARA UEE
    0x0E37: 103,  # SARA U
    0x0E38: 103,  # SARA UU
    0x0E39: 103,  # SARA UUE
    0x0E3A: 9,    # PHINTHU
    0x0E47: 107,  # MAITAIKHU (not in standard block but used)
    0x0E48: 107,  # MAI EK
    0x0E49: 107,  # MAI THO
    0x0E4A: 107,  # MAI TRI
    0x0E4B: 107,  # MAI CHATTAWA
    0x0E4C: 107,  # THANTHAKHAT
    0x0E4D: 107,  # NIKKHAHIT
    0x0E4E: 107,  # YAMAKKAN
}

THAI_LEADING_VOWELS = {0x0E40, 0x0E41, 0x0E42, 0x0E43, 0x0E44}
THAI_ABOVE_VOWELS = {0x0E31, 0x0E34, 0x0E35, 0x0E36, 0x0E37, 0x0E38, 0x0E39}
THAI_BELOW_VOWELS = {0x0E3A}
THAI_TONE_MARK_CODEPOINTS = {0x0E48, 0x0E49, 0x0E4A, 0x0E4B}
THAI_SIGN_CODEPOINTS = {0x0E4C, 0x0E4D, 0x0E4E}
THAI_VOWEL_CODEPOINTS = set(THAI_VOWELS.keys())
THAI_CONSONANT_CODEPOINTS = set(THAI_CONSONANTS.keys())
THAI_DIGIT_CODEPOINTS = set(THAI_DIGITS.keys())


def is_thai_char(ch: str) -> bool:
    """Check if a character is in the Thai Unicode block (U+0E00–U+0E7F)."""
    if not ch or len(ch) != 1:
        return False
    cp = ord(ch)
    return THAI_BLOCK_START <= cp <= THAI_BLOCK_END


def is_thai_consonant(ch: str) -> bool:
    """Check if a character is a Thai consonant."""
    if not ch or len(ch) != 1:
        return False
    return ord(ch) in THAI_CONSONANT_CODEPOINTS


def is_thai_vowel(ch: str) -> bool:
    """Check if a character is a Thai vowel."""
    if not ch or len(ch) != 1:
        return False
    return ord(ch) in THAI_VOWEL_CODEPOINTS


def is_thai_tone_mark(ch: str) -> bool:
    """Check if a character is a Thai tone mark."""
    if not ch or len(ch) != 1:
        return False
    return ord(ch) in THAI_TONE_MARK_CODEPOINTS


def is_thai_sign(ch: str) -> bool:
    """Check if a character is a Thai sign (thanthakhat, nikkhahit, yamakkan)."""
    if not ch or len(ch) != 1:
        return False
    return ord(ch) in THAI_SIGN_CODEPOINTS


def is_thai_digit(ch: str) -> bool:
    """Check if a character is a Thai digit."""
    if not ch or len(ch) != 1:
        return False
    return ord(ch) in THAI_DIGIT_CODEPOINTS


def is_thai_combining(ch: str) -> bool:
    """Check if a character is a Thai combining character (vowel, tone mark, sign)."""
    return is_thai_vowel(ch) or is_thai_tone_mark(ch) or is_thai_sign(ch)


def get_thai_char_info(ch: str) -> Optional[ThaiCharInfo]:
    """Get detailed information about a Thai character."""
    if not ch or len(ch) != 1:
        return None
    cp = ord(ch)
    if not (THAI_BLOCK_START <= cp <= THAI_BLOCK_END):
        return None

    name = ALL_THAI_CHARS.get(cp, "UNKNOWN")
    is_combining = is_thai_combining(ch)
    combining_class = THAI_COMBINING_CLASSES.get(cp, 0)

    if cp in THAI_CONSONANT_CODEPOINTS:
        category = ThaiCharCategory.CONSONANT
    elif cp in THAI_VOWEL_CODEPOINTS:
        category = ThaiCharCategory.VOWEL
    elif cp in THAI_TONE_MARK_CODEPOINTS:
        category = ThaiCharCategory.TONE_MARK
    elif cp in THAI_SIGN_CODEPOINTS:
        category = ThaiCharCategory.SIGN
    elif cp in THAI_DIGIT_CODEPOINTS:
        category = ThaiCharCategory.DIGIT
    elif cp in THAI_PUNCTUATION:
        category = ThaiCharCategory.PUNCTUATION
    else:
        category = ThaiCharCategory.OTHER

    return ThaiCharInfo(
        char=ch,
        codepoint=cp,
        name=name,
        category=category,
        is_combining=is_combining,
        combining_class=combining_class,
    )


def get_thai_category(ch: str) -> ThaiCharCategory:
    """Get the category of a Thai character."""
    info = get_thai_char_info(ch)
    return info.category if info else ThaiCharCategory.OTHER


def normalize_thai_nfc(text: str) -> str:
    """Normalize Thai text to NFC (composed form)."""
    if not text:
        return ""
    return unicodedata.normalize("NFC", text)


def normalize_thai_nfd(text: str) -> str:
    """Normalize Thai text to NFD (decomposed form)."""
    if not text:
        return ""
    return unicodedata.normalize("NFD", text)


def normalize_thai_nfkc(text: str) -> str:
    """Normalize Thai text to NFKC (compatibility composed)."""
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text)


def normalize_thai_nfkd(text: str) -> str:
    """Normalize Thai text to NFKD (compatibility decomposed)."""
    if not text:
        return ""
    return unicodedata.normalize("NFKD", text)


def decompose_thai_grapheme(text: str) -> List[str]:
    """Decompose Thai text into individual codepoints."""
    if not text:
        return []
    return list(text)


def compose_thai_grapheme(chars: List[str]) -> str:
    """Compose Thai characters back into a string (NFC)."""
    if not chars:
        return ""
    return unicodedata.normalize("NFC", "".join(chars))


def segment_thai_graphemes(text: str) -> List[str]:
    """Segment Thai text into grapheme clusters (base + combining marks)."""
    if not text:
        return []

    clusters = []
    current_cluster = ""

    for ch in text:
        cp = ord(ch)
        is_combining = (
            cp in THAI_COMBINING_CLASSES or
            unicodedata.combining(ch) > 0 or
            is_thai_combining(ch)
        )

        if is_combining and current_cluster:
            current_cluster += ch
        else:
            if current_cluster:
                clusters.append(current_cluster)
            current_cluster = ch

    if current_cluster:
        clusters.append(current_cluster)

    return clusters


def count_thai_graphemes(text: str) -> int:
    """Count the number of grapheme clusters in Thai text."""
    return len(segment_thai_graphemes(text))


def validate_thai_text(text: str) -> Tuple[bool, List[str]]:
    """Validate that text contains only valid Thai characters (U+0E00–U+0E7F) and ASCII.
    
    Returns:
        Tuple of (is_valid, list_of_invalid_chars)
    """
    if not text:
        return True, []

    invalid = []
    for ch in text:
        cp = ord(ch)
        if not (
            (THAI_BLOCK_START <= cp <= THAI_BLOCK_END) or
            (0x0000 <= cp <= 0x007F) or
            cp in (0x00A0, 0x200B, 0x200C, 0x200D, 0xFEFF)
        ):
            invalid.append(ch)

    return len(invalid) == 0, invalid


def filter_thai_chars(text: str, keep_ascii: bool = True) -> str:
    """Filter text to keep only Thai characters and optionally ASCII."""
    if not text:
        return ""

    result = []
    for ch in text:
        cp = ord(ch)
        if THAI_BLOCK_START <= cp <= THAI_BLOCK_END:
            result.append(ch)
        elif keep_ascii and 0x0000 <= cp <= 0x007F:
            result.append(ch)
        elif cp in (0x00A0, 0x200B, 0x200C, 0x200D, 0xFEFF):
            result.append(ch)

    return "".join(result)


def thai_to_arabic_digits(text: str) -> str:
    """Convert Thai digits to Arabic digits."""
    thai_to_arabic = {
        '๐': '0', '๑': '1', '๒': '2', '๓': '3', '๔': '4',
        '๕': '5', '๖': '6', '๗': '7', '๘': '8', '๙': '9',
    }
    return "".join(thai_to_arabic.get(ch, ch) for ch in text)


def arabic_to_thai_digits(text: str) -> str:
    """Convert Arabic digits to Thai digits."""
    arabic_to_thai = {
        '0': '๐', '1': '๑', '2': '๒', '3': '๓', '4': '๔',
        '5': '๕', '6': '๖', '7': '๗', '8': '๘', '9': '๙',
    }
    return "".join(arabic_to_thai.get(ch, ch) for ch in text)


def get_thai_char_name(ch: str) -> str:
    """Get the Unicode name of a Thai character."""
    if not ch or len(ch) != 1:
        return "INVALID"
    cp = ord(ch)
    return ALL_THAI_CHARS.get(cp, unicodedata.name(ch, "UNKNOWN"))


def is_valid_thai_sequence(text: str) -> bool:
    """Check if a Thai text sequence is valid (no orphaned combining marks)."""
    if not text:
        return True

    prev_was_consonant = False
    for ch in text:
        if is_thai_consonant(ch):
            prev_was_consonant = True
        elif is_thai_combining(ch):
            if not prev_was_consonant:
                return False
        else:
            prev_was_consonant = False

    return True


def sort_thai_chars(chars: List[str]) -> List[str]:
    """Sort Thai characters by Unicode codepoint (Thai first, then others)."""
    def sort_key(ch):
        cp = ord(ch)
        if THAI_BLOCK_START <= cp <= THAI_BLOCK_END:
            return (0, cp)
        elif 0x0041 <= cp <= 0x005A or 0x0061 <= cp <= 0x007A:
            return (1, cp)
        elif 0x0030 <= cp <= 0x0039:
            return (2, cp)
        return (3, cp)

    return sorted(chars, key=sort_key)


def get_thai_codepoint_range() -> Tuple[int, int]:
    """Get the Thai Unicode block range."""
    return (THAI_BLOCK_START, THAI_BLOCK_END)


def iterate_thai_block() -> List[ThaiCharInfo]:
    """Iterate through all defined Thai characters in the block."""
    return [get_thai_char_info(chr(cp)) for cp in range(THAI_BLOCK_START, THAI_BLOCK_END + 1)
            if get_thai_char_info(chr(cp)) is not None]


class ThaiUnicodeEngine:
    """Main engine for Thai Unicode processing."""

    def __init__(
        self,
        normalization_form: str = "NFC",
        preserve_zwj: bool = True,
        validate_on_encode: bool = True,
    ):
        self.normalization_form = normalization_form
        self.preserve_zwj = preserve_zwj
        self.validate_on_encode = validate_on_encode

        self._zwj_chars = {"\u200B", "\u200C", "\u200D", "\uFEFF"}

    def normalize(self, text: str) -> str:
        """Normalize text using the configured normalization form."""
        if not text:
            return ""

        if self.preserve_zwj:
            protected = {}
            for i, ch in enumerate(text):
                if ch in self._zwj_chars:
                    protected[f"\x00P{i}\x00"] = ch

            normalized = unicodedata.normalize(self.normalization_form, text)

            for ph, ch in protected.items():
                normalized = normalized.replace(ph, ch)

            return normalized

        return unicodedata.normalize(self.normalization_form, text)

    def encode_to_utf8(self, text: str) -> bytes:
        """Encode text to UTF-8 bytes."""
        if not text:
            return b""
        normalized = self.normalize(text)
        return normalized.encode("utf-8")

    def decode_from_utf8(self, data: bytes) -> str:
        """Decode UTF-8 bytes to text."""
        if not data:
            return ""
        return data.decode("utf-8")

    def validate_and_normalize(self, text: str) -> Tuple[str, List[str]]:
        """Validate Thai text and normalize it."""
        if not text:
            return "", []

        normalized = self.normalize(text)

        if self.validate_on_encode:
            is_valid, invalid = validate_thai_text(normalized)
            if not is_valid:
                return normalized, invalid

        return normalized, []

    def segment_graphemes(self, text: str) -> List[str]:
        """Segment text into grapheme clusters."""
        normalized = self.normalize(text)
        return segment_thai_graphemes(normalized)

    def get_char_info(self, ch: str) -> Optional[ThaiCharInfo]:
        """Get detailed info for a character."""
        return get_thai_char_info(ch)

    def analyze_text(self, text: str) -> Dict:
        """Analyze Thai text and return statistics."""
        if not text:
            return {
                "length": 0,
                "grapheme_count": 0,
                "thai_chars": 0,
                "consonants": 0,
                "vowels": 0,
                "tone_marks": 0,
                "signs": 0,
                "digits": 0,
                "ascii_chars": 0,
                "other_chars": 0,
                "is_valid": True,
                "invalid_chars": [],
            }

        normalized = self.normalize(text)
        graphemes = self.segment_graphemes(normalized)
        is_valid, invalid = validate_thai_text(normalized)

        stats = {
            "length": len(normalized),
            "grapheme_count": len(graphemes),
            "thai_chars": 0,
            "consonants": 0,
            "vowels": 0,
            "tone_marks": 0,
            "signs": 0,
            "digits": 0,
            "ascii_chars": 0,
            "other_chars": 0,
            "is_valid": is_valid,
            "invalid_chars": invalid,
        }

        for ch in normalized:
            cp = ord(ch)
            if THAI_BLOCK_START <= cp <= THAI_BLOCK_END:
                stats["thai_chars"] += 1
                if is_thai_consonant(ch):
                    stats["consonants"] += 1
                elif is_thai_vowel(ch):
                    stats["vowels"] += 1
                elif is_thai_tone_mark(ch):
                    stats["tone_marks"] += 1
                elif is_thai_sign(ch):
                    stats["signs"] += 1
                elif is_thai_digit(ch):
                    stats["digits"] += 1
            elif 0x0000 <= cp <= 0x007F:
                stats["ascii_chars"] += 1
            else:
                stats["other_chars"] += 1

        return stats


DEFAULT_ENGINE = ThaiUnicodeEngine()


def normalize(text: str) -> str:
    """Normalize text using default engine (NFC)."""
    return DEFAULT_ENGINE.normalize(text)


def encode_utf8(text: str) -> bytes:
    """Encode text to UTF-8 using default engine."""
    return DEFAULT_ENGINE.encode_to_utf8(text)


def decode_utf8(data: bytes) -> str:
    """Decode UTF-8 using default engine."""
    return DEFAULT_ENGINE.decode_from_utf8(data)


def segment_graphemes(text: str) -> List[str]:
    """Segment text into grapheme clusters using default engine."""
    return DEFAULT_ENGINE.segment_graphemes(text)


def analyze(text: str) -> Dict:
    """Analyze text using default engine."""
    return DEFAULT_ENGINE.analyze_text(text)


__all__ = [
    "ThaiCharCategory",
    "ThaiCharInfo",
    "ThaiUnicodeEngine",
    "THAI_BLOCK_START",
    "THAI_BLOCK_END",
    "THAI_CONSONANTS",
    "THAI_VOWELS",
    "THAI_TONE_MARKS",
    "THAI_SIGNS",
    "THAI_DIGITS",
    "THAI_PUNCTUATION",
    "ALL_THAI_CHARS",
    "THAI_COMBINING_CLASSES",
    "THAI_LEADING_VOWELS",
    "THAI_ABOVE_VOWELS",
    "THAI_BELOW_VOWELS",
    "THAI_TONE_MARK_CODEPOINTS",
    "THAI_SIGN_CODEPOINTS",
    "THAI_VOWEL_CODEPOINTS",
    "THAI_CONSONANT_CODEPOINTS",
    "THAI_DIGIT_CODEPOINTS",
    "is_thai_char",
    "is_thai_consonant",
    "is_thai_vowel",
    "is_thai_tone_mark",
    "is_thai_sign",
    "is_thai_digit",
    "is_thai_combining",
    "get_thai_char_info",
    "get_thai_category",
    "normalize_thai_nfc",
    "normalize_thai_nfd",
    "normalize_thai_nfkc",
    "normalize_thai_nfkd",
    "decompose_thai_grapheme",
    "compose_thai_grapheme",
    "segment_thai_graphemes",
    "count_thai_graphemes",
    "validate_thai_text",
    "filter_thai_chars",
    "thai_to_arabic_digits",
    "arabic_to_thai_digits",
    "get_thai_char_name",
    "is_valid_thai_sequence",
    "sort_thai_chars",
    "get_thai_codepoint_range",
    "iterate_thai_block",
    "DEFAULT_ENGINE",
    "normalize",
    "encode_utf8",
    "decode_utf8",
    "segment_graphemes",
    "analyze",
]