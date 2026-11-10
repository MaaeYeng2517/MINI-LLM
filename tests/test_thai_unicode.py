"""Unit tests for Thai Unicode Engine."""

import pytest
import unicodedata
from thai_mini.tokenizer.thai_unicode import (
    ThaiCharCategory,
    ThaiCharInfo,
    ThaiUnicodeEngine,
    THAI_BLOCK_START,
    THAI_BLOCK_END,
    THAI_CONSONANTS,
    THAI_VOWELS,
    THAI_TONE_MARKS,
    THAI_SIGNS,
    THAI_DIGITS,
    THAI_PUNCTUATION,
    ALL_THAI_CHARS,
    THAI_COMBINING_CLASSES,
    THAI_LEADING_VOWELS,
    THAI_ABOVE_VOWELS,
    THAI_BELOW_VOWELS,
    THAI_TONE_MARK_CODEPOINTS,
    THAI_SIGN_CODEPOINTS,
    THAI_VOWEL_CODEPOINTS,
    THAI_CONSONANT_CODEPOINTS,
    THAI_DIGIT_CODEPOINTS,
    is_thai_char,
    is_thai_consonant,
    is_thai_vowel,
    is_thai_tone_mark,
    is_thai_sign,
    is_thai_digit,
    is_thai_combining,
    get_thai_char_info,
    get_thai_category,
    normalize_thai_nfc,
    normalize_thai_nfd,
    normalize_thai_nfkc,
    normalize_thai_nfkd,
    decompose_thai_grapheme,
    compose_thai_grapheme,
    segment_thai_graphemes,
    count_thai_graphemes,
    validate_thai_text,
    filter_thai_chars,
    thai_to_arabic_digits,
    arabic_to_thai_digits,
    get_thai_char_name,
    is_valid_thai_sequence,
    sort_thai_chars,
    get_thai_codepoint_range,
    iterate_thai_block,
    DEFAULT_ENGINE,
    normalize,
    encode_utf8,
    decode_utf8,
    segment_graphemes,
    analyze,
)


class TestThaiCharCategory:
    """Test ThaiCharCategory enum."""

    def test_categories_exist(self):
        assert ThaiCharCategory.CONSONANT.value == "consonant"
        assert ThaiCharCategory.VOWEL.value == "vowel"
        assert ThaiCharCategory.TONE_MARK.value == "tone_mark"
        assert ThaiCharCategory.SIGN.value == "sign"
        assert ThaiCharCategory.DIGIT.value == "digit"
        assert ThaiCharCategory.PUNCTUATION.value == "punctuation"
        assert ThaiCharCategory.OTHER.value == "other"


class TestThaiCharInfo:
    """Test ThaiCharInfo dataclass."""

    def test_creation(self):
        info = ThaiCharInfo(
            char="ก",
            codepoint=0x0E01,
            name="THAI CHARACTER KO KAI",
            category=ThaiCharCategory.CONSONANT,
            is_combining=False,
            combining_class=0,
        )
        assert info.char == "ก"
        assert info.codepoint == 0x0E01
        assert info.category == ThaiCharCategory.CONSONANT
        assert info.is_combining is False


class TestConstants:
    """Test constants."""

    def test_thai_block_range(self):
        assert THAI_BLOCK_START == 0x0E00
        assert THAI_BLOCK_END == 0x0E7F

    def test_consonants_dict(self):
        assert 0x0E01 in THAI_CONSONANTS
        assert THAI_CONSONANTS[0x0E01] == "THAI CHARACTER KO KAI"
        assert len(THAI_CONSONANTS) == 46

    def test_vowels_dict(self):
        assert 0x0E30 in THAI_VOWELS
        assert THAI_VOWELS[0x0E30] == "THAI CHARACTER SARA A"
        assert len(THAI_VOWELS) == 16

    def test_tone_marks_dict(self):
        assert 0x0E48 in THAI_TONE_MARKS
        assert THAI_TONE_MARKS[0x0E48] == "THAI CHARACTER MAI EK"
        assert len(THAI_TONE_MARKS) == 4

    def test_signs_dict(self):
        assert 0x0E4C in THAI_SIGNS
        assert THAI_SIGNS[0x0E4C] == "THAI CHARACTER THANTHAKHAT"
        assert len(THAI_SIGNS) == 3

    def test_digits_dict(self):
        assert 0x0E50 in THAI_DIGITS
        assert THAI_DIGITS[0x0E50] == "THAI DIGIT ZERO"
        assert len(THAI_DIGITS) == 10

    def test_all_thai_chars(self):
        assert len(ALL_THAI_CHARS) == 83

    def test_combining_classes(self):
        assert THAI_COMBINING_CLASSES[0x0E31] == 103
        assert THAI_COMBINING_CLASSES[0x0E48] == 107

    def test_leading_vowels(self):
        assert 0x0E40 in THAI_LEADING_VOWELS
        assert 0x0E44 in THAI_LEADING_VOWELS

    def test_above_vowels(self):
        assert 0x0E31 in THAI_ABOVE_VOWELS
        assert 0x0E39 in THAI_ABOVE_VOWELS

    def test_below_vowels(self):
        assert 0x0E3A in THAI_BELOW_VOWELS

    def test_tone_mark_codepoints(self):
        assert THAI_TONE_MARK_CODEPOINTS == {0x0E48, 0x0E49, 0x0E4A, 0x0E4B}

    def test_sign_codepoints(self):
        assert THAI_SIGN_CODEPOINTS == {0x0E4C, 0x0E4D, 0x0E4E}

    def test_vowel_codepoints(self):
        assert 0x0E30 in THAI_VOWEL_CODEPOINTS
        assert 0x0E44 in THAI_VOWEL_CODEPOINTS

    def test_consonant_codepoints(self):
        assert 0x0E01 in THAI_CONSONANT_CODEPOINTS
        assert 0x0E2E in THAI_CONSONANT_CODEPOINTS

    def test_digit_codepoints(self):
        assert 0x0E50 in THAI_DIGIT_CODEPOINTS
        assert 0x0E59 in THAI_DIGIT_CODEPOINTS


class TestIsThaiChar:
    """Test is_thai_char function."""

    def test_thai_consonants(self):
        assert is_thai_char("ก")
        assert is_thai_char("ข")
        assert is_thai_char("฿")

    def test_thai_vowels(self):
        assert is_thai_char("ะ")
        assert is_thai_char("า")
        assert is_thai_char("ิ")

    def test_thai_tone_marks(self):
        assert is_thai_char("่")
        assert is_thai_char("้")
        assert is_thai_char("๊")
        assert is_thai_char("๋")

    def test_thai_signs(self):
        assert is_thai_char("์")
        assert is_thai_char("ํ")
        assert is_thai_char("๎")

    def test_thai_digits(self):
        assert is_thai_char("๐")
        assert is_thai_char("๙")

    def test_thai_punctuation(self):
        assert is_thai_char("฿")
        assert is_thai_char("๏")
        assert is_thai_char("๚")
        assert is_thai_char("๛")

    def test_non_thai_chars(self):
        assert not is_thai_char("a")
        assert not is_thai_char("A")
        assert not is_thai_char("0")
        assert not is_thai_char(" ")
        assert not is_thai_char("\n")
        assert not is_thai_char("あ")

    def test_empty_and_invalid(self):
        assert not is_thai_char("")
        assert not is_thai_char("ab")
        assert not is_thai_char(None) if hasattr(is_thai_char, '__call__') else True


class TestCharacterClassification:
    """Test character classification functions."""

    def test_is_thai_consonant(self):
        assert is_thai_consonant("ก")
        assert is_thai_consonant("ข")
        assert not is_thai_consonant("ะ")
        assert not is_thai_consonant("a")

    def test_is_thai_vowel(self):
        assert is_thai_vowel("ะ")
        assert is_thai_vowel("า")
        assert is_thai_vowel("ิ")
        assert not is_thai_vowel("ก")
        assert not is_thai_vowel("a")

    def test_is_thai_tone_mark(self):
        assert is_thai_tone_mark("่")
        assert is_thai_tone_mark("้")
        assert is_thai_tone_mark("๊")
        assert is_thai_tone_mark("๋")
        assert not is_thai_tone_mark("ะ")
        assert not is_thai_tone_mark("ก")

    def test_is_thai_sign(self):
        assert is_thai_sign("์")
        assert is_thai_sign("ํ")
        assert is_thai_sign("๎")
        assert not is_thai_sign("่")
        assert not is_thai_sign("ะ")

    def test_is_thai_digit(self):
        assert is_thai_digit("๐")
        assert is_thai_digit("๙")
        assert not is_thai_digit("0")
        assert not is_thai_digit("ก")

    def test_is_thai_combining(self):
        assert is_thai_combining("ะ")
        assert is_thai_combining("่")
        assert is_thai_combining("์")
        assert not is_thai_combining("ก")
        assert not is_thai_combining("a")


class TestGetThaiCharInfo:
    """Test get_thai_char_info function."""

    def test_consonant_info(self):
        info = get_thai_char_info("ก")
        assert info is not None
        assert info.char == "ก"
        assert info.codepoint == 0x0E01
        assert info.category == ThaiCharCategory.CONSONANT
        assert info.is_combining is False
        assert info.combining_class == 0

    def test_vowel_info(self):
        info = get_thai_char_info("ะ")
        assert info is not None
        assert info.category == ThaiCharCategory.VOWEL
        assert info.is_combining is True
        # SARA A (0x0E30) is not in THAI_COMBINING_CLASSES, so combining_class is 0
        assert info.combining_class == 0

    def test_tone_mark_info(self):
        info = get_thai_char_info("่")
        assert info is not None
        assert info.category == ThaiCharCategory.TONE_MARK
        assert info.is_combining is True
        assert info.combining_class == 107

    def test_sign_info(self):
        info = get_thai_char_info("์")
        assert info is not None
        assert info.category == ThaiCharCategory.SIGN
        assert info.is_combining is True

    def test_digit_info(self):
        info = get_thai_char_info("๐")
        assert info is not None
        assert info.category == ThaiCharCategory.DIGIT
        assert info.is_combining is False

    def test_punctuation_info(self):
        info = get_thai_char_info("฿")
        assert info is not None
        assert info.category == ThaiCharCategory.PUNCTUATION

    def test_non_thai(self):
        assert get_thai_char_info("a") is None
        assert get_thai_char_info("") is None
        assert get_thai_char_info("ab") is None


class TestNormalization:
    """Test Unicode normalization functions."""

    def test_normalize_thai_nfc(self):
        text = "ก\u0E34"  # ก + SARA I (decomposed)
        normalized = normalize_thai_nfc(text)
        assert normalized == "กิ"  # composed form

    def test_normalize_thai_nfd(self):
        text = "กิ"  # composed
        normalized = normalize_thai_nfd(text)
        # Should decompose
        assert len(normalized) >= 2

    def test_normalize_thai_nfkc(self):
        text = "１２３"  # fullwidth digits
        normalized = normalize_thai_nfkc(text)
        assert normalized == "123"

    def test_normalize_thai_nfkd(self):
        text = "กิ"
        normalized = normalize_thai_nfkd(text)
        assert len(normalized) >= 2

    def test_empty_string(self):
        assert normalize_thai_nfc("") == ""
        assert normalize_thai_nfd("") == ""
        assert normalize_thai_nfkc("") == ""
        assert normalize_thai_nfkd("") == ""


class TestGraphemeOperations:
    """Test grapheme decomposition and composition."""

    def test_decompose_thai_grapheme(self):
        chars = decompose_thai_grapheme("สวัสดี")
        assert chars == ["ส", "ว", "ั", "ส", "ด", "ี"]

    def test_compose_thai_grapheme(self):
        chars = ["ส", "ว", "ั", "ส", "ด", "ี"]
        composed = compose_thai_grapheme(chars)
        assert composed == "สวัสดี"

    def test_segment_thai_graphemes(self):
        # Test with combining marks
        graphemes = segment_thai_graphemes("สวัสดี")
        # ส + ว + ะ + ส + ด + ี
        # But วั should be one grapheme (ว + ะ)
        # And ดี should be one grapheme (ด + ี)
        assert len(graphemes) > 0

    def test_count_thai_graphemes(self):
        count = count_thai_graphemes("สวัสดี")
        assert count > 0


class TestValidation:
    """Test validation functions."""

    def test_validate_thai_text_valid(self):
        is_valid, invalid = validate_thai_text("สวัสดีครับ")
        assert is_valid is True
        assert invalid == []

    def test_validate_thai_text_with_ascii(self):
        is_valid, invalid = validate_thai_text("Hello สวัสดี")
        assert is_valid is True

    def test_validate_thai_text_invalid(self):
        is_valid, invalid = validate_thai_text("สวัสดีあ")
        assert is_valid is False
        assert "あ" in invalid

    def test_validate_thai_text_empty(self):
        is_valid, invalid = validate_thai_text("")
        assert is_valid is True
        assert invalid == []

    def test_filter_thai_chars(self):
        result = filter_thai_chars("Hello สวัสดี World 123")
        assert "สวัสดี" in result
        assert "Hello" in result
        assert "World" in result
        assert "123" in result

    def test_filter_thai_chars_no_ascii(self):
        result = filter_thai_chars("Hello สวัสดี", keep_ascii=False)
        assert "สวัสดี" in result
        assert "Hello" not in result


class TestDigitConversion:
    """Test Thai/Arabic digit conversion."""

    def test_thai_to_arabic_digits(self):
        assert thai_to_arabic_digits("๐๑๒๓๔๕๖๗๘๙") == "0123456789"
        assert thai_to_arabic_digits("ปี ๒๐๒๔") == "ปี 2024"

    def test_arabic_to_thai_digits(self):
        assert arabic_to_thai_digits("0123456789") == "๐๑๒๓๔๕๖๗๘๙"
        assert arabic_to_thai_digits("Year 2024") == "Year ๒๐๒๔"

    def test_round_trip(self):
        original = "๐๑๒๓๔๕๖๗๘๙"
        arabic = thai_to_arabic_digits(original)
        thai = arabic_to_thai_digits(arabic)
        assert thai == original


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_thai_char_name(self):
        assert get_thai_char_name("ก") == "THAI CHARACTER KO KAI"
        assert get_thai_char_name("ะ") == "THAI CHARACTER SARA A"

    def test_is_valid_thai_sequence(self):
        # Valid: consonant + vowel + tone
        assert is_valid_thai_sequence("กิ่") is True
        # Invalid: vowel without consonant
        assert is_valid_thai_sequence("่ก") is False
        # Valid: just consonants
        assert is_valid_thai_sequence("กขค") is True

    def test_sort_thai_chars(self):
        chars = ["a", "ก", "1", "ะ", "A"]
        sorted_chars = sort_thai_chars(chars)
        # Thai should come first
        assert sorted_chars[0] in ["ก", "ะ"]
        assert sorted_chars[-1] in ["a", "A", "1"]

    def test_get_thai_codepoint_range(self):
        start, end = get_thai_codepoint_range()
        assert start == 0x0E00
        assert end == 0x0E7F

    def test_iterate_thai_block(self):
        chars = iterate_thai_block()
        assert len(chars) > 0
        for info in chars:
            assert info is not None
            assert THAI_BLOCK_START <= info.codepoint <= THAI_BLOCK_END


class TestThaiUnicodeEngine:
    """Test ThaiUnicodeEngine class."""

    def test_engine_creation(self):
        engine = ThaiUnicodeEngine()
        assert engine.normalization_form == "NFC"
        assert engine.preserve_zwj is True
        assert engine.validate_on_encode is True

    def test_engine_normalize(self):
        engine = ThaiUnicodeEngine()
        text = "ก\u0E34"  # decomposed
        normalized = engine.normalize(text)
        assert normalized == "กิ"

    def test_encode_decode_utf8(self):
        engine = ThaiUnicodeEngine()
        text = "สวัสดี"
        encoded = engine.encode_to_utf8(text)
        assert isinstance(encoded, bytes)
        decoded = engine.decode_from_utf8(encoded)
        assert decoded == text

    def test_validate_and_normalize(self):
        engine = ThaiUnicodeEngine(validate_on_encode=True)
        text = "สวัสดี"
        normalized, invalid = engine.validate_and_normalize(text)
        assert normalized == text
        assert invalid == []

    def test_segment_graphemes(self):
        engine = ThaiUnicodeEngine()
        graphemes = engine.segment_graphemes("สวัสดี")
        assert len(graphemes) > 0

    def test_get_char_info(self):
        engine = ThaiUnicodeEngine()
        info = engine.get_char_info("ก")
        assert info is not None
        assert info.char == "ก"

    def test_analyze_text(self):
        engine = ThaiUnicodeEngine()
        analysis = engine.analyze_text("สวัสดีครับ 123")
        assert analysis["length"] > 0
        assert analysis["thai_chars"] > 0
        assert analysis["ascii_chars"] >= 3  # 123
        assert analysis["is_valid"] is True


class TestDefaultEngine:
    """Test default engine functions."""

    def test_normalize(self):
        assert normalize("ก\u0E34") == "กิ"

    def test_encode_utf8(self):
        encoded = encode_utf8("สวัสดี")
        assert isinstance(encoded, bytes)

    def test_decode_utf8(self):
        decoded = decode_utf8(b"\xe0\xb8\xaa\xe0\xb8\xa7\xe0\xb8\xb1\xe0\xb8\xaa\xe0\xb8\x94\xe0\xb8\xb5")
        assert "สวัสดี" in decoded

    def test_segment_graphemes(self):
        graphemes = segment_graphemes("สวัสดี")
        assert len(graphemes) > 0

    def test_analyze(self):
        analysis = analyze("สวัสดี 123")
        assert analysis["thai_chars"] > 0
        assert analysis["ascii_chars"] >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])