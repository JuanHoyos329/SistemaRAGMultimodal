import html
import re
import unicodedata


_ACUTE = {
    "a": "\u00e1", "e": "\u00e9", "i": "\u00ed", "o": "\u00f3", "u": "\u00fa",
    "A": "\u00c1", "E": "\u00c9", "I": "\u00cd", "O": "\u00d3", "U": "\u00da",
}
_ACCENT_BEFORE_VOWEL = re.compile("\u00b4([aeiouAEIOU])")
_ACCENT_AFTER_VOWEL = re.compile("([aeiouAEIOU])\u00b4")


def normalize_extracted_text(value: str) -> str:
    """Clean common PDF text artifacts while retaining the original wording."""
    text = html.unescape(value).replace("\u00ad", "").replace("\u00a0", " ")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00b4\u0131", "\u00ed").replace("\u00b4\u0130", "\u00cd")
    text = _ACCENT_BEFORE_VOWEL.sub(lambda match: _ACUTE[match.group(1)], text)
    text = _ACCENT_AFTER_VOWEL.sub(lambda match: _ACUTE[match.group(1)], text)
    return text
