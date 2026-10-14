"""Name normalization and a script-independent phonetic skeleton key.

Hangul transliterations of English names and catalog English names are both
projected to a coarse consonant-class string so they can be compared directly.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np

# Hangul jamo -> consonant class ("" = silent). Index order follows Unicode.
_INITIAL = "K K N T T L M P P S S _ S S S K T P _".split()
_FINAL = ["", "K", "K", "KS", "N", "NS", "N", "T", "L", "LK", "LM", "LP", "LS", "LT", "LP", "L",
          "M", "P", "PS", "T", "T", "N", "T", "T", "K", "T", "P", ""]
# Korean letter names used when an English acronym is spelled out (KMTC -> 케이엠티씨).
_LETTER = dict(A="", B="P", C="S", D="T", E="", F="P", G="S", H="S", I="", J="S", K="K", L="L",
               M="M", N="N", O="", P="P", Q="K", R="L", S="S", T="T", U="", V="P", W="TPL",
               X="KS", Y="", Z="S")
_EN_DIGRAPHS = [("PH", "P"), ("TH", "S"), ("CH", "S"), ("SH", "S"), ("CK", "K"), ("NG", "N"),
                ("GH", ""), ("X", "KS"), ("Q", "K")]
_EN_CLASS = str.maketrans({"B": "P", "V": "P", "F": "P", "D": "T", "R": "L", "J": "S", "Z": "S",
                           "G": "K", "C": "K"})


def is_hangul(value: object) -> bool:
    return isinstance(value, str) and any("가" <= ch <= "힣" for ch in value)


def normalize_name(value: object) -> str:
    """Uppercase alphanumerics only, with a leading M/V prefix removed."""
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKC", value).upper()
    value = re.sub(r"^\s*M\s*/\s*V\s+", "", value)
    return "".join(ch for ch in value if ch.isalnum())


def _collapse(value: str) -> str:
    return re.sub(r"(.)\1+", r"\1", value.replace(" ", ""))


def _en_skeleton(text: str) -> str:
    s = unicodedata.normalize("NFKC", text).upper()
    s = re.sub(r"\bNO\b\.?", "", s)
    s = re.sub(r"\bM/V\b", "", s)
    s = re.sub(r"\bX", "S", s)  # pinyin: Xiamen -> 샤먼
    for src, dst in _EN_DIGRAPHS:
        s = s.replace(src, dst)
    s = re.sub(r"C(?=[EIY])", "S", s)
    s = re.sub(r"G(?=[EIY])", "J", s)
    s = re.sub(r"R(?![AEIOUY])", "", s)  # non-rhotic: Mercury -> 머큐리, Polar -> 폴라
    s = s.translate(_EN_CLASS)
    s = re.sub(r"[AEIOUWYH]", "", s)
    return _collapse(re.sub(r"[^A-Z0-9 ]", "", s))


def en_skeletons(name: object) -> tuple[str, ...]:
    """Skeleton of an English name, plus a spelled-acronym variant when relevant."""
    if not isinstance(name, str):
        return ("",)
    base = _en_skeleton(name)
    tokens = re.findall(r"[A-Za-z0-9]+", unicodedata.normalize("NFKC", name))
    acronym = [t for t in tokens if t.isalpha() and t.isupper() and 2 <= len(t) <= 5]
    if not acronym:
        return (base,)
    spelled = "".join(
        "".join(_LETTER[c] for c in t) if t in acronym else _en_skeleton(t) for t in tokens
    )
    spelled = _collapse(spelled)
    return (base,) if spelled == base else (base, spelled)


def ko_skeleton(name: object) -> str:
    """Skeleton of a Hangul (possibly mixed) name, comparable with en_skeletons."""
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKC", name).upper().strip()
    s = re.sub(r"^제(?=\d)", "", s)
    s = re.sub(r"호$", "", s)
    out: list[str] = []
    for ch in s:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            code -= 0xAC00
            initial, final = _INITIAL[code // 588], _FINAL[code % 28]
            out.append("" if initial == "_" else initial)
            out.append(final)
        elif ch.isalpha():
            out.append(_en_skeleton(ch))
        elif ch.isdigit():
            out.append(ch)
    return _collapse("".join(out))


_ROMAN_INITIAL = "G KK N D TT R M B PP S SS _ J JJ CH K T P H".split()
_ROMAN_VOWEL = "A AE YA YAE EO E YEO YE O WA WAE OE YO U WO WE WI YU EU UI I".split()
_ROMAN_FINAL = ["", "K", "K", "KS", "N", "NJ", "NH", "T", "L", "LK", "LM", "LB", "LS", "LT", "LP", "LH",
                "M", "P", "PS", "T", "T", "NG", "T", "T", "K", "T", "P", "T"]


def romanize(name: object) -> str:
    """Revised-Romanization-like spelling with vowels; English passes through normalized."""
    if not isinstance(name, str):
        return ""
    s = re.sub(r"호$", "", re.sub(r"^제(?=\d)", "", unicodedata.normalize("NFKC", name).upper().strip()))
    out: list[str] = []
    for ch in s:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            code -= 0xAC00
            initial = _ROMAN_INITIAL[code // 588]
            out.append(("" if initial == "_" else initial) + _ROMAN_VOWEL[(code % 588) // 28] + _ROMAN_FINAL[code % 28])
        elif ch.isalnum():
            out.append(ch)
    return "".join(out)


def query_skeleton(name: object) -> str:
    return ko_skeleton(name) if is_hangul(name) else _en_skeleton(name) if isinstance(name, str) else ""


_KO_NUMBER = dict(원=1, 투=2, 쓰리=3, 포=4, 파이브=5, 식스=6, 세븐=7, 에잇=8, 나인=9, 텐=10)
_EN_NUMBER = dict(ONE=1, TWO=2, THREE=3, FOUR=4, FIVE=5, SIX=6, SEVEN=7, EIGHT=8, NINE=9, TEN=10,
                  I=1, II=2, III=3, IV=4, V=5, VI=6, VII=7, VIII=8, IX=9, X=10)


def hull_numbers(name: object) -> frozenset[int]:
    """Numbers in a name: digits, standalone Korean/English number words, Roman numerals (T9.2)."""
    if not isinstance(name, str):
        return frozenset()
    out = {int(d) for d in re.findall(r"\d+", name)}
    for token in re.findall(r"[가-힣A-Za-z]+", name):
        value = _KO_NUMBER.get(token) or _EN_NUMBER.get(token.upper())
        if value:
            out.add(value)
    return frozenset(out)


def positive_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0 else None


def reference_tonnage(row: dict) -> float | None:
    """International GT first (closer to catalog GT per EDA T1.4), then GT."""
    return positive_float(row.get("international_gross_tonnage")) or positive_float(row.get("gross_tonnage"))


def _check() -> None:
    # Real train pairs (EDA T1.3 / P2 samples).
    assert normalize_name("STOLT-CEDAR") == normalize_name("Stolt Cedar")
    assert normalize_name("M/V HEMISPHERE") == normalize_name("Hemisphere")
    pairs = [("글로비스 시리우스", "Glovis Sirius"), ("루비 에이스", "Ruby Ace"), ("머큐리 에이스", "Mercury Ace"),
             ("폴라 브라이트", "Polar Bright"), ("자이언트 삼부", "Giant Sambu"), ("골든 하치", "Golden Hachi"),
             ("케이엠티씨 서울", "KMTC Seoul"), ("제8에스제이가스", "No. 8 SJ Gas"), ("써니 퀸", "Sunny Queen"),
             ("부에나 반데라", "Buena Bandera"), ("글로비스 챔피언", "Glovis Champion"), ("뉴콘크", "New Konk")]
    for ko, en in pairs:
        assert ko_skeleton(ko) in en_skeletons(en), (ko, ko_skeleton(ko), en_skeletons(en))
    assert ko_skeleton("흥아 샤먼") in en_skeletons("Heung-A Xiamen")
    assert ko_skeleton("골든하나호") == ko_skeleton("골든 하나")
    assert romanize("후난") == "HUNAN" and romanize("제7하나호") == "7HANA"
    assert hull_numbers("제2 오션파이오니어") == {2} == hull_numbers("Ocean Pioneer No. 2")
    assert hull_numbers("씨와이 인터오션 투") == hull_numbers("CY Interocean II") != hull_numbers("CY Interocean I")
    assert reference_tonnage({"gross_tonnage": 100, "international_gross_tonnage": float("nan")}) == 100
    print("normalize ok")


if __name__ == "__main__":
    _check()
