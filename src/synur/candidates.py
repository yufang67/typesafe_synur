"""Transcript-only candidate discovery. No reference labels or model calls."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation


@dataclass(frozen=True)
class Candidate:
    start: int
    end: int
    text: str
    value: str | int | float
    kind: str

    def to_dict(self) -> dict:
        return asdict(self)


_SMALL: dict[str, int] = dict(
    zip(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen".split(),
        range(20),
        strict=True,
    )
)
_TENS: dict[str, int] = dict(
    zip(
        "twenty thirty forty fifty sixty seventy eighty ninety".split(),
        range(20, 100, 10),
        strict=True,
    )
)
_WORDS = (
    set(_SMALL)
    | set(_TENS)
    | {
        "hundred",
        "thousand",
        "million",
        "point",
        "minus",
        "negative",
        "plus",
        "positive",
    }
)
_SCALES: dict[str, int] = {"thousand": 1000, "million": 1_000_000}
_WORD = "(?:" + "|".join(sorted(_WORDS, key=len, reverse=True)) + ")"
_SPOKEN = re.compile(rf"\b{_WORD}\b(?:(?:[\s-]+and)?[\s-]+{_WORD}\b)*", re.IGNORECASE)
_DIGITS = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\.\d)")
_COMPOSITE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:/|:|[-\u2013\u2014]|\bto\b|\bout\s+of\b)\s*\d+(?:\.\d+)?",
    re.IGNORECASE,
)
_SPOKEN_COMPOSITE = re.compile(
    rf"\b{_WORD}(?:[\s-]+{_WORD})*\s+(?:to|out\s+of|over)\s+{_WORD}(?:[\s-]+{_WORD})*\b",
    re.IGNORECASE,
)
_UNSUPPORTED_NUMERIC = re.compile(
    r"(?<!\w)[-+]?\d+(?:\.\d+)?[eE][-+]?\d+|(?<!\w)\d+(?:,\d+)+(?:\.\d+)?"
)
_TEXT_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+|,\s+")
_TOKEN = re.compile(r"\S+")


def _under_thousand(words: list[str]) -> int:
    if not words:
        raise ValueError("Missing number words")
    value = 0
    if len(words) >= 2 and words[0] in _SMALL and 1 <= _SMALL[words[0]] <= 9:
        if words[1] == "hundred":
            value = 100 * _SMALL[words[0]]
            words = words[2:]
            if words and words[0] == "and":
                words = words[1:]
            if not words:
                return value
    if len(words) == 1 and words[0] in _SMALL:
        return value + _SMALL[words[0]]
    if words and words[0] in _TENS:
        value += _TENS[words[0]]
        if len(words) == 1:
            return value
        if len(words) == 2 and words[1] in _SMALL and 1 <= _SMALL[words[1]] <= 9:
            return value + _SMALL[words[1]]
    raise ValueError("Ambiguous or unsupported spoken number")


def parse_number(text: str) -> int | float:
    """Parse a scalar, never a range, ratio, unit conversion, or approximate value."""
    stripped = text.strip().lower()
    if re.fullmatch(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", stripped):
        try:
            decimal = Decimal(stripped.replace(",", ""))
        except InvalidOperation as exc:
            raise ValueError("Invalid numeric candidate") from exc
    else:
        words = stripped.replace("-", " ").split()
        sign = 1
        if words and words[0] in {"minus", "negative", "plus", "positive"}:
            sign = -1 if words[0] in {"minus", "negative"} else 1
            words = words[1:]
        if words.count("point") > 1:
            raise ValueError("Ambiguous decimal")
        fraction = ""
        if "point" in words:
            at = words.index("point")
            digits = words[at + 1 :]
            if not digits or any(word not in _SMALL or _SMALL[word] > 9 for word in digits):
                raise ValueError("Decimal words must be individual digits")
            fraction = "." + "".join(str(_SMALL[word]) for word in digits)
            words = words[:at]
        whole = 0
        last_scale = 1_000_000_000
        while any(word in {"thousand", "million"} for word in words):
            at = next(i for i, word in enumerate(words) if word in {"thousand", "million"})
            scale = _SCALES[words[at]]
            if scale >= last_scale:
                raise ValueError("Ambiguous number scale")
            whole += _under_thousand(words[:at]) * scale
            last_scale = scale
            words = words[at + 1 :]
            if words and words[0] == "and":
                words = words[1:]
        if words:
            whole += _under_thousand(words)
        elif not whole:
            raise ValueError("Missing integer part")
        decimal = sign * Decimal(str(whole) + fraction)
    if decimal == decimal.to_integral_value():
        return int(decimal)
    value = float(decimal)
    if not math.isfinite(value):
        raise ValueError("Non-finite numeric candidate")
    return value


def numeric_candidates(transcript: str) -> tuple[list[Candidate], list[dict]]:
    """Return scalars with exact offsets and explicit unsupported-span diagnostics."""
    issues: list[dict] = []
    excluded = []
    for pattern in (_COMPOSITE, _SPOKEN_COMPOSITE, _UNSUPPORTED_NUMERIC):
        for match in pattern.finditer(transcript):
            if pattern is _UNSUPPORTED_NUMERIC and re.fullmatch(
                r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", match.group()
            ):
                continue
            excluded.append((match.start(), match.end()))
            issues.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "reason": "composite_or_unsupported_numeric_value",
                }
            )
    result = []
    for pattern in (_DIGITS, _SPOKEN):
        for match in pattern.finditer(transcript):
            if any(match.start() < end and match.end() > start for start, end in excluded):
                continue
            try:
                value = parse_number(match.group())
            except ValueError as exc:
                issues.append(
                    {
                        "start": match.start(),
                        "end": match.end(),
                        "text": match.group(),
                        "reason": str(exc),
                    }
                )
                continue
            result.append(Candidate(match.start(), match.end(), match.group(), value, "number"))
    return sorted(result, key=lambda item: (item.start, item.end)), issues


def text_candidates(transcript: str) -> list[Candidate]:
    """Sentence/clause candidates, retaining exact transcript offsets."""
    boundaries = [0]
    ends = []
    for match in _TEXT_SPLIT.finditer(transcript):
        ends.append(match.start())
        boundaries.append(match.end())
    ends.append(len(transcript))
    result = []
    for start, end in zip(boundaries, ends, strict=True):
        while start < end and transcript[start].isspace():
            start += 1
        speaker = re.match(
            r"\[[^\]\n]+\]\s*|(?:Doctor|Patient|Clinician|Nurse):\s*",
            transcript[start:end],
            re.IGNORECASE,
        )
        if speaker:
            start += speaker.end()
        while end > start and transcript[end - 1].isspace():
            end -= 1
        if start < end:
            text = transcript[start:end]
            result.append(Candidate(start, end, text, text, "clause"))
    return result


def phrase_candidates(transcript: str, clause: Candidate) -> list[Candidate]:
    """All contiguous token spans inside the chosen clause, with punctuation trimmed."""
    if transcript[clause.start : clause.end] != clause.text:
        raise ValueError("Clause offsets do not match the transcript")
    tokens = list(_TOKEN.finditer(clause.text))
    result = []
    seen: set[tuple[int, int]] = set()
    for left in range(len(tokens)):
        for right in range(left, len(tokens)):
            start = clause.start + tokens[left].start()
            end = clause.start + tokens[right].end()
            while start < end and transcript[start] in "\"'([{":
                start += 1
            while end > start and transcript[end - 1] in "\"')]}.,;:!?":
                end -= 1
            if start == end or (start, end) in seen:
                continue
            seen.add((start, end))
            text = transcript[start:end]
            result.append(Candidate(start, end, text, text, "phrase"))
    return result


def phrase_starts(transcript: str, clause: Candidate) -> list[Candidate]:
    """Select a first token before selecting the end, avoiding an O(n^2) Choice payload."""
    phrases = phrase_candidates(transcript, clause)
    first_by_start: dict[int, Candidate] = {}
    for candidate in phrases:
        first_by_start.setdefault(candidate.start, candidate)
    return list(first_by_start.values())


def candidate_coverage(
    transcript: str,
    references: list[dict],
    numeric: list[Candidate],
    clauses: list[Candidate],
) -> dict:
    """Post-hoc diagnostic only. Gold labels never affect candidate discovery."""
    pools = candidate_value_pools(transcript, numeric, clauses)
    report = {}
    for kind, pool in pools.items():
        values = [obs["value"] for obs in references if obs.get("value_type") == kind]
        covered = sum(
            any(type(value) is not bool and candidate == value for candidate in pool)
            for value in values
        )
        report[kind] = {
            "reference_values": len(values),
            "covered": covered,
            "coverage": covered / len(values) if values else None,
        }
    return report


def candidate_value_pools(
    transcript: str,
    numeric: list[Candidate],
    clauses: list[Candidate],
) -> dict[str, list[str | int | float]]:
    """Compact value-only diagnostics; selection still uses distinct source offsets."""
    return {
        "NUMERIC": list(dict.fromkeys(candidate.value for candidate in numeric)),
        "STRING": list(
            dict.fromkeys(
                candidate.value
                for clause in clauses
                for candidate in phrase_candidates(transcript, clause)
            )
        ),
    }
