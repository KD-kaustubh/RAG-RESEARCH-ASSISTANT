"""Cleanup for text extracted from PDFs.

Extraction interleaves page furniture (page numbers, figure captions) with the
running text and breaks words at line ends. Both hurt chunking and retrieval, so
this module repairs them without deleting real content.
"""

import re
from typing import List, Set, Tuple

CAPTION = re.compile(r"^(Figure|Table)\s+\d+\s*[:.]")
PAGE_NUMBER = re.compile(r"^\d{1,4}$")
HYPHEN_BREAK = re.compile(r"([A-Za-z]{2,})-\n([a-z]+)")
WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")

SENTENCE_END = (".", "!", "?", ":", ";")


def build_vocabulary(texts: List[str]) -> Set[str]:
    """Words the document uses elsewhere, used to judge hyphenated line breaks."""
    vocabulary = set()
    for text in texts:
        vocabulary.update(word.lower() for word in WORD.findall(text.replace("\n", " ")))
    return vocabulary


def strip_page_number(text: str) -> str:
    """Drop a bare page number, but only where headers and footers live."""
    lines = text.split("\n")
    filled = [index for index, line in enumerate(lines) if line.strip()]
    if not filled:
        return text

    drop = {index for index in (filled[0], filled[-1]) if PAGE_NUMBER.match(lines[index].strip())}
    if not drop:
        return text
    return "\n".join(line for index, line in enumerate(lines) if index not in drop)


def take_leading_captions(text: str) -> Tuple[List[str], str]:
    """Separate figure/table captions sitting at the top of a page from the body text."""
    lines = text.split("\n")
    captions = []
    index = 0

    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if CAPTION.match(line):
            captions.append(lines[index])
            index += 1
            continue
        break

    if not captions:
        return [], text
    return captions, "\n".join(lines[index:]).lstrip("\n")


def join_hyphenated(text: str, vocabulary: Set[str]) -> str:
    """Rejoin words broken at a line end.

    The hyphen is kept unless the document uses the merged spelling elsewhere, so
    real compounds such as "position-wise" survive while "convolu-tional" is healed.
    """

    def replace(match):
        left, right = match.group(1), match.group(2)
        if (left + right).lower() in vocabulary:
            return left + right
        return f"{left}-{right}"

    return HYPHEN_BREAK.sub(replace, text)


def continues_sentence(text: str) -> bool:
    """True when a page stops mid-sentence and the next page carries it on."""
    stripped = text.rstrip()
    if not stripped:
        return False
    return not stripped.endswith(SENTENCE_END)
