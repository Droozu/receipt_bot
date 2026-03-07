from __future__ import annotations

import json
import os
from difflib import get_close_matches

from app.config import MODELS_DIR


DICT_FILE = os.path.join(MODELS_DIR, "dictionary.json")


def _load_dictionary() -> list[str]:
    if not os.path.exists(DICT_FILE):
        return []

    try:
        with open(DICT_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return list(data.keys())
    except Exception:
        return []


def get_dictionary():
    return _load_dictionary()

DICTIONARY = get_dictionary()


def correct_word(word: str) -> str:
    """
    Fix OCR mistakes using learned dictionary
    """

    if not DICTIONARY:
        return word

    word = word.upper()

    matches = get_close_matches(word, DICTIONARY, n=1, cutoff=0.8)

    if matches:
        return matches[0]

    return word


def correct_text(text: str) -> str:
    tokens = text.split()

    corrected = []

    for t in tokens:
        corrected.append(correct_word(t))

    return " ".join(corrected)