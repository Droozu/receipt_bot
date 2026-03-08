from __future__ import annotations
import re
from pathlib import Path

from app.parser.patterns import LearnedPatterns


class PatternApplier:

    def __init__(self, model_path: Path | str) -> None:
        self.patterns = LearnedPatterns.load(model_path)

    def fix_text(self, text: str) -> str:
            if not text:
                return text

            # OCR corrections
            for wrong, right in self.patterns.corrections.items():
                text = re.sub(
                    rf"\b{re.escape(wrong)}\b",
                    right,
                    text,
                    flags=re.IGNORECASE,
                )

            # Abbreviation expansion
            for abbr, full in self.patterns.abbreviations.items():
                text = re.sub(
                    rf"\b{re.escape(abbr)}\b",
                    full,
                    text,
                    flags=re.IGNORECASE,
                )

            return text

    def apply(self, text: str) -> str:
        text = self._apply_corrections(text)
        text = self._apply_abbreviations(text)

        return text

    def fix_store(self, store_name: str | None) -> str | None:

        if not store_name:
            return store_name

        key = store_name.upper()

        return self.patterns.store_aliases.get(key, store_name)

    def _apply_corrections(self, text: str) -> str:

        for wrong, right in self.patterns.corrections.items():

            text = re.sub(
                rf"\b{re.escape(wrong)}\b",
                right,
                text,
                flags=re.IGNORECASE,
            )

        return text

    def _apply_abbreviations(self, text: str) -> str:

        for abbr, full in self.patterns.abbreviations.items():

            text = re.sub(
                rf"\b{re.escape(abbr)}\b",
                full,
                text,
                flags=re.IGNORECASE,
            )

        return text