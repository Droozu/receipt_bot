from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
import json


DATE_PATTERNS = [
    re.compile(r"(?P<date>\b\d{2}[./-]\d{2}[./-]\d{2,4}\b)\s+(?P<time>\b\d{2}:\d{2}(?::\d{2})?\b)"),
    re.compile(r"(?P<date>\b\d{4}[./-]\d{2}[./-]\d{2}\b)\s+(?P<time>\b\d{2}:\d{2}(?::\d{2})?\b)"),
]
LEGAL_ENTITY_PATTERNS = [
    re.compile(r"\b(ООО|ЗАО|ОАО|АО|ИП)\b.*", re.IGNORECASE),
]
INN_PATTERN = re.compile(r"\bИНН\s*[:№]?\s*(\d{10,12})\b", re.IGNORECASE)
TOTAL_PATTERN = re.compile(r"\b(ИТОГ|ПОДЫТОГ|СУММА\s+ПО\s+ЧЕКУ|ВСЕГО)\b.*?(\d+[.,]\d{2})", re.IGNORECASE)
MONEY_PATTERN = re.compile(r"\d+[.,]\d{2}")
QTY_PATTERN = re.compile(
    r"(?:\d+[.,]\d{2}\s+)?(\d+[.,]?\d*)\s*(?:x|х|\*)\s*(\d+[.,]\d{2})",
    re.IGNORECASE,
)


@dataclass(slots=True)
class LearnedPatterns:
    dictionary: set[str] = field(default_factory=set)
    store_aliases: dict[str, str] = field(default_factory=dict)
    corrections: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, model_path: str | Path) -> "LearnedPatterns":
        path = Path(model_path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            dictionary=set(data.get("dictionary", [])),
            store_aliases=dict(data.get("store_aliases", {})),
            corrections=dict(data.get("corrections", {})),
        )

    def save(self, model_path: str | Path) -> None:
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dictionary": sorted(self.dictionary),
            "store_aliases": self.store_aliases,
            "corrections": self.corrections,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
