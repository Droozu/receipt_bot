from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import ParserConfig
from app.ocr.engine import OCRResult
from app.parser.items_parser import ItemsParser, ParsedItem
from app.parser.patterns import DATE_PATTERNS, INN_PATTERN, LEGAL_ENTITY_PATTERNS, LearnedPatterns, TOTAL_PATTERN


@dataclass(slots=True)
class ReceiptData:
    store_name: str | None
    legal_name: str | None
    datetime: str | None
    items: list[dict[str, Any]]
    total: float | None
    inn: str | None
    confidence: float
    raw_text: str


class ReceiptParser:
    def __init__(self, config: ParserConfig, model_path: str | Path | None = None) -> None:
        self.config = config
        self.items_parser = ItemsParser(config)
        self.learned = LearnedPatterns.load(model_path) if model_path else LearnedPatterns()

    def parse(self, ocr: OCRResult) -> ReceiptData:
        corrected_lines = [self._apply_corrections(line) for line in ocr.lines]
        store_name = self._extract_store_name(corrected_lines)
        legal_name = self._extract_legal_name(corrected_lines)
        dt = self._extract_datetime(corrected_lines)
        items = self.items_parser.parse(corrected_lines)
        total = self._extract_total(corrected_lines)
        inn = self._extract_inn(corrected_lines)
        confidence = self._compute_confidence(ocr, store_name, legal_name, dt, items, total)
        return ReceiptData(
            store_name=store_name,
            legal_name=legal_name,
            datetime=dt,
            items=[asdict(item) for item in items],
            total=total,
            inn=inn,
            confidence=round(confidence, 4),
            raw_text="\n".join(corrected_lines),
        )

    def to_json(self, receipt: ReceiptData) -> str:
        return json.dumps(asdict(receipt), ensure_ascii=False, indent=2)

    def _apply_corrections(self, line: str) -> str:
        fixed = line
        for wrong, correct in self.learned.corrections.items():
            fixed = re.sub(rf"\b{re.escape(wrong)}\b", correct, fixed, flags=re.IGNORECASE)
        return fixed

    def _extract_store_name(self, lines: list[str]) -> str | None:
        header = lines[:8]
        candidates: list[str] = []
        for line in header:
            norm = re.sub(r"[^\wА-Яа-яЁё\s\-\"']", " ", line).strip()
            if len(norm) < 3:
                continue
            if any(x in norm.upper() for x in ["КАССОВ", "ЧЕК", "ИНН", "ООО", "АО", "ИП"]):
                continue
            candidates.append(norm)
        if not candidates:
            return None
        best = max(candidates, key=lambda s: (sum(ch.isalpha() for ch in s), -len(s.split())))
        return self.learned.store_aliases.get(best.upper(), best)

    def _extract_legal_name(self, lines: list[str]) -> str | None:
        for line in lines[:20]:
            for pattern in LEGAL_ENTITY_PATTERNS:
                match = pattern.search(line)
                if match:
                    return match.group(0).strip()
        return None

    def _extract_datetime(self, lines: list[str]) -> str | None:
        for line in lines:
            for pattern in DATE_PATTERNS:
                match = pattern.search(line)
                if match:
                    raw = f"{match.group('date')} {match.group('time')}"
                    parsed = self._try_parse_datetime(raw)
                    return parsed or raw
        return None

    def _try_parse_datetime(self, raw: str) -> str | None:
        formats = [
            "%d.%m.%Y %H:%M",
            "%d.%m.%y %H:%M",
            "%d/%m/%Y %H:%M",
            "%d/%m/%y %H:%M",
            "%Y-%m-%d %H:%M",
            "%d.%m.%Y %H:%M:%S",
            "%d.%m.%y %H:%M:%S",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(raw, fmt).isoformat(timespec="minutes")
            except ValueError:
                continue
        return None

    def _extract_total(self, lines: list[str]) -> float | None:
        for line in reversed(lines):
            match = TOTAL_PATTERN.search(line)
            if match:
                return float(match.group(2).replace(",", "."))
        return None

    def _extract_inn(self, lines: list[str]) -> str | None:
        for line in lines:
            match = INN_PATTERN.search(line)
            if match:
                return match.group(1)
        return None

    def _compute_confidence(
        self,
        ocr: OCRResult,
        store_name: str | None,
        legal_name: str | None,
        dt: str | None,
        items: list[ParsedItem],
        total: float | None,
    ) -> float:
        score = 0.0
        score += min(ocr.mean_confidence, 1.0) * 0.35
        score += 0.10 if store_name else 0.0
        score += 0.10 if legal_name else 0.0
        score += 0.10 if dt else 0.0
        score += min(len(items) / 5, 1.0) * 0.25
        score += 0.10 if total else 0.0
        if items:
            score += sum(item.line_confidence for item in items) / len(items) * 0.10
        return min(score, 1.0)
