from __future__ import annotations

import re
from dataclasses import dataclass
from app.learning.corrector import correct_text
from app.config import ParserConfig
from app.parser.patterns import MONEY_PATTERN



@dataclass(slots=True)
class ParsedItem:
    name: str
    quantity: float
    price: float
    line_confidence: float


class ItemsParser:
    def __init__(self, config: ParserConfig) -> None:
        self.config = config
    
        self.qty_price_inline = re.compile(
            r"(?P<name>.+?)\s+(?P<price>\d+[.,]\d{2})\s*[*xх4]\s*(?P<qty>\d+[.,]?\d*)\s*[=:]?\s*(?P<total>\d+[.,]\d{2})?",
            re.IGNORECASE,
        )
        self.two_line_value = re.compile(
            r"(?P<price>\d+[.,]\d{2})\s*[*xх4]\s*(?P<qty>\d+[.,]?\d*)\s*(?P<total>\d+[.,]\d{2})?",
            re.IGNORECASE,
        )


    def _looks_like_garbage(self, line: str) -> bool:

        if len(line) < 3:
            return True

        if re.fullmatch(r"[0-9 .,:-]+", line):
            return True

        return False
    
    
    def parse(self, lines: list[str]) -> list[ParsedItem]:
        cleaned = [self._normalize_line(line) for line in lines if line.strip()]
        items: list[ParsedItem] = []
        i = 0
        name_buffer: list[str] = []
        while i < len(cleaned):
            line = cleaned[i]
            if self._looks_like_garbage(line):
                i += 1
                continue
        
            if self._looks_like_service_line(line):
                if self._looks_like_name(line):
                    name_buffer.append(line)
                    i += 1
                    continue
                i += 1
                continue
            inline = self.qty_price_inline.search(line)
            if inline and self._looks_like_name(inline.group("name")):
                name = self._cleanup_name(inline.group("name"))
                dict_score = self._dictionary_score(name)
                confidence = min(1.0, 0.85 + dict_score * 0.15)
                items.append(
                    ParsedItem(
                        name=name,
                        quantity=self._to_float(inline.group("qty")),
                        price=self._to_float(inline.group("price")),
                        line_confidence=confidence,
                    )
                )
                i += 1
                continue

            match = None
            maybe_name = line
            if i + 1 < len(cleaned):
                maybe_values = cleaned[i + 1]
                maybe_values = re.sub(r"(\d+[.,]\d{2})\s+\1", r"\1", maybe_values)
                match = self.two_line_value.search(maybe_values)
                
            if match and self._looks_like_name(maybe_name):

                if name_buffer:
                    maybe_name = " ".join(name_buffer + [maybe_name])
                    name_buffer.clear()

                name = self._cleanup_name(maybe_name)

                dict_score = self._dictionary_score(name)
                confidence = min(1.0, 0.80 + dict_score * 0.2)

                items.append(
                    ParsedItem(
                        name=name,
                        quantity=self._to_float(match.group("qty")),
                        price=self._to_float(match.group("price")),
                        line_confidence=confidence,
                    )
                )

                i += 2
                continue
            if self._looks_like_name(line):
                values = MONEY_PATTERN.findall(line)
                if len(values) >= 2:
                    qty = 1.0
                    price = self._to_float(values[0])
                    items.append(
                        ParsedItem(
                            name=self._cleanup_name(re.sub(MONEY_PATTERN, "", line)),
                            quantity=qty,
                            price=price,
                            line_confidence=0.58,
                        )
                    )
            i += 1
        return self._deduplicate(items)

    def _normalize_line(self, line: str) -> str:
        # Безопасные замены — не трогают буквы в словах
        line = line.replace("=", " ")
        line = line.replace("|", " ")
        line = line.replace("_", " ")
        line = line.replace("—", "-")
        line = line.replace(",", ".")
        line = re.sub(r"\s+", " ", line)
        return line.strip()

    def _normalize_numeric(self, value: str) -> str:
        """Применяется только к уже извлечённому числовому токену."""
        value = value.replace("О", "0").replace("O", "0")
        value = value.replace("I", "1").replace("l", "1")
        value = value.replace("B", "8")
        value = value.replace(",", ".")
        return value

    def _looks_like_service_line(self, line: str) -> bool:
        upper = line.upper()
        blocked = (
            "ИНН", "КАССИР", "СДАЧА", "ИТОГ", "НАЛИЧ", "ЭЛЕКТРОН", "НДС", "САЙТ",
            "ККТ", "ПРИХОД", "ПОДЫТОГ", "СКИД", "СМЕНА", "ЧЕК", "КОД",
        )
        return any(token in upper for token in blocked)

    def _looks_like_name(self, text: str) -> bool:
        text = text.strip()
        if len(text) < self.config.min_item_name_len:
            return False
        digits_ratio = sum(ch.isdigit() for ch in text) / max(len(text), 1)
        return digits_ratio < 0.45 and any(ch.isalpha() for ch in text)

    def _cleanup_name(self, name: str) -> str:

        # remove product code
        name = re.sub(r"^\^?\d{5,}\s+", "", name)
        name = re.sub(r"^\d+[.)]?\s*", "", name)

        name = re.sub(r"\s+", " ", name)

        name = name.strip(" -*")

        name = correct_text(name)

        return name

    def _to_float(self, value: str) -> float:
        return float(self._normalize_numeric(value))

    def _deduplicate(self, items: list[ParsedItem]) -> list[ParsedItem]:
        result: list[ParsedItem] = []
        for item in items:
            if (
            item.name
            and item.price > 0
            and item.quantity > 0
            and item.line_confidence > 0.4
            ):
                result.append(item)
        return result
    
    def _dictionary_score(self, name: str) -> float:

        try:
            from app.learning.corrector import DICTIONARY
        except:
            return 0.0

        tokens = name.upper().split()

        if not tokens:
            return 0.0

        hits = sum(1 for t in tokens if t in DICTIONARY)

        return hits / len(tokens)
