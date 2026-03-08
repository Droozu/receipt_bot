from dataclasses import dataclass  
from typing import Any

@dataclass
class Item:
    name: str
    quantity: float | None
    price: float | None

@dataclass(slots=True)
class OCRWord:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]
    line_num: int

@dataclass(slots=True)
class OCRResult:
    text: str
    lines: list[str]
    words: list[OCRWord]
    mean_confidence: float
    metadata: dict[str, Any]

    