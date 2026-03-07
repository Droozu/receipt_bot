from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


BASE_DIR = Path(__file__).resolve().parent

STORAGE_DIR = BASE_DIR / "storage"

MODELS_DIR = STORAGE_DIR / "models"
SAMPLES_DIR = STORAGE_DIR / "samples"
CORRECTIONS_DIR = STORAGE_DIR / "corrections"


@dataclass(slots=True)
class OCRConfig:
    engine: Literal["auto", "tesseract", "easyocr"] = "auto"
    languages: tuple[str, ...] = ("rus", "eng")
    min_word_confidence: float = 0.25
    psm: int = 6
    oem: int = 3
    use_orientation_detection: bool = True
    use_perspective_fix: bool = True


@dataclass(slots=True)
class ParserConfig:
    low_confidence_threshold: float = 0.72
    item_price_tolerance: float = 0.15
    max_line_join_gap: int = 1
    max_item_name_lines: int = 2
    min_item_name_len: int = 3


@dataclass(slots=True)
class StorageConfig:
    base_dir: Path = Path(__file__).resolve().parent / "storage"
    samples_dir: Path = field(init=False)
    corrections_dir: Path = field(init=False)
    models_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.samples_dir = self.base_dir / "samples"
        self.corrections_dir = self.base_dir / "corrections"
        self.models_dir = self.base_dir / "models"
        for path in (self.base_dir, self.samples_dir, self.corrections_dir, self.models_dir):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class AppConfig:
    ocr: OCRConfig = field(default_factory=OCRConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

DEFAULT_CONFIG = AppConfig()
