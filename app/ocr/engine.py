from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from app.config import OCRConfig
from app.ocr.preprocess import ReceiptPreprocessor


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


class OCREngine:
    def __init__(self, config: OCRConfig) -> None:
        self.config = config
        self.preprocessor = ReceiptPreprocessor(use_perspective_fix=config.use_perspective_fix)
        self._backend = None

    def recognize(self, file_path: str | Path) -> OCRResult:
        image = self.preprocessor.load_image(file_path)
        preprocessed = self.preprocessor.preprocess(image)
        backend = self._get_backend()
        return backend(preprocessed.image, preprocessed.metadata)

    def _get_backend(self):
        if self._backend is not None:
            return self._backend
        if self.config.engine in {"auto", "tesseract"}:
            try:
                import pytesseract
                self._backend = self._run_tesseract
                return self._backend
            except Exception:
                if self.config.engine == "tesseract":
                    raise
        if self.config.engine in {"auto", "easyocr"}:
            try:
                import easyocr  # noqa: F401
                self._backend = self._run_easyocr
                return self._backend
            except Exception:
                if self.config.engine == "easyocr":
                    raise
        raise RuntimeError(
            "No OCR backend available. Install pytesseract+tesseract or easyocr."
        )

    def _run_tesseract(self, image, metadata: dict[str, Any]) -> OCRResult:
        import pytesseract
        from pytesseract import Output

        config = f"--oem {self.config.oem} --psm {self.config.psm}"
        lang = "+".join(self.config.languages)
        data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=Output.DICT)
        words: list[OCRWord] = []
        lines_map: dict[int, list[str]] = {}
        confidences: list[float] = []
        for i, text in enumerate(data["text"]):
            text = (text or "").strip()
            conf_raw = data["conf"][i]
            try:
                conf = max(float(conf_raw), 0.0) / 100.0
            except Exception:
                conf = 0.0
            if not text:
                continue
            line_num = int(data["line_num"][i]) + 1000 * int(data["block_num"][i])
            if conf >= self.config.min_word_confidence:
                words.append(
                    OCRWord(
                        text=text,
                        confidence=conf,
                        bbox=(
                            int(data["left"][i]),
                            int(data["top"][i]),
                            int(data["width"][i]),
                            int(data["height"][i]),
                        ),
                        line_num=line_num,
                    )
                )
                lines_map.setdefault(line_num, []).append(text)
                confidences.append(conf)
        lines = [" ".join(parts).strip() for _, parts in sorted(lines_map.items()) if parts]
        return OCRResult(
            text="\n".join(lines),
            lines=lines,
            words=words,
            mean_confidence=sum(confidences) / len(confidences) if confidences else 0.0,
            metadata=metadata,
        )

    def _run_easyocr(self, image, metadata: dict[str, Any]) -> OCRResult:
        import easyocr
        langs = ["ru", "en"] if set(self.config.languages) >= {"rus", "eng"} else ["en"]
        reader = easyocr.Reader(langs, gpu=False)
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image
        raw = reader.readtext(rgb, detail=1, paragraph=False)
        words: list[OCRWord] = []
        confidences: list[float] = []
        lines: list[str] = []
        for idx, item in enumerate(raw):
            bbox, text, conf = item
            text = text.strip()
            if not text:
                continue
            xs = [int(p[0]) for p in bbox]
            ys = [int(p[1]) for p in bbox]
            words.append(
                OCRWord(
                    text=text,
                    confidence=float(conf),
                    bbox=(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)),
                    line_num=idx,
                )
            )
            lines.append(text)
            confidences.append(float(conf))
        return OCRResult(
            text="\n".join(lines),
            lines=lines,
            words=words,
            mean_confidence=sum(confidences) / len(confidences) if confidences else 0.0,
            metadata=metadata,
        )
