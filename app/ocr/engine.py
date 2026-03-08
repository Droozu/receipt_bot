from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


from app.ocr.engines.paddleocr_engine import PaddleOcrEngine
from app.ocr.engines.tesseract_engine import TesseractEngine
from app.ocr.engines.easyocr_engine import EasyOCREngine
from app.ocr.engines.engines_class import OCRResult

from app.config import OCRConfig
from app.ocr.preprocess import ReceiptPreprocessor

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
        if self.config.engine in {"auto", "paddleocr"}:
            try:
                PaddleOcr = PaddleOcrEngine(self.config)
                self._backend = PaddleOcr.recognize
                print("Using PaddleOCR backend")
                return self._backend
            except Exception:
                if self.config.engine == "paddleocr":
                    raise
        if self.config.engine in {"auto", "tesseract"}:
            try:
                Tesseract = TesseractEngine(self.config)
                self._backend = Tesseract.recognize
                print("Using Tesseract backend")
                return self._backend
            except Exception:
                if self.config.engine == "tesseract":
                    raise
        if self.config.engine in {"auto", "easyocr"}:
            try:
                Easyocr = EasyOCREngine(self.config)
                self._backend = Easyocr.recognize
                print("Using EasyOCR backend")
                return self._backend
            except Exception:
                if self.config.engine == "easyocr":
                    raise
        raise RuntimeError(
            "No OCR backend available. Install pytesseract+tesseract or easyocr."
        )

   



