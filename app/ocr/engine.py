from __future__ import annotations

from pathlib import Path


from app.ocr.engines.engines_class import OCRResult

from app.config import OCRConfig
from app.ocr.preprocess import ReceiptPreprocessor
from app.utils.utils import try_import

class OCREngine:
    def __init__(self, config: OCRConfig) -> None:

        self.config = config
        self.preprocessor = ReceiptPreprocessor(use_perspective_fix=config.use_perspective_fix)
        self._backend = []

    def recognize(self, file_path: str | Path) -> OCRResult:

        image = self.preprocessor.load_image(file_path)
        preprocessed = self.preprocessor.preprocess(image)

        backend = self._get_backend()

        return backend(preprocessed.image, preprocessed.metadata)

    def _get_backend(self):

        if self.config.engine in {"auto", "paddleocr"}:
            print("Try to using PaddleOCR backend")

            PaddleOcr = try_import(
                "app.ocr.engines.paddleocr_engine", 
                "PaddleOcrEngine"
                )
                
            if PaddleOcr:
                self._backend.appends = (("PaddleOcr", PaddleOcr))
                return self._backend
        

        if self.config.engine in {"auto", "tesseract"}:
            print("Try to using Tesseract backend")

            Tesseract = try_import(
                    "app.ocr.engines.tesseract_engine",
                    "TesseractEngine",
                    )
            
            if Tesseract:
                self._backend.append(("tesseract", Tesseract))
                return self._backend


        if self.config.engine in ("auto", "easyocr"):
            print("Try to using EasyOCR backend")

            Easyocr = try_import(
                "app.ocr.engines.easyocr_engine",
                "EasyOCREngine",
                )
            if Easyocr:
                self._backend.append(("easyocr", Easyocr))
                return self._backend
            
        for name, cls in self._backend:
            try:
                return cls(self.config)
            except Exception:
                continue
    
        raise RuntimeError(
            "No OCR backend available. Install pytesseract+tesseract or easyocr."
        )

   



