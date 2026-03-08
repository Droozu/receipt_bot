try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None
    
from app.ocr.engines.engines_class import OCRResult, OCRWord
from app.utils.utils import clean_text, apply_corrections
from typing import Any
from app.config import OCRConfig



class PaddleOcrEngine:

    def __init__(self, config: OCRConfig, ) -> None:
        if PaddleOCR is None:
            raise RuntimeError("paddleocr not installed")

        self.config = config

        self.ocr = PaddleOCR(
            lang="ru",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_side_len=1280,
        )

    def recognize(self, image, metadata: dict[str, Any]) -> OCRResult:
        import cv2

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        raw = self.ocr.predict(image)
        # from pprint import pformat
        # Path("output").mkdir(exist_ok=True)
        # (Path("output") / "ocr_raw_debug.txt").write_text(
        #     pformat(raw, width=120, compact=False),
        #     encoding="utf-8"
        # )

        lines: list[str] = []
        confs: list[float] = []
        words: list[OCRWord] = []

        line_index = 0

        # PaddleOCR v3 output structure
        # raw -> list[page]
        # page -> dict
        # dt_polys -> bounding boxes
        # rec_texts -> recognized text
        # rec_scores -> confidence

        if isinstance(raw, list):
            for page in raw:
                if not isinstance(page, dict):
                    continue

                texts = page.get("rec_texts", [])
                scores = page.get("rec_scores", [])
                polys = page.get("dt_polys", [])

                for i, text in enumerate(texts):
                    text = clean_text(text)
                    if not text:
                        continue
                    conf = float(scores[i]) if i < len(scores) else 0.0

                    # bbox from polygon
                    if i < len(polys):
                        poly = polys[i]
                        xs = [int(p[0]) for p in poly]
                        ys = [int(p[1]) for p in poly]
                        x = min(xs)
                        y = min(ys)
                        w = max(xs) - x
                        h = max(ys) - y
                    else:
                        x = y = w = h = 0
                    words.append(
                        OCRWord(
                            text=text,
                            confidence=conf,
                            bbox=(x, y, w, h),
                            line_num=line_index,
                        )
                    )
                    lines.append(text)
                    confs.append(conf)
                    line_index += 1

            full_text = "\n".join(lines)
            full_text = apply_corrections(full_text)

            if not full_text.strip():
                raise ValueError(
                    "OCR returned empty text. Check output/ocr_raw_debug.txt and preprocessed image."
                )

            confidences = float(sum(confs) / len(confs)) if confs else 0.0

        return OCRResult(
        text=full_text,
        lines=lines,
        words=words,
        mean_confidence=confidences,
        metadata=metadata,
        )