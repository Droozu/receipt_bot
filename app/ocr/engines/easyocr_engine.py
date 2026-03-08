try:
    import easyocr
except Exception:
    easyocr = None

import cv2
from app.ocr.engines.engines_class import OCRResult, OCRWord


class EasyOCREngine:
    def __init__(self, config):

        if easyocr is None:
            raise RuntimeError("easyocr not installed")
        
        langs = ["ru", "en"] if set(config.languages) >= {"rus", "eng"} else ["en"]
        self.reader = easyocr.Reader(langs, gpu=False)

    def recognize(self, image, metadata):
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image
        raw = self.reader.readtext(rgb, detail=1)

        words = []
        lines = []
        confs = []

        for idx, (bbox, text, conf) in enumerate(raw):
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
            confs.append(conf)

        mean_conf = sum(confs) / len(confs) if confs else 0.0

        return OCRResult(
            text="\n".join(lines),
            lines=lines,
            words=words,
            mean_confidence=mean_conf,
            metadata=metadata,
        )