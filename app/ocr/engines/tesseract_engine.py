try:
    from pytesseract import Output, image_to_data
except Exception:
    pytesseract = None

from app.ocr.engines.engines_class import OCRResult, OCRWord 

class TesseractEngine:
    def __init__(self, config):
        if pytesseract is None:
            raise RuntimeError("pytesseract not installed")
        
        self.config = config

    def recognize(self, image, metadata):
        config = f"--oem {self.config.oem} --psm {self.config.psm}"
        lang = "+".join(self.config.languages)

        data = image_to_data(
            image,
            lang=lang,
            config=config,
            output_type=Output.DICT,
        )

        words = []
        lines_map = {}
        confidences = []

        for i, text in enumerate(data["text"]):
            text = (text or "").strip()
            if not text:
                continue
            conf = max(float(data["conf"][i]), 0.0) / 100.0
            line_num = int(data["line_num"][i]) + 1000 * int(data["block_num"][i])
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
        lines = [" ".join(parts) for _, parts in sorted(lines_map.items())]
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return OCRResult(
            text="\n".join(lines),
            lines=lines,
            words=words,
            mean_confidence=mean_conf,
            metadata=metadata,
        )