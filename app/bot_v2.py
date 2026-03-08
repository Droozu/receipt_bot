from __future__ import annotations

import base64
import csv
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import cv2
import numpy as np
import requests
from openai import OpenAI
from paddleocr import PaddleOCR
from PIL import Image
from dotenv import load_dotenv
from app.parser.pattern_apply import PatternApplier
from app.learning.dataset import DatasetStore
from app.learning.trainer import PatternTrainer
from app.parser.patterns import LearnedPatterns
from app.config import DEFAULT_CONFIG

load_dotenv()
# =========================
# Config
# =========================

config = DEFAULT_CONFIG

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

DATA_DIR = config.storage.base_dir
SAMPLES_DIR = config.storage.samples_dir
CORRECTIONS_DIR = config.storage.corrections_dir
MODELS_DIR = config.storage.models_dir
OUTPUT_DIR = Path("output")

LOW_CONFIDENCE_THRESHOLD = 0.7


# =========================
# Data models
# =========================

@dataclass
class Item:
    name: str
    quantity: float | None
    price: float | None


@dataclass
class ReceiptResult:
    store_name: str | None
    legal_name: str | None
    datetime: str | None
    items: list[Item]
    total: float | None
    inn: str | None
    ocr_text: str
    confidence: float
    image_source: str


# =========================
# Utils
# =========================

def ensure_dirs() -> None:
    for p in [DATA_DIR, SAMPLES_DIR, CORRECTIONS_DIR, MODELS_DIR, OUTPUT_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    text = re.sub(r"[^0-9.\-]", "", text)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def clean_text(text: str | None) -> str | None:
    if text is None:
        return None
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


# =========================
# Downloader
# =========================

def download_or_copy_input(src: str) -> Path:
    temp_dir = OUTPUT_DIR / "_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    if is_url(src):
        resp = requests.get(src, timeout=60)
        resp.raise_for_status()
        ext = ".jpg"
        content_type = resp.headers.get("Content-Type", "").lower()
        if "png" in content_type:
            ext = ".png"
        elif "jpeg" in content_type or "jpg" in content_type:
            ext = ".jpg"

        out = temp_dir / f"input_{uuid.uuid4().hex}{ext}"
        out.write_bytes(resp.content)
        return out

    path = Path(src)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return path


# =========================a
# Preprocess
# =========================

# =========================
# Learning memory
# =========================

def load_corrections() -> dict[str, str]:
    return load_json(MODELS_DIR / "ocr_corrections.json", {})


def apply_corrections(text: str) -> str:
    corrections = load_corrections()
    for wrong, right in corrections.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", right, text, flags=re.IGNORECASE)
    return text


def add_feedback(sample_id: str, corrected_json_path: str) -> None:
    corrected = json.loads(Path(corrected_json_path).read_text(encoding="utf-8"))
    sample_file = SAMPLES_DIR/f"{sample_id}.json"
    if not sample_file.exists():
        raise FileNotFoundError(f"Sample not found: {sample_file}")

    sample = load_json(sample_file, {})
    sample["corrected_data"] = corrected
    save_json(CORRECTIONS_DIR / f"{sample_id}.json", sample)
    print(f"Feedback saved: {CORRECTIONS_DIR / f'{sample_id}.json'}")


def train_from_feedback() -> None:
    corrections_map = load_corrections()

    for file in CORRECTIONS_DIR.glob("*.json"):
        payload = load_json(file, {})
        ocr_text = payload.get("ocr_text", "")
        corrected = payload.get("corrected_data", {})
        items = corrected.get("items", []) or []

        ocr_tokens = set(re.findall(r"[A-Za-zА-Яа-я0-9%./-]{3,}", ocr_text))
        corrected_tokens = set()

        for item in items:
            corrected_tokens.update(re.findall(r"[A-Za-zА-Яа-я0-9%./-]{3,}", item.get("name", "")))

        # very simple local memory:
        # if OCR token and corrected token differ by 1-2 chars, remember it
        for o in ocr_tokens:
            for c in corrected_tokens:
                if o.lower() == c.lower():
                    continue
                if levenshtein(o.lower(), c.lower()) <= 2:
                    corrections_map[o] = c

    save_json(MODELS_DIR / "ocr_corrections.json", corrections_map)
    print(f"Training complete. Learned corrections: {len(corrections_map)}")


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = curr[j - 1] + 1
            delete_cost = prev[j] + 1
            replace_cost = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(insert_cost, delete_cost, replace_cost))
        prev = curr
    return prev[-1]


# =========================
# OCR
# =========================

class OCRService:

    def __init__(self) -> None:

        self.ocr = PaddleOCR(
            lang="ru",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_side_len=1280,
        )

    def recognize(self, image_path: Path) -> tuple[str, float]:

        raw = self.ocr.predict(str(image_path))

        from pprint import pformat

        Path("output").mkdir(exist_ok=True)

        (Path("output") / "ocr_raw_debug.txt").write_text(
            pformat(raw, width=120, compact=False),
            encoding="utf-8"
        )

        lines: list[str] = []
        confs: list[float] = []

        # === PaddleOCR v3 result ===
        # raw -> list -> dict -> rec_texts / rec_scores

        if isinstance(raw, list):

            for page in raw:

                if not isinstance(page, dict):
                    continue

                rec_texts = page.get("rec_texts", [])
                rec_scores = page.get("rec_scores", [])

                for i, text in enumerate(rec_texts):

                    text = clean_text(text)

                    if not text:
                        continue

                    lines.append(text)

                    if i < len(rec_scores):
                        confs.append(float(rec_scores[i]))
                    else:
                        confs.append(0.0)

        full_text = "\n".join(lines)

        full_text = apply_corrections(full_text)

        if not full_text.strip():

            raise ValueError(
                "OCR returned empty text. Check output/ocr_raw_debug.txt and preprocessed image."
            )

        confidence = float(sum(confs) / len(confs)) if confs else 0.0

        return full_text, confidence


# =========================
# DeepSeek parser
# =========================

class DeepSeekReceiptParser:
    def __init__(self, api_key: str) -> None:
        if not api_key.strip():
            raise ValueError("DEEPSEEK_API_KEY is empty")

        self.client = OpenAI(
            api_key=api_key,
            base_url=DEEPSEEK_BASE_URL,
        )

    def parse(self, ocr_text: str) -> dict[str, Any]:
        system_prompt = """
You are a receipt parser.
Return only valid JSON.
The word json is required in this prompt intentionally.

Extract:
- store_name
- legal_name
- datetime
- inn
- items: [{name, quantity, price}]
- total
- confidence_reason

Rules:
- Use null if field is missing.
- quantity and price must be numbers when possible.
- Keep item names in original language.
- Ignore payment terminal noise if not relevant.
- Prefer receipt line items over summary lines.
- Output only JSON.
"""

        user_prompt = f"""
Parse this OCR receipt text into json.

OCR TEXT:
{ocr_text}

JSON SCHEMA:
{{
  "store_name": null,
  "legal_name": null,
  "datetime": null,
  "inn": null,
  "items": [
    {{
      "name": "",
      "quantity": null,
      "price": null
    }}
  ],
  "total": null,
  "confidence_reason": ""
}}
"""

        response = self.client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=2000,
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("DeepSeek returned empty response")

        return json.loads(content)


# =========================
# Result normalization
# =========================

def normalize_items(items: list[dict[str, Any]]) -> list[Item]:
    out: list[Item] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        name = clean_text(str(item.get("name", "")))
        if not name:
            continue

        out.append(
            Item(
                name=name,
                quantity=to_float(item.get("quantity")),
                price=to_float(item.get("price")),
            )
        )
    return out


def calculate_confidence(ocr_conf: float, parsed: dict[str, Any]) -> float:
    score = 0.0
    score += min(max(ocr_conf, 0.0), 1.0) * 0.45

    if parsed.get("store_name"):
        score += 0.10
    if parsed.get("datetime"):
        score += 0.10
    if parsed.get("total") is not None:
        score += 0.10

    items = parsed.get("items") or []
    if items:
        score += 0.15

        complete_items = 0
        for item in items:
            if item.get("name") and item.get("price") is not None:
                complete_items += 1

        if items:
            score += 0.10 * (complete_items / len(items))

    return round(min(score, 1.0), 4)

def apply_dictionary(text: str):

    import json
    from pathlib import Path

    path = Path("app/storage/models/dictionary.json")

    if not path.exists():
        return text

    data = json.loads(path.read_text(encoding="utf-8"))

    for wrong, right in data.items():
        text = text.replace(wrong, right)

    return text

# =========================
# Export
# =========================

def save_receipt_outputs(result: ReceiptResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "store_name": result.store_name,
        "legal_name": result.legal_name,
        "datetime": result.datetime,
        "inn": result.inn,
        "items": [asdict(item) for item in result.items],
        "total": result.total,
        "ocr_text": result.ocr_text,
        "confidence": result.confidence,
        "image_source": result.image_source,
    }

    save_json(output_dir / "receipt.json", json_payload)

    with (output_dir / "receipt.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "quantity", "price"])
        for item in result.items:
            writer.writerow([item.name, item.quantity, item.price])

    (output_dir / "ocr.txt").write_text(result.ocr_text, encoding="utf-8")


# =========================
# Samples
# =========================

def save_low_confidence_sample(
    image_source: str,
    ocr_text: str,
    parsed_json: dict[str, Any],
    confidence: float,
) -> str:
    sample_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    payload = {
        "sample_id": sample_id,
        "image_source": image_source,
        "ocr_text": ocr_text,
        "parsed_result": parsed_json,
        "confidence": confidence,
        "created_at": datetime.now().isoformat(),
    }
    save_json(SAMPLES_DIR / f"{sample_id}.json", payload)
    return sample_id


# =========================
# Main pipeline
# =========================

def process_receipt(input_value: str) -> ReceiptResult:
    ensure_dirs()

    input_path = download_or_copy_input(input_value)
    preprocessed_path = preprocess_image(input_path)

    # OCR
    ocr_service = OCRService()
    ocr_text, ocr_conf = ocr_service.recognize(preprocessed_path)

    # LOAD PATTERNS
    applier = PatternApplier(Path("app/storage/models/patterns.json"))

    # APPLY PATTERNS TO OCR
    ocr_text = applier.fix_text(ocr_text)

    # dictionary
    ocr_text = apply_dictionary(ocr_text)

    # LLM parser
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    parser = DeepSeekReceiptParser(api_key=api_key)

    parsed_json = parser.parse(ocr_text)
    parsed_json["store_name"] = applier.fix_store(parsed_json.get("store_name"))

    for item in parsed_json.get("items", []):
        item["name"] = applier.fix_text(item["name"])

    # APPLY PATTERNS TO RESULT
    parsed_json["store_name"] = applier.fix_store(
        parsed_json.get("store_name")
    )

    for item in parsed_json.get("items", []):
        item["name"] = applier.fix_text(item.get("name", ""))

    items = normalize_items(parsed_json.get("items", []))

    confidence = calculate_confidence(ocr_conf, parsed_json)

    result = ReceiptResult(
        store_name=clean_text(parsed_json.get("store_name")),
        legal_name=clean_text(parsed_json.get("legal_name")),
        datetime=clean_text(parsed_json.get("datetime")),
        items=items,
        total=to_float(parsed_json.get("total")),
        inn=clean_text(parsed_json.get("inn")),
        ocr_text=ocr_text,
        confidence=confidence,
        image_source=input_value,
    )

    save_receipt_outputs(result, OUTPUT_DIR)

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        sample_id = save_low_confidence_sample(
            image_source=input_value,
            ocr_text=ocr_text,
            parsed_json=parsed_json,
            confidence=confidence,
        )
        print(f"[LOW CONFIDENCE] sample saved: {sample_id}")

    return result


# =========================
# CLI
# =========================

def print_usage() -> None:
    print(
        "Usage:\n"
        "  python bot_v2.py parse <image_path_or_url>\n"
        "  python bot_v2.py feedback <sample_id> <corrected_json_path>\n"
        "  python bot_v2.py train\n"
    )


def main() -> None:
    if len(sys.argv) < 2:
        print_usage()
        raise SystemExit(1)

    cmd = sys.argv[1].lower()

    if cmd == "parse":
        if len(sys.argv) < 3:
            print_usage()
            raise SystemExit(1)

        result = process_receipt(sys.argv[2])
        print(json.dumps({
            "store_name": result.store_name,
            "legal_name": result.legal_name,
            "datetime": result.datetime,
            "inn": result.inn,
            "items": [asdict(i) for i in result.items],
            "total": result.total,
            "confidence": result.confidence,
        }, ensure_ascii=False, indent=2))
        print("Saved: output/receipt.json")
        print("Saved: output/receipt.csv")
        print("Saved: output/ocr.txt")
        return

    if cmd == "feedback":
        if len(sys.argv) < 4:
            print_usage()
            raise SystemExit(1)
        add_feedback(sys.argv[2], sys.argv[3])
        return

    if cmd == "train":
        dataset = DatasetStore(
            Path("app/storage/samples"),
            Path("app/storage/corrections"),
            )
        trainer = PatternTrainer(
            dataset,
            Path("app/storage/models/patterns.json"),
        )
        trainer.train()
        print(f"Training complete. Learned corrections: {len(load_corrections())}")


        return

    print_usage()
    raise SystemExit(1)


if __name__ == "__main__":
    main()