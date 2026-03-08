import json
import re

from app.config import DEFAULT_CONFIG
from pathlib import Path
from typing import Any

settings = DEFAULT_CONFIG

def clean_text(text: str | None) -> str | None:
    if text is None:
        return None
    text = re.sub(r"\s+", " ", text).strip()
    return text or None

def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def load_corrections() -> dict[str, str]:
    return load_json(settings.storage.models_dir / "ocr_corrections.json", {})

def apply_corrections(text: str) -> str:
    corrections = load_corrections()
    for wrong, right in corrections.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", right, text, flags=re.IGNORECASE)
    return text

def try_import(path, cls_name):
    try:
        module = __import__(path, fromlist=[cls_name])
        return getattr(module, cls_name)
    except Exception:
        return None