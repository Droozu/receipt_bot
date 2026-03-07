from __future__ import annotations

import json
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class DatasetStore:
    def __init__(self, samples_dir: Path, corrections_dir: Path) -> None:
        self.samples_dir = samples_dir
        self.corrections_dir = corrections_dir
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.corrections_dir.mkdir(parents=True, exist_ok=True)

    def save_sample(
        self,
        image_path: str | Path,
        ocr_text: str,
        parsed_data: Any,
        confidence: float,
    ) -> str:
        sample_id = uuid4().hex
        sample_dir = self.samples_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        image_path = Path(image_path)
        shutil.copy2(image_path, sample_dir / image_path.name)
        payload = {
            "id": sample_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "image_name": image_path.name,
            "ocr_text": ocr_text,
            "parsed_data": self._serialize(parsed_data),
            "confidence": confidence,
        }
        (sample_dir / "sample.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return sample_id

    def save_correction(
        self,
        sample_id: str,
        corrected_data: dict[str, Any],
        notes: str | None = None,
    ) -> Path:
        path = self.corrections_dir / f"{sample_id}.json"
        payload = {
            "sample_id": sample_id,
            "corrected_data": corrected_data,
            "notes": notes,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def iter_samples(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for path in self.samples_dir.glob("*/sample.json"):
            samples.append(json.loads(path.read_text(encoding="utf-8")))
        return sorted(samples, key=lambda x: x["created_at"])

    def iter_corrections(self) -> list[dict[str, Any]]:
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(self.corrections_dir.glob("*.json"))]

    def _serialize(self, value: Any) -> Any:
        if is_dataclass(value):
            return asdict(value)
        return value
