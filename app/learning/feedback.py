from __future__ import annotations

from pathlib import Path
from typing import Any

from app.learning.dataset import DatasetStore


class FeedbackManager:
    def __init__(self, dataset: DatasetStore) -> None:
        self.dataset = dataset

    def submit(self, sample_id: str, corrected_data: dict[str, Any], notes: str | None = None) -> Path:
        return self.dataset.save_correction(sample_id, corrected_data, notes)
