from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Any


def export_items_csv(items: Iterable[Mapping[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["name", "quantity", "price"])
        for item in items:
            writer.writerow([item.get("name"), item.get("quantity"), item.get("price")])
    return path
