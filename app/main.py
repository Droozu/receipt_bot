from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from app.config import DEFAULT_CONFIG
from app.export.csv_export import export_items_csv
from app.export.json_export import export_json
from app.learning.dataset import DatasetStore
from app.learning.feedback import FeedbackManager
from app.learning.trainer import PatternTrainer
from app.ocr.engine import OCREngine
from app.parser.receipt_parser import ReceiptParser


def build_services():
    config = DEFAULT_CONFIG
    dataset = DatasetStore(config.storage.samples_dir, config.storage.corrections_dir)
    trainer = PatternTrainer(dataset, config.storage.models_dir / "patterns.json")
    parser = ReceiptParser(config.parser, config.storage.models_dir / "patterns.json")
    ocr = OCREngine(config.ocr)
    feedback = FeedbackManager(dataset)
    return config, dataset, trainer, parser, ocr, feedback

def command_save(args: argparse.Namespace) -> int:
    config, dataset, _trainer, parser, ocr, _feedback = build_services()
    ocr_result = ocr.recognize(args.input)
    receipt = parser.parse(ocr_result)
    sample_id = dataset.save_sample(args.input, ocr_result.text, receipt, receipt.confidence)
    print(f"Sample saved: {sample_id}")
    return 0

def command_parse(args: argparse.Namespace) -> int:
    config, dataset, _trainer, parser, ocr, _feedback = build_services()
    ocr_result = ocr.recognize(args.input)
    receipt = parser.parse(ocr_result)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem
    json_path = export_json(asdict(receipt), out_dir / f"{stem}.json")
    csv_path = export_items_csv(receipt.items, out_dir / f"{stem}.csv")
    print(json.dumps(asdict(receipt), ensure_ascii=False, indent=2))
    print(f"JSON saved to: {json_path}")
    print(f"CSV saved to:  {csv_path}")
    if receipt.confidence < config.parser.low_confidence_threshold:
        sample_id = dataset.save_sample(args.input, ocr_result.text, receipt, receipt.confidence)
        print(f"Low confidence sample saved: {sample_id}")
    return 0

def command_feedback(args: argparse.Namespace) -> int:
    _config, _dataset, _trainer, _parser, _ocr, feedback = build_services()
    corrected = json.loads(Path(args.corrected_json).read_text(encoding="utf-8"))
    path = feedback.submit(args.sample_id, corrected, notes=args.notes)
    print(f"Correction saved to: {path}")
    return 0

def command_train(_: argparse.Namespace) -> int:
    _config, _dataset, trainer, _parser, _ocr, _feedback = build_services()
    learned = trainer.train()
    print("Training completed.")
    print(f"Dictionary size: {len(learned.dictionary)}")
    print(f"Corrections size: {len(learned.corrections)}")
    print(f"Store aliases: {len(learned.store_aliases)}")
    return 0

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Self-learning receipt parser bot")
    sub = parser.add_subparsers(dest="command", required=True)

    parse_cmd = sub.add_parser("parse", help="Parse receipt image")
    parse_cmd.add_argument("input", help="Path to receipt image")
    parse_cmd.add_argument("--output-dir", default="outputs", help="Directory for JSON and CSV outputs")
    parse_cmd.set_defaults(func=command_parse)

    save_cmd = sub.add_parser("save", help="Save receipt data")
    save_cmd.add_argument("input", help="Path to receipt image")
    save_cmd.set_defaults(func=command_save)

    feedback_cmd = sub.add_parser("feedback", help="Submit corrected data for low-confidence sample")
    feedback_cmd.add_argument("sample_id", help="Sample ID returned by parse command")
    feedback_cmd.add_argument("corrected_json", help="Path to corrected JSON")
    feedback_cmd.add_argument("--notes", default=None, help="Optional note")
    feedback_cmd.set_defaults(func=command_feedback)

    train_cmd = sub.add_parser("train", help="Retrain learned patterns from dataset")
    train_cmd.set_defaults(func=command_train)
    return parser

def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
