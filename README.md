# Receipt Bot

Self-learning receipt parser for noisy paper receipt photos.

## Features

- OCR via Tesseract or EasyOCR
- Receipt preprocessing: grayscale, denoise, threshold, deskew, crop, perspective fix
- Heuristic parser for store name, legal name, date/time, total and item table
- Confidence score
- Automatic low-confidence sample storage
- User feedback loop with local correction dataset
- Trainer that builds correction memory, store aliases and domain dictionary
- JSON + CSV export

## Project structure

```text
receipt_bot/
  app/
    main.py
    config.py
    ocr/
      engine.py
      preprocess.py
    parser/
      receipt_parser.py
      patterns.py
      items_parser.py
    learning/
      trainer.py
      dataset.py
      feedback.py
    storage/
      samples/
      corrections/
      models/
    export/
      csv_export.py
      json_export.py
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install Tesseract OCR engine on the host OS and ensure it is available in `PATH`.

## Usage

### Parse receipt

```bash
PYTHONPATH=. python -m app.main parse /path/to/receipt.jpg --output-dir outputs
```

### Submit correction

```bash
PYTHONPATH=. python -m app.main feedback <sample_id> corrected.json --notes "fixed item names"
```

### Retrain patterns

```bash
PYTHONPATH=. python -m app.main train
```

## Example correction JSON

```json
{
  "store_name": "Пятёрочка",
  "legal_name": "ООО \"Агроторг\"",
  "datetime": "2018-12-02T17:40",
  "items": [
    {"name": "К.Ц. Молоко ультрапастеризованное 1.5% 970мл", "quantity": 1, "price": 49.99},
    {"name": "Напиток яблоко-вишня-черноплодная рябина 1.93л", "quantity": 1, "price": 82.99}
  ]
}
```
