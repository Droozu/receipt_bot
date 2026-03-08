import sys
sys.path.insert(0, ".")

from app.config import DEFAULT_CONFIG
from app.parser.patterns import LEGAL_ENTITY_PATTERNS
from app.parser.receipt_parser import ReceiptParser

config = DEFAULT_CONFIG

raw_text = """4 ..
ПЯТЁРОЧКА
KNCCOBbIN ЧЕК
Цена со
Цена Скидка скидкой Кол-во Итого НАС
2111189 К.Ц.Нол.ул/паст. 1,5% 970мл
49.99 49.99 * 1 49.99 6
43467778 ЛЮБ.Напит. ябл/ВИШНЯ/чер
82.39 82.99 4 1 82.99 Б
3648084 Томат СЛИВОВИДНЫЙ 600г
139.99 _ 139.99 4 1 139.99 Б
^3498026 ЛЮБ.Нап.ЯПЕЛ.МАНГО дет.мяк1,93л
82.99 82.39 * 1 82.99 6
3442578 PICNIC Батончик BIG 76r
. 42.80 42.80 * 1 42.80 A
3442578 PICNIC Батончик BIC 76г
42.80 42.80 4 1 42.80 A
3277399 PABA Найон.ПРОВАНС.67Х ana 400Г ;
64.00 64.00 * 1 64.00 A
2111189 К.Ц.Мол.ул/ПАСТ.,5% 970мл
CKHAKA: 0.00 ПОДЫТОГ: 555.55
ОКРУГ ЛЕНИЕ: 0.00 ИТОГ: 555.55
000 "Агроторг"
ИНН: 7825706086 = СНО:ОСН _ Код: 0085
184430 Печенгский р-н, Г. Заполярный, ул. Ленина, д
Кассир: Секерина Екатерина 02.12.18 17:40"""

lines = raw_text.strip().split("\n")

parser = ReceiptParser(config.parser, config.storage.models_dir / "patterns.json")

corrected_lines = [parser._apply_corrections(line) for line in lines]

print("=== Строки после _apply_corrections ===")
for i, line in enumerate(corrected_lines):
    print(f"  [{i:02d}] {line!r}")

print("\n=== Проверяем каждую строку паттернами ===")
for i, line in enumerate(corrected_lines):
    for pattern in LEGAL_ENTITY_PATTERNS:
        m = pattern.search(line)
        if m:
            print(f"  СОВПАЛО [{i:02d}]: {line!r} -> {m.group(0)!r}")

print("\n=== Результат _extract_legal_name ===")
result = parser._extract_legal_name(corrected_lines)
print(f"  legal_name = {result!r}")