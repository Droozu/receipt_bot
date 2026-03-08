from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from app.learning.dataset import DatasetStore
from app.parser.patterns import LearnedPatterns


class PatternTrainer:
    def __init__(self, dataset: DatasetStore, model_path: Path) -> None:
        self.dataset = dataset
        self.model_path = model_path

    def train(self) -> LearnedPatterns:
        learned = LearnedPatterns.load(self.model_path)
        samples_by_id = {sample["id"]: sample for sample in self.dataset.iter_samples()}
        correction_votes: dict[str, Counter[str]] = defaultdict(Counter)
        store_alias_votes: dict[str, Counter[str]] = defaultdict(Counter)
        abbreviation_votes: dict[str, Counter[str]] = defaultdict(Counter)
        dictionary = set(learned.dictionary)

        for correction in self.dataset.iter_corrections():
            sample = samples_by_id.get(correction["sample_id"])
            if not sample:
                continue
            corrected = correction["corrected_data"]
            parsed = sample.get("parsed_data", {})
            sample_store = (parsed.get("store_name") or "").strip().upper()
            corrected_store = (corrected.get("store_name") or "").strip()
            if sample_store and corrected_store:
                store_alias_votes[sample_store][corrected_store] += 1
            self._learn_token_corrections(
                sample.get("ocr_text", ""),
                json.dumps(corrected, ensure_ascii=False),
                correction_votes,
            )
            for item in corrected.get("items", []):
                for token in self._tokenize(item.get("name", "")):
                    dictionary.add(token.upper())
            for token in self._tokenize(corrected_store):
                dictionary.add(token.upper())
            self._learn_abbreviations(
                sample.get("ocr_text", ""),
                corrected,
                abbreviation_votes,  # передаём как dict
            )
        learned.dictionary = dictionary
        for wrong, votes in correction_votes.items():
            learned.corrections[wrong] = votes.most_common(1)[0][0]
        for wrong_store, votes in store_alias_votes.items():
            learned.store_aliases[wrong_store] = votes.most_common(1)[0][0]
        for abbr, votes in abbreviation_votes.items():
            learned.abbreviations[abbr] = votes.most_common(1)[0][0]
        learned.save(self.model_path)
        return learned


    def _learn_token_corrections(self, ocr_text: str, corrected_text: str, votes: dict) -> None:
        ocr_tokens = [t.upper() for t in self._tokenize(ocr_text) if len(t) >= 4]
        corrected_tokens = [t.upper() for t in self._tokenize(corrected_text) if len(t) >= 4]
        corrected_set = set(corrected_tokens)
        for token in ocr_tokens:
            if token in corrected_set:
                continue
            # Пропускаем числовые токены — цены не исправляем через словарь
            if re.match(r"^\d+[.,]?\d*$", token):
                continue
            nearest = self._find_nearest(token, corrected_set)
            if nearest:
                # Также не добавляем если nearest — число
                if re.match(r"^\d+[.,]?\d*$", nearest):
                    continue
                votes[token][nearest] += 1

    def _find_nearest(self, token: str, candidates: set[str]) -> str | None:
        best = None
        best_score = 0.0
        for candidate in candidates:
            score = self._similarity(token, candidate)
            if score > best_score and score >= 0.72:
                best = candidate
                best_score = score
        return best

    def _similarity(self, a: str, b: str) -> float:
        import difflib

        return difflib.SequenceMatcher(None, a, b).ratio()

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[A-Za-zА-Яа-яЁё0-9%.-]+", text)

    def _learn_abbreviations(
        self,
        ocr_text: str,
        corrected_data: dict,
        abbreviations: dict[str, Counter],
    ) -> None:
        """
        Если OCR-токен выглядит как аббревиатура (содержит точки, короткий),
        и в corrected_data есть похожий полный токен — запоминаем пару.
        """
        ocr_tokens = self._tokenize(ocr_text)
        
        for item in corrected_data.get("items", []):
            full_name_tokens = self._tokenize(item.get("name", ""))
            
            for ocr_tok in ocr_tokens:
                # Аббревиатура: короткий токен с точкой или всё заглавными ≤ 6 букв
                is_abbr = (
                    "." in ocr_tok and len(ocr_tok) <= 8
                ) or (
                    ocr_tok.isupper() and len(ocr_tok) <= 5
                )
                if not is_abbr:
                    continue
                
                # Ищем полный токен, который начинается с тех же букв
                prefix = ocr_tok.rstrip(".").upper()
                for full_tok in full_name_tokens:
                    if (
                        len(prefix) >= 4                              # минимум 4 буквы в префиксе
                        and full_tok.upper().startswith(prefix[:4])   # сравниваем 4 буквы, не 3
                        and len(full_tok) >= len(prefix) * 2          # полное слово минимум вдвое длиннее
                        and full_tok.upper() != prefix
                    ):
                        abbreviations[ocr_tok.upper()][full_tok] += 1
                        break

