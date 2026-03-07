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

        learned.dictionary = dictionary
        for wrong, votes in correction_votes.items():
            learned.corrections[wrong] = votes.most_common(1)[0][0]
        for wrong_store, votes in store_alias_votes.items():
            learned.store_aliases[wrong_store] = votes.most_common(1)[0][0]
        learned.save(self.model_path)
        return learned

    def _learn_token_corrections(self, ocr_text: str, corrected_text: str, votes: dict[str, Counter[str]]) -> None:
        ocr_tokens = [t.upper() for t in self._tokenize(ocr_text) if len(t) >= 4]
        corrected_tokens = [t.upper() for t in self._tokenize(corrected_text) if len(t) >= 4]
        corrected_set = set(corrected_tokens)
        for token in ocr_tokens:
            if token in corrected_set:
                continue
            nearest = self._find_nearest(token, corrected_set)
            if nearest:
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
