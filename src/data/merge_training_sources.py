from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


SOURCE_PATHS = [
    Path("data/processed/custom_aligned_banking_v3.jsonl"),
    Path("data/processed/custom_fraud_v5.jsonl"),
]

OUTPUT_PATH = Path("data/processed/banking_support_merged.jsonl")

RANDOM_SEED = 42


REQUIRED_FIELDS = [
    "instruction",
    "input",
    "output",
    "category",
    "language",
    "source",
    "intent",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {error}") from error

            rows.append(item)

    return rows


def validate_item(item: dict[str, Any], source_path: Path) -> None:
    for field in REQUIRED_FIELDS:
        if field not in item:
            raise ValueError(f"Missing field '{field}' in {source_path}: {item}")

        if not str(item[field]).strip():
            raise ValueError(f"Empty field '{field}' in {source_path}: {item}")


def main() -> None:
    merged: list[dict[str, Any]] = []
    seen_inputs: set[str] = set()

    for source_path in SOURCE_PATHS:
        rows = load_jsonl(source_path)
        print(f"Loaded {len(rows)} rows from {source_path}")

        for item in rows:
            validate_item(item, source_path)

            normalized_input = item["input"].lower().strip()

            if normalized_input in seen_inputs:
                continue

            seen_inputs.add(normalized_input)
            merged.append(item)

    random.seed(RANDOM_SEED)
    random.shuffle(merged)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for item in merged:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Merged examples: {len(merged)}")
    print(f"Output path: {OUTPUT_PATH}")

    source_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    language_counts: dict[str, int] = {}

    for item in merged:
        source_counts[item["source"]] = source_counts.get(item["source"], 0) + 1
        category_counts[item["category"]] = category_counts.get(item["category"], 0) + 1
        language_counts[item["language"]] = language_counts.get(item["language"], 0) + 1

    print("\nSource distribution:")
    for source, count in sorted(source_counts.items()):
        print(f"  - {source}: {count}")

    print("\nLanguage distribution:")
    for language, count in sorted(language_counts.items()):
        print(f"  - {language}: {count}")

    print("\nCategory distribution:")
    for category, count in sorted(category_counts.items()):
        print(f"  - {category}: {count}")


if __name__ == "__main__":
    main()
