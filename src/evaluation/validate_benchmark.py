from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

BENCHMARK_PATH = Path('data/processed/benchmark_banking_support.jsonl')

class BenchmarkExample(BaseModel):
    id: str = Field(min_length=3)
    category: str = Field(min_length=2)
    input: str = Field(min_length=5)
    expected_keywords: list[str] = Field(min_length=1)
    must_not_contain: list[str] = Field(default_factory=list)
    ideal_behavior: str = Field(min_length=10)


def load_json(path: Path) -> list[dict[str, Any]]:
    examples = []

    if not path.exists():
        raise FileNotFoundError(f'Dataset not found at {path}')

    with path.open('r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f'Invalid JSON on line {line_num}: {e}') from e

    return examples


def validate_benchmark(path: Path) -> None:
    raw_examples = load_json(path)
    ids = set()
    validated = []

    for index, item in enumerate(raw_examples, 1):
        try:
            example = BenchmarkExample.model_validate(item)
        except ValidationError as e:
            raise ValueError(f'Invalid schema on line {index}: {e}') from e

        if example.id in ids:
            raise ValueError(f'Duplicate ID found on line {index}: {example.id}')
        
        ids.add(example.id)
        validated.append(example)
    categories = {}

    for example in validated:
        categories[example.category] = categories.get(example.category, 0) + 1

    print("Benchmark validation passed.")
    print(f"Total examples: {len(validated)}")
    print("Category distribution:")

    for category, count in sorted(categories.items()):
        print(f"  - {category}: {count}")


if __name__ == '__main__':
    validate_benchmark(BENCHMARK_PATH)