from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError


DATASET_PATH = Path('data/processed/sample_banking_support.jsonl')


class BankingExample(BaseModel):
    instruction: str = Field(min_length=10)
    input: str = Field(min_length=3)
    output: str = Field(min_length=10)
    category: str = Field(min_length=2)


def load_json(path: Path) -> list[dict[str, Any]]:
    ''' Loading the dataset '''
    examples: list[dict[str, Any]] = []

    if not path.exists():
        raise FileNotFoundError(f'Dataset not found at {path}')

    with path.open('r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            
            if not line:
                continue

            try:
                item = json.loads(line)

            except json.JSONDecodeError as e:
                raise ValueError(f'Invalid JSON on line {line_num}: {e}') from e

            examples.append(item)

    return examples


def validate_dataset(path: Path) -> None:
    raw_examples = load_json(path)
    validate_examples: list[BankingExample] = []
    seen_inputs: set[str] = set()


    for index, item in enumerate(raw_examples, 1):
        try:
            example = BankingExample.model_validate(item)
        
        except ValidationError as e:
            raise ValueError(f'Invalid schema on line {index}: {e}') from e

        normalized_input = example.input.strip().lower()

        if normalized_input in seen_inputs:
            raise ValueError(f'Duplicate input found on line {index}: {example.input}')
        
        seen_inputs.add(normalized_input)
        validate_examples.append(example)


    categories: dict[str, int] = {}

    for example in validate_examples:
        categories[example.category] = categories.get(example.category, 0) + 1


    print("Dataset validation passed.")
    print(f"Total examples: {len(validate_examples)}")
    print("Category distribution:")

    for category, count in sorted(categories.items()):
        print(f"  - {category}: {count}")


if __name__ == '__main__':
    validate_dataset(DATASET_PATH)