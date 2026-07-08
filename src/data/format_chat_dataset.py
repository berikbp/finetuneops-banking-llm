from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


INPUT_PATH = Path("data/processed/banking_support_merged.jsonl")
OUTPUT_DIR = Path("data/processed")

TRAIN_PATH = OUTPUT_DIR / "train.jsonl"
VAL_PATH = OUTPUT_DIR / "val.jsonl"
TEST_PATH = OUTPUT_DIR / "test.jsonl"

RANDOM_SEED = 42

SYSTEM_PROMPT = (
    "You are a professional banking AI assistant. "
    "Answer customer questions in a safe, clear, and concise way. "
    "Do not promise loan approval, do not invent bank policy, "
    "and never ask for PIN codes, passwords, or SMS codes."
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []

    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error

    return examples


def validate_raw_example(example: dict[str, Any], index: int) -> None:
    required_fields = [
        "instruction",
        "input",
        "output",
        "category",
        "language",
        "source",
        "intent",
    ]

    for field in required_fields:
        if field not in example:
            raise ValueError(f"Example {index} is missing required field: {field}")

        if not str(example[field]).strip():
            raise ValueError(f"Example {index} has empty field: {field}")


def to_chat_format(example: dict[str, Any]) -> dict[str, Any]:
    user_message = (
        f"{example['instruction']}\n\n"
        f"Customer question:\n{example['input']}"
    )

    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_message,
            },
            {
                "role": "assistant",
                "content": example["output"],
            },
        ],
        "category": example["category"],
        "language": example["language"],
        "source": example["source"],
        "intent": example["intent"],
    }


def save_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")


def split_examples(
    examples: list[dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not examples:
        raise ValueError("No examples to split.")

    shuffled = examples.copy()

    random.seed(RANDOM_SEED)
    random.shuffle(shuffled)

    total = len(shuffled)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]

    return train, val, test


def print_distribution(examples: list[dict[str, Any]], field: str) -> None:
    counts: dict[str, int] = {}

    for example in examples:
        value = str(example.get(field, "unknown"))
        counts[value] = counts.get(value, 0) + 1

    print(f"{field.capitalize()} distribution:")

    for value, count in sorted(counts.items()):
        print(f"  - {value}: {count}")


def main() -> None:
    print(f"Loading dataset from: {INPUT_PATH}")
    raw_examples = load_jsonl(INPUT_PATH)
    print(f"Loaded {len(raw_examples)} raw examples")

    for index, example in enumerate(raw_examples, start=1):
        validate_raw_example(example, index)

    print("Converting to chat format...")
    chat_examples = [to_chat_format(example) for example in raw_examples]

    print("Splitting dataset...")
    train, val, test = split_examples(chat_examples)

    print("Saving datasets...")
    save_jsonl(TRAIN_PATH, train)
    save_jsonl(VAL_PATH, val)
    save_jsonl(TEST_PATH, test)

    print("Done!")
    print(f"Total examples: {len(chat_examples)}")
    print(f"Train: {len(train)} -> {TRAIN_PATH}")
    print(f"Validation: {len(val)} -> {VAL_PATH}")
    print(f"Test: {len(test)} -> {TEST_PATH}")

    print()
    print_distribution(chat_examples, "language")
    print()
    print_distribution(chat_examples, "category")
    print()
    print_distribution(chat_examples, "source")


if __name__ == "__main__":
    main()