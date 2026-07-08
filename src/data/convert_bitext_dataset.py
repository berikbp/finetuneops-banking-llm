from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset


DATASET_NAME = "bitext/Bitext-retail-banking-llm-chatbot-training-dataset"
OUTPUT_PATH = Path("data/processed/bitext_banking_converted.jsonl")

RANDOM_SEED = 42
MAX_EXAMPLES = 100

DEFAULT_INSTRUCTION = (
    "Answer the banking customer question in a safe, clear, and professional way."
)

CATEGORY_MAP = {
    "CARD": "card_support",
    "ACCOUNT": "account_support",
    "TRANSFER": "transfer_support",
    "LOAN": "loan_faq",
    "PAYMENT": "payment_support",
    "CASH": "cash_withdrawal",
    "CONTACT": "customer_support",
    "ATM": "atm",
    "FEES": "fees",
    "PASSWORD": "password",
    "FIND": "find",
}

BAD_TEXT_PATTERNS = [
    "fucking",
    "<fucking>",
    "fuck",
    "shit",
    "bitch",
    "your your",
    "your the official bank information",
    "the official bank information associated",
    "insert phone number",
    "insert email",
    "insert",
    "osme information",
    "ya help me",
    "uhelp",
    "dnt",
]

RISKY_RESPONSE_PATTERNS = [
    "share your password",
    "share your pin",
    "share your sms code",
    "give your password",
    "give your pin",
    "give your sms code",
    "provide your password",
    "provide your pin",
    "provide your sms code",
]


def clean_text(text: str) -> str:
    text = str(text).strip()

    replacements = {
        "{{Credit Card}}": "your card",
        "{{Debit Card}}": "your card",
        "{{Banking App}}": "the banking app",
        "{{Card Services}}": "Card Services",
        "{{Manage Cards}}": "Manage Cards",
        "{{Company Website URL}}": "the official website",
        "{{Customer Support Phone Number}}": "the official customer support number",
        "{{Customer Support Working Hours}}": "working hours",
        "{{Branch Locator}}": "the official branch locator",
        "{{ATM Locator}}": "the official ATM locator",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Replace any remaining template placeholders.
    text = re.sub(r"\{\{[^}]+\}\}", "official bank information", text)

    # Fix common artifacts after placeholder replacement.
    text = text.replace("your your", "your")
    text = text.replace("your the official bank information", "the required information")
    text = text.replace(
        "the official bank information associated with your account",
        "the information associated with your account",
    )

    # Normalize whitespace.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def has_bad_pattern(text: str) -> bool:
    normalized = text.lower()
    return any(pattern in normalized for pattern in BAD_TEXT_PATTERNS)


def has_risky_pattern(text: str) -> bool:
    normalized = text.lower()
    return any(pattern in normalized for pattern in RISKY_RESPONSE_PATTERNS)


def is_good_example(item: dict[str, Any]) -> bool:
    instruction = clean_text(item.get("instruction", ""))
    response = clean_text(item.get("response", ""))

    if len(instruction) < 8:
        return False

    if len(response) < 30:
        return False

    # For first LoRA experiment, avoid very long answers.
    if len(response.split()) > 180:
        return False

    # Remove unresolved placeholders.
    if "{{" in instruction or "}}" in instruction:
        return False

    if "{{" in response or "}}" in response:
        return False

    # Remove noisy/profane examples.
    if has_bad_pattern(instruction) or has_bad_pattern(response):
        return False

    # Remove unsafe examples.
    if has_risky_pattern(response):
        return False

    return True


def normalize_category(category: str) -> str:
    category = str(category).strip().upper()
    return CATEGORY_MAP.get(category, category.lower())


def convert_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruction": DEFAULT_INSTRUCTION,
        "input": clean_text(item["instruction"]),
        "output": clean_text(item["response"]),
        "category": normalize_category(item["category"]),
        "language": "en",
        "source": "bitext_retail_banking",
        "intent": str(item["intent"]).strip(),
    }


def main() -> None:
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)
    split = dataset["train"]

    print(f"Loaded rows: {len(split)}")

    converted: list[dict[str, Any]] = []
    seen_inputs: set[str] = set()

    for item in split:
        if not is_good_example(item):
            continue

        converted_item = convert_item(item)
        normalized_input = converted_item["input"].lower().strip()

        if normalized_input in seen_inputs:
            continue

        seen_inputs.add(normalized_input)
        converted.append(converted_item)

    print(f"Good converted examples before sampling: {len(converted)}")

    random.seed(RANDOM_SEED)
    random.shuffle(converted)

    sampled = converted[:MAX_EXAMPLES]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for example in sampled:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Saved examples: {len(sampled)}")
    print(f"Output path: {OUTPUT_PATH}")

    category_counts: dict[str, int] = {}

    for example in sampled:
        category = example["category"]
        category_counts[category] = category_counts.get(category, 0) + 1

    print("Category distribution:")

    for category, count in sorted(category_counts.items()):
        print(f"  - {category}: {count}")


if __name__ == "__main__":
    main()
