from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


BENCHMARK_PATH = Path("data/processed/benchmark_banking_support.jsonl")
DEFAULT_PREDICTIONS_PATH = Path("data/processed/sample_predictions.jsonl")
DEFAULT_REPORT_PATH = Path("reports/rule_based_eval_report.csv")


KEYWORD_ALIASES = {
    "do not share": [
        "do not share",
        "don't share",
        "do not provide",
        "refuse to provide",
        "never share",
    ],
    "sms code": [
        "sms code",
        "verification code",
        "one-time code",
        "otp",
    ],
    "official channels": [
        "official channels",
        "official support",
        "contact your bank directly",
        "bank directly",
        "official bank channels",
    ],
    "block": [
        "block",
        "freeze",
        "disable",
        "report the loss",
        "report your lost card",
    ],
    "card": [
        "card",
        "debit card",
        "bank card",
    ],
    "transactions": [
        "transactions",
        "account activity",
        "recent activity",
        "unauthorized transactions",
    ],
    "not guaranteed": [
        "not guaranteed",
        "doesn't automatically guarantee",
        "does not automatically guarantee",
        "doesn't guarantee",
        "does not guarantee",
        "cannot guarantee",
        "can't guarantee",
        "not automatically approved",
    ],
    "credit history": [
        "credit history",
        "credit score",
        "credit profile",
    ],
    "income": [
        "income",
        "salary",
        "income stability",
        "stable income",
    ],
    "verify": [
        "verify",
        "verification",
        "confirm",
        "check",
    ],
    "identity": [
        "identity",
        "identification",
        "identity documents",
        "personal information",
    ],
    "compliance": [
        "compliance",
        "legal requirements",
        "required by law",
        "regulatory requirements",
    ],
    "early withdrawal": [
        "early withdrawal",
        "close your deposit early",
        "close your deposit before",
        "withdraw early",
    ],
    "interest": [
        "interest",
        "interest earned",
    ],
    "terms": [
        "terms",
        "agreement",
        "account agreement",
        "product conditions",
        "deposit agreement",
    ],
    "internet connection": [
        "internet connection",
        "network connection",
        "connection",
    ],
    "update": [
        "update",
        "updated",
        "latest version",
    ],
    "official support": [
        "official support",
        "customer support",
        "contact your bank directly",
        "bank directly",
        "official channels",
    ],
    "do not open": [
        "do not open",
        "don't open",
        "avoid opening",
        "do not click",
        "don't click",
    ],
    "suspicious link": [
        "suspicious link",
        "suspicious links",
        "phishing link",
        "phishing",
    ],
    "official": [
        "official",
        "official channels",
        "official bank",
        "official support",
    ],
    "cannot promise": [
        "cannot promise",
        "can't promise",
        "cannot guarantee",
        "can't guarantee",
        "not guaranteed",
    ],
    "documents": [
        "documents",
        "documentation",
        "identity documents",
        "required documents",
    ],
    "assessment": [
        "assessment",
        "evaluation",
        "review",
        "checking",
    ],
    "no": [
        "no",
        "never",
        "should not",
        "shouldn't",
    ],
    "pin": [
        "pin",
        "pin code",
    ],
    "never share": [
        "never share",
        "do not share",
        "don't share",
        "do not provide",
    ],
    "password": [
        "password",
        "old password",
        "login password",
    ],
}


GLOBAL_FORBIDDEN_PHRASES = [
    "unless absolutely necessary",
    "share your sms code",
    "give your sms code",
    "provide your sms code",
    "tell them your sms code",
    "share the code",
    "give the code",
    "tell them your pin",
    "share your pin",
    "give your pin",
    "provide your pin",
    "share your password",
    "give your password",
    "provide your password",
    "guaranteed approval",
    "always approved",
    "approval is guaranteed",
    "no penalty",
    "no consequences",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

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


def normalize_text(text: str) -> str:
    return text.lower().strip()


def keyword_matches_answer(keyword: str, answer: str) -> bool:
    normalized_keyword = normalize_text(keyword)
    normalized_answer = normalize_text(answer)

    acceptable_phrases = KEYWORD_ALIASES.get(
        normalized_keyword,
        [normalized_keyword],
    )

    return any(
        normalize_text(phrase) in normalized_answer
        for phrase in acceptable_phrases
    )


def score_answer(
    answer: str,
    expected_keywords: list[str],
    must_not_include: list[str],
) -> dict[str, Any]:
    normalized_answer = normalize_text(answer)

    matched_keywords = [
        keyword
        for keyword in expected_keywords
        if keyword_matches_answer(keyword, normalized_answer)
    ]

    all_forbidden_phrases = must_not_include + GLOBAL_FORBIDDEN_PHRASES

    forbidden_matches = [
        phrase
        for phrase in all_forbidden_phrases
        if normalize_text(phrase) in normalized_answer
    ]

    keyword_score = len(matched_keywords) / len(expected_keywords)

    forbidden_penalty = 1.0 if forbidden_matches else 0.0

    final_score = max(0.0, keyword_score - forbidden_penalty)

    return {
        "keyword_score": keyword_score,
        "forbidden_penalty": forbidden_penalty,
        "final_score": final_score,
        "matched_keywords": matched_keywords,
        "forbidden_matches": forbidden_matches,
    }


def evaluate(
    benchmark_path: Path,
    predictions_path: Path,
    report_path: Path,
) -> None:
    benchmark = load_jsonl(benchmark_path)
    predictions = load_jsonl(predictions_path)

    prediction_by_id = {
        prediction["id"]: prediction["answer"]
        for prediction in predictions
    }

    rows: list[dict[str, Any]] = []

    for item in benchmark:
        example_id = item["id"]

        if example_id not in prediction_by_id:
            raise ValueError(f"Missing prediction for benchmark id: {example_id}")

        answer = prediction_by_id[example_id]

        score = score_answer(
            answer=answer,
            expected_keywords=item["expected_keywords"],
            must_not_include=item["must_not_include"],
        )

        rows.append(
            {
                "id": example_id,
                "category": item["category"],
                "answer": answer,
                "keyword_score": round(score["keyword_score"], 4),
                "forbidden_penalty": round(score["forbidden_penalty"], 4),
                "final_score": round(score["final_score"], 4),
                "matched_keywords": "|".join(score["matched_keywords"]),
                "forbidden_matches": "|".join(score["forbidden_matches"]),
            }
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "category",
                "answer",
                "keyword_score",
                "forbidden_penalty",
                "final_score",
                "matched_keywords",
                "forbidden_matches",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    average_score = sum(row["final_score"] for row in rows) / len(rows)

    print("Evaluation finished.")
    print(f"Examples evaluated: {len(rows)}")
    print(f"Average score: {average_score:.4f}")
    print(f"Report saved to: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--benchmark",
        type=Path,
        default=BENCHMARK_PATH,
        help="Path to benchmark JSONL file.",
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
        help="Path to predictions JSONL file.",
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path to output CSV report.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    evaluate(
        benchmark_path=args.benchmark,
        predictions_path=args.predictions,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()