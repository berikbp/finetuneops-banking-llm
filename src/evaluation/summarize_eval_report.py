from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_REPORT_PATH = Path("reports/base_model_1_5b_eval_report.csv")


def summarize_report(report_path: Path) -> None:
    if not report_path.exists():
        raise FileNotFoundError(f"Report file not found: {report_path}")

    rows: list[dict[str, str]] = []

    with report_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise ValueError("Report is empty.")

    scores = [float(row["final_score"]) for row in rows]
    average_score = sum(scores) / len(scores)

    category_scores: dict[str, list[float]] = {}

    for row in rows:
        category = row["category"]
        score = float(row["final_score"])
        category_scores.setdefault(category, []).append(score)

    print(f"Report: {report_path}")
    print(f"Examples: {len(rows)}")
    print(f"Average score: {average_score:.4f}")
    print()
    print("Category scores:")

    for category, values in sorted(category_scores.items()):
        category_average = sum(values) / len(values)
        print(f"  - {category}: {category_average:.4f} ({len(values)} examples)")

    print()
    print("Lowest scoring examples:")

    sorted_rows = sorted(rows, key=lambda row: float(row["final_score"]))

    for row in sorted_rows[:5]:
        print(
            f"  - {row['id']} | {row['category']} | "
            f"score={float(row['final_score']):.4f} | "
            f"matched={row['matched_keywords']} | "
            f"forbidden={row['forbidden_matches']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path to evaluation CSV report.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize_report(args.report)


if __name__ == "__main__":
    main()