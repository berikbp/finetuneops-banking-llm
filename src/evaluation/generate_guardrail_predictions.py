from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.serving.safety_guardrails import get_safety_guardrail_response


BENCHMARK_PATH = Path("data/processed/benchmark_banking_support.jsonl")
BASE_PREDICTIONS_PATH = Path("data/processed/base_model_1_5b_predictions.jsonl")
OUTPUT_PATH = Path("data/processed/guardrail_base_predictions.jsonl")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error

    return rows


def generate_predictions(
    benchmark_path: Path,
    base_predictions_path: Path,
    output_path: Path,
) -> None:
    benchmark = load_jsonl(benchmark_path)
    base_predictions = load_jsonl(base_predictions_path)

    base_answer_by_id = {
        prediction["id"]: prediction["answer"]
        for prediction in base_predictions
    }

    predictions: list[dict[str, str]] = []
    guardrail_count = 0

    for item in benchmark:
        example_id = item["id"]
        guardrail = get_safety_guardrail_response(item["input"])

        if guardrail is not None:
            answer = guardrail.response
            guardrail_count += 1
        else:
            if example_id not in base_answer_by_id:
                raise ValueError(f"Missing base prediction for benchmark id: {example_id}")
            answer = base_answer_by_id[example_id]

        predictions.append(
            {
                "id": example_id,
                "answer": answer,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for prediction in predictions:
            file.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    print(f"Saved predictions to: {output_path}")
    print(f"Benchmark examples: {len(predictions)}")
    print(f"Guardrail responses: {guardrail_count}")
    print(f"Base fallback responses: {len(predictions) - guardrail_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK_PATH)
    parser.add_argument("--base-predictions", type=Path, default=BASE_PREDICTIONS_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_predictions(
        benchmark_path=args.benchmark,
        base_predictions_path=args.base_predictions,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
