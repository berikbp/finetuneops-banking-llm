from __future__ import annotations

from datasets import load_dataset


DATASET_NAME = "bitext/Bitext-retail-banking-llm-chatbot-training-dataset"


def main() -> None:
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)

    print("\nDataset splits:")
    print(dataset)

    split_name = list(dataset.keys())[0]
    split = dataset[split_name]

    print(f"\nUsing split: {split_name}")
    print(f"Number of rows: {len(split)}")

    print("\nColumns:")
    print(split.column_names)

    print("\nFirst 5 examples:")
    for i in range(min(5, len(split))):
        print("=" * 80)
        print(split[i])


if __name__ == "__main__":
    main()