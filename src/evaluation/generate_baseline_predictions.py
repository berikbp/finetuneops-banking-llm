from __future__ import annotations 

import json 
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


BENCHMARK_PATH = Path("data/processed/benchmark_banking_support.jsonl")
OUTPUT_PATH = Path("data/processed/base_model_1_5b_predictions.jsonl")

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = (
    "You are a professional banking AI assistant. "
    "Answer customer questions in a safe, clear, and concise way. "
    "Do not promise loan approval, do not invent bank policy, "
    "and never ask for PIN codes, passwords, or SMS codes."
)


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


def build_messages(question: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                "Answer the banking customer question in a safe, clear, and professional way.\n\n"
                f"Customer question:\n{question}"
            ),
        },
    ]


def generate_answer(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, question: str) -> str:
    messages = build_messages(question)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
        
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return answer.strip()


def main() -> None:
    print('Loading model', MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    benchmark = load_jsonl(BENCHMARK_PATH)
    predictions = []

    for item in tqdm(benchmark, desc="Generating predictions"):
        answer = generate_answer(model, tokenizer, item["input"])
        predictions.append({
            "id": item["id"],
            "answer": answer,
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for prediction in predictions:
            file.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    print("Predictions saved to:", OUTPUT_PATH)

if __name__ == "__main__":
    main()
