from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


BENCHMARK_PATH = Path('data/processed/benchmark_banking_support.jsonl')
OUTPUT_PATH = Path('data/processed/lora_model_v5_predictions.jsonl')

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_MODEL_PATH = "outputs/qwen2_5_1_5b_banking_lora_v5"

SYSTEM_PROMPT = (
    "You are a professional banking AI assistant. "
    "Answer customer questions in a safe, clear, and concise way. "
    "Do not promise loan approval, do not invent bank policy, "
    "and never ask for PIN codes, passwords, or SMS codes."
)

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    exmaples = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                exmaples.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error

    return exmaples


def build_messages(question: str) -> list[dict[str, str]]:
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
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=True,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return answer.strip()


def main():
    print(f'Loading tokenizer: {BASE_MODEL}')
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f'Loading LoRA model: {BASE_MODEL}')
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )

    print(f'Loading LoRA model: {LORA_MODEL_PATH}')
    model = PeftModel.from_pretrained(
        base_model,
        LORA_MODEL_PATH,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    print(f'Loading benchmark dataset: {BENCHMARK_PATH}')
    benchmark_examples = load_jsonl(BENCHMARK_PATH)
    print(f'Loaded {len(benchmark_examples)} examples')

    results = []

    print('Generating predictions...')
    for example in tqdm(benchmark_examples, desc='Generating answers'):
        answer = generate_answer(
            model=model,
            tokenizer=tokenizer,
            question=example['input']
        )

        results.append({
            'id': example['id'],
            'answer': answer
        })

    print(f'Saving predictions to: {OUTPUT_PATH}')
    with OUTPUT_PATH.open('w', encoding='utf-8') as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + '\n')

    print('Done!')

if __name__ == '__main__':
    main()

    
