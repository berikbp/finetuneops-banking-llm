from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


DEFAULT_CONFIG_PATH = Path("configs/train_qwen_1_5b_lora.yaml")


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_example(example: dict[str, Any], tokenizer: AutoTokenizer) -> dict[str, str]:
    messages = example["messages"]

    prompt_messages = messages[:-1]
    assistant_message = messages[-1]

    if assistant_message["role"] != "assistant":
        raise ValueError("Last message must be assistant message.")

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {
        "prompt_text": prompt_text,
        "full_text": full_text,
    }


def tokenize_example(
    example: dict[str, str],
    tokenizer: AutoTokenizer,
    max_seq_length: int,
) -> dict[str, Any]:
    prompt_tokens = tokenizer(
        example["prompt_text"],
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )

    full_tokens = tokenizer(
        example["full_text"],
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]
    labels = input_ids.copy()

    prompt_length = min(len(prompt_tokens["input_ids"]), len(labels))

    for i in range(prompt_length):
        labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class DataCollatorForCausalLM:
    def __init__(self, tokenizer: AutoTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["labels"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels

        return batch


def print_trainable_parameters(model: torch.nn.Module) -> None:
    trainable_params = 0
    all_params = 0

    for _, param in model.named_parameters():
        num_params = param.numel()
        all_params += num_params

        if param.requires_grad:
            trainable_params += num_params

    trainable_percent = 100 * trainable_params / all_params

    print(
        f"Trainable params: {trainable_params:,} "
        f"|| All params: {all_params:,} "
        f"|| Trainable: {trainable_percent:.4f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to training config YAML.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    seed = int(config["training"]["seed"])
    set_seed(seed)

    model_name = config["model"]["name"]
    max_seq_length = int(config["model"]["max_seq_length"])

    train_path = Path(config["data"]["train_path"])
    val_path = Path(config["data"]["val_path"])
    output_dir = Path(config["output"]["dir"])

    print(f"Model: {model_name}")
    print(f"Train path: {train_path}")
    print(f"Validation path: {val_path}")
    print(f"Output dir: {output_dir}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading datasets...")
    train_rows = load_jsonl(train_path)
    val_rows = load_jsonl(val_path)

    print(f"Train examples: {len(train_rows)}")
    print(f"Validation examples: {len(val_rows)}")

    train_dataset = Dataset.from_list(train_rows)
    val_dataset = Dataset.from_list(val_rows)

    print("Formatting chat examples...")
    train_dataset = train_dataset.map(
        lambda example: format_example(example, tokenizer),
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        lambda example: format_example(example, tokenizer),
        remove_columns=val_dataset.column_names,
    )

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda example: tokenize_example(example, tokenizer, max_seq_length),
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        lambda example: tokenize_example(example, tokenizer, max_seq_length),
        remove_columns=val_dataset.column_names,
    )

    print("Loading model in 4-bit...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=int(config["lora"]["r"]),
        lora_alpha=int(config["lora"]["alpha"]),
        lora_dropout=float(config["lora"]["dropout"]),
        target_modules=config["lora"]["target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    training_config = config["training"]

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(training_config["num_train_epochs"]),
        per_device_train_batch_size=int(training_config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training_config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training_config["gradient_accumulation_steps"]),
        learning_rate=float(training_config["learning_rate"]),
        warmup_ratio=float(training_config["warmup_ratio"]),
        weight_decay=float(training_config["weight_decay"]),
        logging_steps=int(training_config["logging_steps"]),
        eval_steps=int(training_config["eval_steps"]),
        save_steps=int(training_config["save_steps"]),
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        seed=seed,
    )

    data_collator = DataCollatorForCausalLM(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print("Saving LoRA adapter...")
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Training finished. Adapter saved to: {output_dir}")


if __name__ == "__main__":
    main()
