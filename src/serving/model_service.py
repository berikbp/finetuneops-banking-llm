from __future__ import annotations

from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.serving.settings import BASE_MODEL_NAME, LORA_V4_PATH, MAX_NEW_TOKENS, SYSTEM_PROMPT


class BankingModelService:
    def __init__(self) -> None:
        self.tokenizer: AutoTokenizer | None = None
        self.base_model: Any | None = None
        self.lora_v4_model: Any | None = None

    def load(self) -> None:
        if self.base_model is not None:
            return

        print(f"Loading tokenizer: {BASE_MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        print(f"Loading base model: {BASE_MODEL_NAME}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.base_model.eval()

        print(f"Loading LoRA v4 adapter: {LORA_V4_PATH}")
        self.lora_v4_model = PeftModel.from_pretrained(
            self.base_model,
            LORA_V4_PATH,
        )
        self.lora_v4_model.eval()

        print("Model service loaded.")

    def generate_base(self, question: str) -> str:
        self.load()

        if self.lora_v4_model is None:
            raise RuntimeError("LoRA v4 model wrapper is not loaded.")

        with self.lora_v4_model.disable_adapter():
            return self._generate(self.lora_v4_model, question)

    def generate_lora_v4(self, question: str) -> str:
        self.load()

        if self.lora_v4_model is None:
            raise RuntimeError("LoRA v4 model wrapper is not loaded.")

        return self._generate(self.lora_v4_model, question)

    def _generate(self, model: Any, question: str) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded.")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        answer = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return answer.strip()


model_service = BankingModelService()
