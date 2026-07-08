from __future__ import annotations

from pathlib import Path


BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_V4_PATH = Path("outputs/qwen2_5_1_5b_banking_lora_v4")

MAX_NEW_TOKENS = 180

SYSTEM_PROMPT = (
    "You are a professional banking AI assistant. "
    "Answer safely, clearly, and concisely. "
    "Never ask for passwords, PIN codes, SMS codes, OTP codes, CVV, or full card details."
)
