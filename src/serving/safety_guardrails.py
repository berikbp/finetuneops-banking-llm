from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailRule:
    intent: str
    patterns: tuple[str, ...]
    answer: str


@dataclass(frozen=True)
class BankingGuardrailResult:
    triggered: bool
    response: str
    intent: str | None = None


GUARDRAIL_RULES: tuple[GuardrailRule, ...] = (
    GuardrailRule(
        intent="phishing_link",
        patterns=(r"suspicious link", r"\blink\b", r"phishing", r"verify my card"),
        answer=(
            "Do not open suspicious links or enter card details. "
            "This may be phishing. Use only official bank channels."
        ),
    ),
    GuardrailRule(
        intent="verification_code_fraud",
        patterns=(r"\bsms\b", r"\botp\b", r"one[- ]?time", r"verification code"),
        answer=(
            "Do not share any SMS code or verification code. "
            "This may be fraud. Contact the bank only through official channels."
        ),
    ),
    GuardrailRule(
        intent="pin_code_fraud",
        patterns=(r"\bpin\b", r"pin code"),
        answer=(
            "No. Never share a PIN code by phone, even if someone claims to be a bank employee. "
            "Contact the bank only through official channels."
        ),
    ),
    GuardrailRule(
        intent="card_data_fraud",
        patterns=(r"\bcvv\b", r"card details", r"full card", r"card data"),
        answer=(
            "Do not share any card details, CVV, PIN code, password, or SMS code. "
            "This may be fraud. Use only official bank channels."
        ),
    ),
    GuardrailRule(
        intent="password_safety",
        patterns=(r"old password", r"\bpassword\b", r"login password"),
        answer=(
            "Do not share any password. Use the official password recovery process "
            "in the banking app or contact official support."
        ),
    ),
    GuardrailRule(
        intent="lost_or_stolen_card",
        patterns=(r"lost .*card", r"card .*lost", r"stolen .*card", r"card .*stolen"),
        answer=(
            "Block your card immediately through the official banking app or support line. "
            "Then check recent transactions and report any suspicious activity to the bank."
        ),
    ),
    GuardrailRule(
        intent="remote_access_fraud",
        patterns=(r"remote access", r"share my screen", r"screen share"),
        answer=(
            "Do not install remote access apps or share your screen while using banking services. "
            "This may be fraud. Contact the bank only through official channels."
        ),
    ),
)


def _has_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def get_safety_guardrail_response(question: str) -> BankingGuardrailResult | None:
    normalized = question.lower()

    for rule in GUARDRAIL_RULES:
        if _has_any(normalized, rule.patterns):
            return BankingGuardrailResult(
                triggered=True,
                response=rule.answer,
                intent=rule.intent,
            )

    return None


def apply_banking_guardrail(question: str) -> BankingGuardrailResult:
    guardrail = get_safety_guardrail_response(question)

    if guardrail is None:
        return BankingGuardrailResult(
            triggered=False,
            response="",
            intent=None,
        )

    return guardrail
