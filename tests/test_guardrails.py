from src.serving.safety_guardrails import apply_banking_guardrail


def test_sms_code_triggers_guardrail():
    result = apply_banking_guardrail(
        "Someone called me and asked for my SMS code to cancel a suspicious transaction."
    )

    assert result.triggered is True
    assert "SMS" in result.response or "code" in result.response
    assert "official" in result.response.lower()


def test_pin_code_triggers_guardrail():
    result = apply_banking_guardrail(
        "A bank employee is asking for my PIN code by phone."
    )

    assert result.triggered is True
    assert "PIN" in result.response or "pin" in result.response.lower()


def test_password_request_triggers_guardrail():
    result = apply_banking_guardrail(
        "Support asked me to send my mobile banking password."
    )

    assert result.triggered is True
    assert "password" in result.response.lower()


def test_safe_loan_question_does_not_trigger_guardrail():
    result = apply_banking_guardrail(
        "Can the bank promise my loan will be approved before checking my documents?"
    )

    assert result.triggered is False
