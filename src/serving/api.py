from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from src.serving.model_service import model_service
from src.serving.safety_guardrails import apply_banking_guardrail


app = FastAPI(
    title="FinTuneOps Banking LLM API",
    description=(
        "Comparison API for base model, LoRA fine-tuned model, "
        "and guardrail-enhanced banking assistant."
    ),
    version="0.1.0",
)


class GenerateRequest(BaseModel):
    question: str


class GenerateResponse(BaseModel):
    question: str
    answer: str
    model_type: str
    guardrail_triggered: bool = False


class CompareResponse(BaseModel):
    question: str
    base_answer: str
    lora_v4_answer: str
    guardrail_base_answer: str
    guardrail_triggered: bool


def _guardrail_or_base_answer(question: str) -> tuple[str, bool]:
    guardrail = apply_banking_guardrail(question)

    if guardrail.triggered:
        return guardrail.response, True

    return model_service.generate_base(question), False


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate/base", response_model=GenerateResponse)
def generate_base(request: GenerateRequest) -> GenerateResponse:
    answer = model_service.generate_base(request.question)

    return GenerateResponse(
        question=request.question,
        answer=answer,
        model_type="base",
        guardrail_triggered=False,
    )


@app.post("/generate/lora-v4", response_model=GenerateResponse)
def generate_lora_v4(request: GenerateRequest) -> GenerateResponse:
    answer = model_service.generate_lora_v4(request.question)

    return GenerateResponse(
        question=request.question,
        answer=answer,
        model_type="lora_v4",
        guardrail_triggered=False,
    )


@app.post("/generate/guardrail-base", response_model=GenerateResponse)
def generate_guardrail_base(request: GenerateRequest) -> GenerateResponse:
    answer, triggered = _guardrail_or_base_answer(request.question)

    return GenerateResponse(
        question=request.question,
        answer=answer,
        model_type="guardrail_base",
        guardrail_triggered=triggered,
    )


@app.post("/compare", response_model=CompareResponse)
def compare(request: GenerateRequest) -> CompareResponse:
    base_answer = model_service.generate_base(request.question)
    lora_v4_answer = model_service.generate_lora_v4(request.question)
    guardrail_result = apply_banking_guardrail(request.question)
    triggered = guardrail_result.triggered
    guardrail_base_answer = guardrail_result.response if triggered else base_answer

    return CompareResponse(
        question=request.question,
        base_answer=base_answer,
        lora_v4_answer=lora_v4_answer,
        guardrail_base_answer=guardrail_base_answer,
        guardrail_triggered=triggered,
    )
