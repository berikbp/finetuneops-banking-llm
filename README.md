# FinTuneOps: Banking LLM Evaluation, LoRA, and Safety Guardrails

FinTuneOps is an AI engineering project for comparing three approaches to banking customer-support assistants:

- a strong instruction base model,
- LoRA fine-tuning experiments,
- deterministic safety guardrails for high-risk banking scenarios.

The final result is a working FastAPI service that compares Base Qwen2.5-1.5B, LoRA v4, and Guardrail + Base responses side by side. On the project benchmark, Guardrail + Base is the strongest system:

```text
Base Qwen2.5-1.5B: 0.7333
Best LoRA, v4:      0.6667
Guardrail + Base:   0.8333
```

## Architecture

```text
User question
    |
    v
FastAPI comparison API
    |
    +--> Base Qwen2.5-1.5B-Instruct
    |
    +--> Qwen2.5-1.5B + LoRA v4 adapter
    |
    +--> Banking safety guardrail
            |
            +-- risky security pattern -> deterministic safe response
            |
            +-- no risky pattern ------> Base model response
```

Runtime modules:

```text
src/serving/api.py               FastAPI routes
src/serving/model_service.py     Lazy Qwen + LoRA v4 model loading
src/serving/safety_guardrails.py Deterministic banking safety rules
src/serving/settings.py          Shared serving constants
```

Experiment modules:

```text
src/training/train_lora.py       Assistant-token-only LoRA SFT
src/evaluation/                 Prediction generation and scoring
src/data/                       Dataset conversion and formatting
```

## Why This Project Matters

The project deliberately compares fine-tuning against a simpler production control: deterministic guardrails.

The main finding is practical:

> Small LoRA fine-tuning did not beat the strong base model on this banking-safety benchmark. The best production direction is a hybrid system: strong base model + deterministic safety guardrails, with RAG later for bank-specific policy grounding.

## Results

| System | Score | Notes |
|---|---:|---|
| Base Qwen2.5-1.5B-Instruct | 0.7333 | Strong baseline |
| LoRA v1 | 0.4667 | Early fine-tune |
| LoRA v2 | 0.5333 | External data still introduced artifacts |
| LoRA v3 custom-only | 0.4333 | Cleaner data, but too small |
| LoRA v4 assistant-token-only | 0.6667 | Correct SFT masking helped substantially |
| LoRA v5 fraud-heavy | 0.5000 | More fraud data made answers generic |
| Guardrail + Base | 0.8333 | Best result |

The important learning was not that every fine-tune should win. It was that evaluation revealed when fine-tuning made the system worse, and that a guardrail layer solved the highest-risk failure mode more reliably.

## Setup

Install dependencies:

```bash
uv sync
```

The real model API expects:

```text
outputs/qwen2_5_1_5b_banking_lora_v4
```

That adapter is used for `/generate/lora-v4`. The base path and LoRA path share one loaded base model; base generation temporarily disables the LoRA adapter.

## Serve The API

For real model serving:

```bash
make serve
```

For lightweight API development only:

```bash
make serve-dev
```

Do not use reload for real model serving because reload can duplicate model loading.

Open:

```text
http://127.0.0.1:8000/docs
```

## API

```text
GET  /health
POST /generate/base
POST /generate/lora-v4
POST /generate/guardrail-base
POST /compare
```

Example:

```bash
curl -X POST "http://127.0.0.1:8000/compare" \
  -H "Content-Type: application/json" \
  -d '{"question": "Someone called me and asked for my SMS code to cancel a suspicious transaction. What should I do?"}'
```

The response includes:

```json
{
  "question": "...",
  "base_answer": "...",
  "lora_v4_answer": "...",
  "guardrail_base_answer": "...",
  "guardrail_triggered": true
}
```

## Demo Examples

### Fraud Case

Input:

```text
Someone called me and asked for my SMS code to cancel a suspicious transaction. What should I do?
```

Expected behavior:

```text
guardrail_triggered: true
```

The guardrail returns a deterministic safety response telling the user not to share the SMS code and to contact the bank through official channels.

### Normal Loan Case

Input:

```text
Can the bank promise my loan will be approved before checking my documents?
```

Expected behavior:

```text
guardrail_triggered: false
```

The system uses the base model response because this is not a fraud or security pattern.

## Evaluation

Score the stored Guardrail + Base benchmark predictions:

```bash
make generate-guardrail-base
make eval-guardrail-base
make summarize-guardrail-base
```

Score the stored base benchmark predictions:

```bash
make eval-base-1-5b
make summarize-base-1-5b
```

## Training

The final LoRA training objective masks system and user prompt tokens with `-100`, so the model learns only assistant answer tokens:

```text
Given system + user -> produce assistant answer
```

Run training with:

```bash
make train-lora
```

Training configuration:

```text
configs/train_qwen_1_5b_lora.yaml
```

## Docker

Build:

```bash
make docker-build
```

Run portable API mode:

```bash
make docker-run
```

Run with Docker GPU reservation, only when NVIDIA Container Toolkit is installed:

```bash
make docker-run-gpu
```

Test:

```bash
make docker-test-health
```

Stop:

```bash
make docker-stop
```

GPU check:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If that command fails, Docker cannot see the GPU. The API can still serve guardrail routes, but real model inference inside Docker requires NVIDIA Container Toolkit.

## Repository Layout

```text
configs/                  Training config
data/processed/           Small benchmark, generated predictions, curated data
reports/                  Evaluation reports and experiment notes
src/data/                 Dataset conversion and formatting
src/evaluation/           Prediction/evaluation scripts
src/serving/              FastAPI service and safety guardrails
src/training/             LoRA training code
outputs/..._lora_v4/      Best LoRA adapter used by the API
```

## Limitations

This project is an engineering prototype, not a production banking assistant.

Current limitations:

- The benchmark is small and should be expanded to 100+ examples.
- The evaluator is rule-based and does not fully measure answer quality.
- Guardrails are deterministic pattern rules and may miss unseen fraud wording.
- The project does not yet include RAG over real bank policies or product documents.
- Docker GPU inference requires NVIDIA Container Toolkit on the host.
- The system is not suitable for real customer use without compliance review, security testing, monitoring, and human escalation flows.

## Final Takeaway

For banking support, high-risk security cases should not rely only on probabilistic model behavior. The most reliable architecture here is:

```text
base instruction model + deterministic fraud/security guardrails + future RAG for bank policy
```
