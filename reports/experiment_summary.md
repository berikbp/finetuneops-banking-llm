# Experiment Summary

## Goal

The goal was to evaluate whether LoRA fine-tuning improves a banking customer-support assistant compared with a strong base model, and whether deterministic safety guardrails improve high-risk banking scenarios.

## Results

| System | Score |
|---|---:|
| Base Qwen2.5-1.5B-Instruct | 0.7333 |
| LoRA v1 | 0.4667 |
| LoRA v2 | 0.5333 |
| LoRA v3 custom-only | 0.4333 |
| LoRA v4 assistant-token-only | 0.6667 |
| LoRA v5 fraud-heavy | 0.5000 |
| Guardrail + Base | 0.8333 |

## Findings

LoRA fine-tuning did not outperform the base model on this benchmark. The best LoRA version was v4, which improved after masking system/user tokens and training only on assistant tokens.

The best overall system was Guardrail + Base. It improved the score from 0.7333 to 0.8333 and reached 1.0000 on fraud-warning and card-security categories.

## Main Lesson

For banking safety use cases, deterministic guardrails can be more reliable than small-data fine-tuning. A production system should combine a strong base model, guardrails, RAG, and continuous evaluation.
