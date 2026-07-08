# Baseline Model Comparison

## Models Compared

| Model | Parameters | Evaluator Version | Score |
|---|---:|---|---:|
| Qwen2.5-0.5B-Instruct | 0.5B | exact keyword evaluator | 0.3000 |
| Qwen2.5-1.5B-Instruct | 1.5B | exact keyword evaluator | 0.4667 |
| Qwen2.5-1.5B-Instruct | 1.5B | alias-aware evaluator | 0.7333 |

## Key Finding

Qwen2.5-1.5B-Instruct is much stronger than Qwen2.5-0.5B-Instruct for banking customer-support answers.

The first evaluator underestimated the 1.5B model because it required exact keyword matches. After adding acceptable paraphrases, the score increased from 0.4667 to 0.7333.

## Observed Improvements in 1.5B Model

- Better fraud-warning behavior.
- Better refusal to guarantee loan approval.
- More cautious deposit-related answers.
- More professional structure.
- Better safety wording around SMS codes and suspicious calls.

## Remaining Problems

- Answers are too long for real customer support.
- Some responses are generic.
- The model sometimes gives broad advice instead of concise banking instructions.
- The model is not grounded in actual bank policy.
- Evaluation is still rule-based and imperfect.

## Engineering Conclusion

The 1.5B baseline is usable as a starting point, but not production-ready.

Fine-tuning may help improve:

- answer brevity,
- consistent banking tone,
- strict fraud warnings,
- refusal to guarantee loan approval,
- avoiding invented bank policy,
- following a predictable support-answer format.

RAG would still be needed for factual bank-specific product information.
