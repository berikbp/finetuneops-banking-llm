# Base Model Error Analysis

## Model

Qwen/Qwen2.5-0.5B-Instruct

## Benchmark

Banking customer support benchmark with 10 examples.

## Average Rule-Based Score

0.3000

## Main Failure Types

### 1. Fraud safety weakness

In SMS/PIN/security questions, the model sometimes gives general advice instead of strict banking safety rules.

Expected behavior:

- Never share SMS codes.
- Never share PIN codes.
- Contact the bank only through official channels.
- Block the card if needed.

Observed issue:

- The model used weak phrasing such as “unless absolutely necessary,” which is unsafe in banking contexts.

### 2. Loan approval hallucination

The model suggested that receiving salary through the bank may imply loan approval.

Expected behavior:

- Loan approval must never be guaranteed.
- The model should mention credit history, income stability, debt load, documents, and bank policy.

### 3. Deposit policy hallucination

The model claimed there may be no penalty for closing a deposit early.

Expected behavior:

- Early withdrawal may affect interest or product conditions.
- The model should avoid inventing exact bank policy.

## Conclusion

The base model can produce fluent answers, but it is not reliable enough for banking support without stronger prompting, RAG, fine-tuning, or guardrails.
