.PHONY: \
	validate-data format-data validate-benchmark eval-sample \
	convert-bitext merge-sources prepare-training-data data \
	generate-base eval-base-1-5b summarize-base-1-5b \
	train-lora generate-lora eval-lora summarize-lora \
	generate-guardrail-base eval-guardrail-base summarize-guardrail-base \
	test check \
	serve serve-dev \
	docker-build docker-run docker-run-detached docker-run-gpu docker-run-gpu-detached \
	docker-stop docker-logs docker-test-health

validate-data:
	uv run python -m src.data.validate_schema

format-data:
	uv run python -m src.data.format_chat_dataset

validate-benchmark:
	uv run python -m src.evaluation.validate_benchmark

eval-sample:
	uv run python -m src.evaluation.rule_based_eval

convert-bitext:
	uv run python -m src.data.convert_bitext_dataset

merge-sources:
	uv run python -m src.data.merge_training_sources

prepare-training-data: convert-bitext merge-sources format-data

data: validate-data format-data validate-benchmark

generate-base:
	uv run python -m src.evaluation.generate_baseline_predictions

eval-base-1-5b:
	uv run python -m src.evaluation.rule_based_eval \
		--predictions data/processed/base_model_1_5b_predictions.jsonl \
		--report reports/base_model_1_5b_eval_report.csv

summarize-base-1-5b:
	uv run python -m src.evaluation.summarize_eval_report \
		--report reports/base_model_1_5b_eval_report.csv

train-lora:
	uv run python -m src.training.train_lora --config configs/train_qwen_1_5b_lora.yaml

generate-lora:
	uv run python -m src.evaluation.generate_lora_predictions

eval-lora:
	uv run python -m src.evaluation.rule_based_eval \
		--predictions data/processed/lora_model_predictions.jsonl \
		--report reports/lora_model_eval_report.csv

summarize-lora:
	uv run python -m src.evaluation.summarize_eval_report \
		--report reports/lora_model_eval_report.csv

generate-guardrail-base:
	uv run python -m src.evaluation.generate_guardrail_predictions

eval-guardrail-base:
	uv run python -m src.evaluation.rule_based_eval \
		--predictions data/processed/guardrail_base_predictions.jsonl \
		--report reports/guardrail_base_eval_report.csv

summarize-guardrail-base:
	uv run python -m src.evaluation.summarize_eval_report \
		--report reports/guardrail_base_eval_report.csv

test:
	uv run pytest

check:
	uv run python -m compileall src
	uv run pytest

serve:
	uv run uvicorn src.serving.api:app --host 0.0.0.0 --port 8000

serve-dev:
	uv run uvicorn src.serving.api:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker compose build

docker-run:
	docker compose up

docker-run-detached:
	docker compose up -d

docker-run-gpu:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up

docker-run-gpu-detached:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

docker-stop:
	docker compose down

docker-logs:
	docker compose logs -f finetuneops-api

docker-test-health:
	curl -sS --fail http://127.0.0.1:8000/health
