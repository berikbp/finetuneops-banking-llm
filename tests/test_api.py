from fastapi.testclient import TestClient

from src.serving.api import app
from src.serving.model_service import model_service


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_compare_guardrail_triggers_for_sms_code(monkeypatch):
    monkeypatch.setattr(
        model_service,
        "generate_base",
        lambda question: "Mock base answer.",
    )
    monkeypatch.setattr(
        model_service,
        "generate_lora_v4",
        lambda question: "Mock LoRA v4 answer.",
    )

    response = client.post(
        "/compare",
        json={
            "question": "Someone called me and asked for my SMS code to cancel a suspicious transaction."
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["guardrail_triggered"] is True
    assert "guardrail_base_answer" in data
    assert data["base_answer"] == "Mock base answer."
    assert data["lora_v4_answer"] == "Mock LoRA v4 answer."
