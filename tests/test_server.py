"""Unit tests for FastAPI live router gateway and OpenAI-compatible endpoints."""

import pytest
from starlette.testclient import TestClient
from camir.server.app import create_app


@pytest.fixture
def client():
    app = create_app(mock_mode=True)
    return TestClient(app)


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "hardware_profile" in data
    assert data["mock_mode"] is True


def test_models_list_endpoint(client):
    res = client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "list"
    model_ids = [m["id"] for m in data["data"]]
    assert "camir-auto" in model_ids
    assert "qwen2.5-1.5b-instruct" in model_ids


def test_chat_completions_endpoint(client):
    payload = {
        "model": "camir-auto",
        "messages": [
            {"role": "user", "content": "Explain Newton's third law of motion."}
        ],
        "max_tokens": 32,
        "temperature": 0.0,
        "policy": "P5",
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) > 0
    assert len(data["choices"][0]["message"]["content"]) > 0
    assert "camir_metadata" in data
    assert "routed_tier" in data["camir_metadata"]


def test_metrics_endpoint(client):
    # Send a request first
    client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Quick question"}]
    })
    res = client.get("/metrics")
    assert res.status_code == 200
    data = res.json()
    assert data["total_requests"] >= 1
    assert "tier_distribution" in data
