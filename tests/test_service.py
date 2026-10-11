"""Offline API contracts with fake components, not real-model predictions.

Real DistilBERT boundary equivalence remains deferred until the pinned local
model is available. These fake counts include two synthetic special tokens.
"""

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest
import torch

from src.app import create_app
from src.backend import V0Backend


class FakeTokenizer:
    def __call__(self, text, *, add_special_tokens, truncation, max_length=None,
                 return_tensors=None):
        assert add_special_tokens is True
        words = text.split()
        ids = [101] + [9 if word == "bad" else 7 for word in words] + [102]
        if truncation:
            ids = ids[:max_length - 1] + [102] if len(ids) > max_length else ids
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids]),
                    "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}
        return {"input_ids": ids}


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Reverse the usual SST-2 ordering to prove configuration drives labels.
        self.config = SimpleNamespace(id2label={0: "POSITIVE", 1: "NEGATIVE"})
        self.inputs = []

    def forward(self, input_ids, attention_mask):
        assert torch.is_inference_mode_enabled()
        assert not self.training
        assert input_ids.device.type == "cpu"
        self.inputs.append(input_ids.tolist()[0])
        negative = bool((input_ids == 9).any())
        return SimpleNamespace(logits=torch.tensor([[0.0, 2.0] if negative else [2.0, 0.0]]))


@pytest.fixture
def service():
    tokenizer = FakeTokenizer()
    model = FakeModel()
    backend = V0Backend(tokenizer, model)
    with TestClient(create_app(lambda: backend)) as client:
        yield client, tokenizer, model


@pytest.fixture(scope="module")
def edge_cases():
    path = Path(__file__).with_name("edge_cases.jsonl")
    return {row["id"]: row["text"] for row in
            (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())}


def test_health_and_single_startup_initialization():
    calls = []

    def factory():
        calls.append("initialized")
        return V0Backend(FakeTokenizer(), FakeModel())

    with TestClient(create_app(factory)) as client:
        assert client.get("/health").json() == {"status": "ok", "version": "v0"}
        assert client.get("/health").status_code == 200
        assert client.post("/classify", json={"text": "good"}).status_code == 200
    assert calls == ["initialized"]


@pytest.mark.parametrize("text,label", [("good service", "POSITIVE"), ("bad service", "NEGATIVE")])
def test_classification_schema_softmax_and_timings(service, text, label):
    client, _, _ = service
    response = client.post("/classify", json={"text": text})
    assert response.status_code == 200
    result = response.json()
    assert set(result) == {"label", "score", "truncated", "timings_ms"}
    assert result["label"] == label
    assert 0 <= result["score"] <= 1
    assert result["score"] == pytest.approx(torch.softmax(torch.tensor([2.0, 0.0]), dim=0)[0].item())
    assert result["truncated"] is False
    timings = result["timings_ms"]
    assert set(timings) == {"tokenize", "model", "postprocess", "server_total"}
    assert all(isinstance(value, (int, float)) and value >= 0 for value in timings.values())
    assert timings["server_total"] >= sum(timings[key] for key in ("tokenize", "model", "postprocess"))


@pytest.mark.parametrize("case", ["empty", "whitespace", "over_character_limit"])
def test_invalid_edge_cases(service, edge_cases, case):
    client, _, model = service
    response = client.post("/classify", json={"text": edge_cases[case]})
    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"]
    assert model.inputs == []


@pytest.mark.parametrize("payload", [{}, {"text": 12}, {"text": None}, {"text": ["good"]}])
def test_missing_or_nonstring_text(service, payload):
    client, _, model = service
    response = client.post("/classify", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"]
    assert model.inputs == []


@pytest.mark.parametrize("case", ["emoji", "accented", "digits"])
def test_unicode_and_digits(service, edge_cases, case):
    client, _, _ = service
    assert client.post("/classify", json={"text": edge_cases[case]}).status_code == 200


@pytest.mark.parametrize("count,truncated", [(256, False), (257, True)])
def test_exact_fake_token_boundaries(service, count, truncated):
    client, tokenizer, model = service
    text = " ".join(["good"] * (count - 2))
    assert len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]) == count
    response = client.post("/classify", json={"text": text})
    assert response.status_code == 200
    assert response.json()["truncated"] is truncated
    assert len(model.inputs[-1]) == 256
    assert model.inputs[-1][0] == 101 and model.inputs[-1][-1] == 102


def test_failed_initialization_is_not_healthy():
    def fail():
        raise RuntimeError("Local model initialization failed")

    application = create_app(fail)
    with pytest.raises(RuntimeError, match="Local model initialization failed"):
        with TestClient(application):
            pass
    # A client that skips startup still cannot receive a healthy response.
    client = TestClient(application)
    assert client.get("/health").status_code == 503
    assert client.post("/classify", json={"text": "good"}).status_code == 503


def test_missing_model_directory_fails_before_transformers_loading(monkeypatch):
    monkeypatch.delenv("MODEL_DIR", raising=False)
    with pytest.raises(RuntimeError, match="MODEL_DIR must explicitly name"):
        with TestClient(create_app()):
            pass
    monkeypatch.setenv("MODEL_DIR", str(Path(__file__)))
    with pytest.raises(RuntimeError, match="Local MODEL_DIR does not exist"):
        with TestClient(create_app()):
            pass


def test_unknown_model_label_mapping_is_rejected():
    model = FakeModel()
    model.config.id2label = {0: "LABEL_0", 1: "LABEL_1"}
    with pytest.raises(ValueError, match="Model configuration must map"):
        V0Backend(FakeTokenizer(), model)
