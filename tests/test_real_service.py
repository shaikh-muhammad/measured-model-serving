"""Local-only correctness checks requiring the pinned, prepared model.

Excluded from minimal CI. No model download, timing trial, or benchmark occurs.
Lightweight-tokenizer equivalence remains a later build-plan step.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest
import torch

from src.app import create_app

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model-cache" / "v0"
SENTENCES = (
    "The support reply was clear and helpful.",
    "The help page left the setup steps unclear.",
    "The update worked well but its explanation was confusing.",
    "The checkout guide was useful and easy to follow.",
    "The support process was slow and disappointing.",
    "The restart instructions solved the setup problem.",
    "The service update was useful despite a frustrating delay.",
    "The help page repeated confusing instructions.",
    "The support reply explained the issue in simple words.",
    "The setup guide was clear but the checkout step was unhelpful.",
)


@pytest.fixture(scope="module")
def service():
    required = ("config.json", "model.safetensors", "tokenizer_config.json", "vocab.txt")
    if not all((MODEL_DIR / name).is_file() for name in required):
        pytest.fail("Pinned local model is missing; run make setup explicitly before these tests")
    with patch.dict(os.environ, {
        "MODEL_DIR": str(MODEL_DIR), "MODEL_THREADS": "4",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
    }):
        with TestClient(create_app()) as client:
            yield client, client.app.state.backend


def check_response(response):
    assert response.status_code == 200
    result = response.json()
    assert set(result) == {"label", "score", "truncated", "timings_ms"}
    assert result["label"] in {"POSITIVE", "NEGATIVE"}
    assert 0 <= result["score"] <= 1
    assert isinstance(result["truncated"], bool)
    assert set(result["timings_ms"]) == {"tokenize", "model", "postprocess", "server_total"}
    assert all(isinstance(value, (int, float)) and value >= 0
               for value in result["timings_ms"].values())
    return result


def test_real_local_model_health_and_configuration(service):
    client, backend = service
    assert client.get("/health").json() == {"status": "ok", "version": "v0"}
    assert backend.model.config.model_type == "distilbert"
    assert not backend.model.training
    assert next(backend.model.parameters()).device.type == "cpu"
    assert torch.get_num_threads() == 4
    text = SENTENCES[0]
    inputs = backend.tokenizer(text, add_special_tokens=True, return_tensors="pt")
    with torch.inference_mode():
        probabilities = torch.softmax(backend.model(**inputs).logits, dim=-1)[0]
    index = int(probabilities.argmax().item())
    result = check_response(client.post("/classify", json={"text": text}))
    assert result["label"] == backend.model.config.id2label[index]
    assert result["score"] == pytest.approx(probabilities[index].item())


def test_ten_invented_sentences(service):
    client, _ = service
    for text in SENTENCES:
        result = check_response(client.post("/classify", json={"text": text}))
        assert result["truncated"] is False


@pytest.mark.parametrize("case,status", [
    ("empty", 422), ("whitespace", 422), ("over_character_limit", 422),
    ("emoji", 200), ("accented", 200), ("over_token_limit", 200), ("digits", 200),
])
def test_existing_real_edge_cases(service, case, status):
    client, backend = service
    cases = {row["id"]: row["text"] for row in
             (json.loads(line) for line in (ROOT / "tests/edge_cases.jsonl").read_text().splitlines())}
    text = cases[case]
    response = client.post("/classify", json={"text": text})
    assert response.status_code == status
    if status == 200:
        result = check_response(response)
        count = len(backend.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
        assert result["truncated"] is (count > 256)
        if case == "over_token_limit":
            assert count > 256


@pytest.mark.parametrize("target,truncated", [(256, False), (257, True)])
def test_exact_real_boundaries_and_model_input(service, target, truncated):
    client, backend = service
    for repetitions in range(1, 258):
        text = " ".join(["good"] * repetitions)
        full_ids = backend.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        if len(full_ids) == target:
            break
    else:
        pytest.fail(f"Could not construct exactly {target} real tokens")
    assert len(text) <= 2000
    plain_ids = backend.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    assert len(full_ids) == len(plain_ids) + 2 == target
    seen = []

    def observe(module, args, kwargs):
        ids = kwargs["input_ids"]
        seen.append(ids.tolist()[0])
        assert ids.shape[1] <= 256

    hook = backend.model.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        result = check_response(client.post("/classify", json={"text": text}))
    finally:
        hook.remove()
    assert result["truncated"] is truncated
    assert len(seen) == 1 and len(seen[0]) == 256
    assert seen[0][0] == backend.tokenizer.cls_token_id
    assert seen[0][-1] == backend.tokenizer.sep_token_id
