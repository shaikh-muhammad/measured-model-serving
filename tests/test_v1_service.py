"""Offline v1 integration correctness; no load runner or performance trials."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
import pytest

from scripts.verify_tokenizer import no_network
from src.app_v1 import ClassifyRequest, ClassifyResponse, Timings, create_app
from src.onnx_backend import OnnxBackend

ROOT = Path(__file__).resolve().parents[1]
TEST_IDS = ['sentence-0001', 'sentence-0100', 'sentence-0200', 'sentence-0400',
            'sentence-0401', 'sentence-0500', 'sentence-0600', 'sentence-0800',
            'sentence-0801', 'sentence-1000']


def saved_cases():
    dataset = {row['id']: row for row in
               (json.loads(line) for line in (ROOT / 'data/sentences.jsonl').read_text().splitlines())}
    labels = None
    for path in sorted((ROOT / 'results/v0-baseline').glob('attempt-*.json')):
        data = json.loads(path.read_text())
        observed = {row['sentence_id']: row['predicted_label'] for row in data['request_records']}
        assert len(observed) == 1000 and (labels is None or observed == labels)
        labels = observed
    return [(dataset[identity], labels[identity]) for identity in TEST_IDS]


@pytest.fixture(scope='module')
def service():
    with patch.dict(os.environ, {'MODEL_DIR': str(ROOT / 'model-cache/v1'), 'MODEL_THREADS': '4'}), no_network():
        from src import onnx_backend
        with patch.object(onnx_backend.ort, 'InferenceSession', wraps=onnx_backend.ort.InferenceSession) as initialize:
            with TestClient(create_app()) as client:
                yield client, client.app.state.backend, initialize


def check_response(response):
    assert response.status_code == 200
    value = response.json()
    assert set(value) == {'label', 'score', 'truncated', 'timings_ms'}
    assert value['label'] in {'NEGATIVE', 'POSITIVE'}
    assert isinstance(value['score'], (float, int)) and math.isfinite(value['score']) and 0 <= value['score'] <= 1
    assert isinstance(value['truncated'], bool)
    assert set(value['timings_ms']) == {'tokenize', 'model', 'postprocess', 'server_total'}
    assert all(isinstance(v, (float, int)) and math.isfinite(v) and v >= 0 for v in value['timings_ms'].values())
    assert value['timings_ms']['server_total'] >= value['timings_ms']['model']
    return value


def test_health_cpu_four_threads_and_single_session(service):
    client, backend, initialize = service
    assert client.get('/health').json() == {'status': 'ok', 'version': 'v1'}
    assert backend.session.get_providers() == ['CPUExecutionProvider']
    assert backend.session.get_session_options().intra_op_num_threads == 4
    assert initialize.call_count == 1
    original = backend.session
    for row, label in saved_cases()[:2]:
        assert check_response(client.post('/classify', json={'text': row['text']}))['label'] == label
    assert backend.session is original and initialize.call_count == 1


@pytest.mark.parametrize('case_index', range(10))
def test_ten_actual_dataset_predictions_match_original_v0(service, case_index):
    row, expected = saved_cases()[case_index]
    value = check_response(service[0].post('/classify', json={'text': row['text']}))
    assert value['label'] == expected and value['truncated'] is False


@pytest.mark.parametrize('case,status', [('empty', 422), ('whitespace', 422), ('over_character_limit', 422),
                                       ('emoji', 200), ('accented', 200), ('over_token_limit', 200), ('digits', 200)])
def test_real_edge_fixture(service, case, status):
    rows = {row['id']: row for row in
            (json.loads(line) for line in (ROOT / 'tests/edge_cases.jsonl').read_text().splitlines())}
    row = rows[case]
    response = service[0].post('/classify', json={'text': row['text']})
    assert response.status_code == status
    if status == 200:
        value = check_response(response)
        full_count = service[1].encoder.encode([row['text']])['actual_token_counts'][0]
        assert value['truncated'] is (full_count > 256)
    else:
        assert isinstance(response.json()['detail'], list)


@pytest.mark.parametrize('body', [{}, {'text': None}, {'text': 123}, {'text': True}, {'text': []}, {'text': {}}])
def test_invalid_missing_text_matches_v0_error_schema(service, body):
    from src.app import create_app as create_v0_app
    with TestClient(create_v0_app(backend_factory=object)) as original:
        previous = original.post('/classify', json=body)
        actual = service[0].post('/classify', json=body)
    assert previous.status_code == actual.status_code == 422
    assert previous.json() == actual.json()


@pytest.mark.parametrize('count,truncated', [(256, False), (257, True)])
def test_boundary_inputs_retain_special_tokens_and_correct_metadata(service, count, truncated):
    text = ' '.join(['good'] * (count - 2))
    client, backend, _ = service
    encoded = backend.encoder.encode([text], truncation=True)
    assert encoded['actual_token_counts'] == [count]
    assert len(encoded['input_ids'][0]) == 256
    assert encoded['input_ids'][0][0] == 101 and encoded['input_ids'][0][-1] == 102
    assert check_response(client.post('/classify', json={'text': text}))['truncated'] is truncated


def test_confidence_matches_softmax_of_actual_logits(service):
    client, backend, _ = service
    row, _ = saved_cases()[0]
    encoded = backend.encoder.encode([row['text']], truncation=True)
    inputs = {name: np.array(encoded[name], dtype=np.int64) for name in ('input_ids', 'attention_mask')}
    values = backend.session.run(['logits'], inputs)[0][0].astype(np.float64)
    probabilities = np.exp(values - max(values))
    expected = float(max(probabilities / sum(probabilities)))
    assert check_response(client.post('/classify', json={'text': row['text']}))['score'] == pytest.approx(expected, abs=1e-12)


def test_v0_public_models_and_health_remain_identical():
    from src import app as v0
    assert ClassifyRequest.model_json_schema() == v0.ClassifyRequest.model_json_schema()
    assert ClassifyResponse.model_json_schema() == v0.ClassifyResponse.model_json_schema()
    assert Timings.model_json_schema() == v0.Timings.model_json_schema()
    with TestClient(v0.create_app(backend_factory=object)) as original:
        assert original.get('/health').json() == {'status': 'ok', 'version': 'v0'}


def test_large_libraries_not_imported_in_v1_runtime():
    code = """import importlib.abc, sys
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers'}:
            raise RuntimeError('Forbidden dependency: ' + fullname)
sys.meta_path.insert(0, Deny())
from src.app_v1 import app
from src.onnx_backend import OnnxBackend
backend = OnnxBackend.from_environment()
assert backend.classify('A good movie.')['label'] in {'POSITIVE', 'NEGATIVE'}
assert not any(name.split('.')[0] in {'torch', 'transformers'} for name in sys.modules)
print('PASS: v1 inference without torch or transformers')
"""
    environment = dict(os.environ, MODEL_DIR=str(ROOT / 'model-cache/v1'), MODEL_THREADS='4')
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'PASS' in result.stdout


def test_missing_local_files_and_wrong_threads_stop(tmp_path):
    with patch.dict(os.environ, MODEL_DIR=str(tmp_path), MODEL_THREADS='4'):
        with pytest.raises(RuntimeError, match='Missing regular local artifact'):
            OnnxBackend.from_environment()
    with patch.dict(os.environ, MODEL_THREADS='2'):
        with pytest.raises(RuntimeError, match='Frozen MODEL_THREADS=4'):
            OnnxBackend.from_environment()


def test_nonfinite_logits_rejected(service):
    class InvalidSession:
        def run(self, outputs, inputs):
            return [np.array([[float('nan'), 1.0]])]
    backend = OnnxBackend(service[1].encoder, InvalidSession())
    with pytest.raises(RuntimeError, match='nonfinite ONNX logits'):
        backend.classify('good')
