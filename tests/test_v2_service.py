"""Offline HTTP correctness using the existing corrected model; no benchmarks."""

import hashlib
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
from src.app_v2 import ClassifyRequest, ClassifyResponse, Timings, create_app
from src.onnx_backend import ARTIFACT_HASHES, OnnxBackend
from src.onnx_backend_v2 import OnnxBackendV2, V2_MODEL_SHA256, ort

ROOT = Path(__file__).resolve().parents[1]
TEST_IDS = ['sentence-0001', 'sentence-0100', 'sentence-0200', 'sentence-0400',
            'sentence-0401', 'sentence-0500', 'sentence-0600', 'sentence-0800',
            'sentence-0801', 'sentence-1000']
ENV = {'MODEL_DIR': str(ROOT / 'model-cache/v2'), 'MODEL_THREADS': '4',
       'TOKENIZER_PATH': str(ROOT / 'model-cache/v1/tokenizer.json')}


def saved_cases():
    dataset = {r['id']: r for r in map(json.loads, (ROOT / 'data/sentences.jsonl').read_text().splitlines())}
    labels = None
    paths = sorted((ROOT / 'results/v0-baseline').glob('attempt-*.json'))
    assert len(paths) == 6
    for path in paths:
        committed = subprocess.check_output(['git', 'show', 'HEAD:' + str(path.relative_to(ROOT))], cwd=ROOT)
        assert committed == path.read_bytes()
        observed = {r['sentence_id']: r['predicted_label'] for r in json.loads(committed)['request_records']}
        assert len(observed) == 1000 and (labels is None or labels == observed)
        labels = observed
    report = ROOT / 'results/parity/v2-quantization.json'
    assert hashlib.sha256(report.read_bytes()).hexdigest() == '20b42bd1b61e57ab18c4f836e3bb1eda3a5c07dcbba89669e1b4fa096bbe221f'
    evidence = json.loads(report.read_text())
    predictions = {r['sentence_id']: r['v2_label'] for b in evidence['full_dataset']['1']['batches'] for r in b['records']}
    return [(dataset[i], labels[i], predictions[i]) for i in TEST_IDS]


@pytest.fixture(scope='module')
def cases():
    return saved_cases()


@pytest.fixture(scope='module')
def service():
    with patch.dict(os.environ, ENV), no_network(), \
         patch.object(ort, 'InferenceSession', wraps=ort.InferenceSession) as initialize:
        with TestClient(create_app()) as client:
            yield client, client.app.state.backend, initialize


def check_response(response):
    assert response.status_code == 200
    value = response.json()
    assert set(value) == {'label', 'score', 'truncated', 'timings_ms'}
    assert value['label'] in {'POSITIVE', 'NEGATIVE'}
    assert math.isfinite(value['score']) and 0 <= value['score'] <= 1
    assert isinstance(value['truncated'], bool)
    assert set(value['timings_ms']) == {'tokenize', 'model', 'postprocess', 'server_total'}
    assert all(math.isfinite(v) and v >= 0 for v in value['timings_ms'].values())
    assert value['timings_ms']['server_total'] >= value['timings_ms']['model']
    return value


def test_health_single_reused_cpu_session_four_threads(service):
    client, backend, initialize = service
    assert client.get('/health').json() == {'status': 'ok', 'version': 'v2'}
    assert initialize.call_count == 1
    assert backend.session.get_providers() == ['CPUExecutionProvider']
    assert backend.session.get_session_options().intra_op_num_threads == 4
    assert backend.session.get_session_options().inter_op_num_threads == 1
    original = backend.session
    check_response(client.post('/classify', json={'text': 'A good movie.'}))
    assert backend.session is original and initialize.call_count == 1
    assert OnnxBackendV2.classify is OnnxBackend.classify


@pytest.mark.parametrize('index', range(10))
def test_ten_saved_v2_and_committed_v0_predictions(service, cases, index):
    row, v0, v2 = cases[index]
    result = check_response(service[0].post('/classify', json={'text': row['text']}))
    assert result['label'] == v2 == v0
    assert result['truncated'] is False


@pytest.mark.parametrize('case,status', [('empty', 422), ('whitespace', 422), ('over_character_limit', 422),
                                       ('emoji', 200), ('accented', 200), ('over_token_limit', 200), ('digits', 200)])
def test_existing_edge_cases(service, case, status):
    rows = {r['id']: r for r in map(json.loads, (ROOT / 'tests/edge_cases.jsonl').read_text().splitlines())}
    text = rows[case]['text']
    response = service[0].post('/classify', json={'text': text})
    assert response.status_code == status
    if status == 200:
        count = service[1].encoder.encode([text])['actual_token_counts'][0]
        assert check_response(response)['truncated'] is (count > 256)
    else:
        assert isinstance(response.json()['detail'], list)


@pytest.mark.parametrize('body', [{}, {'text': None}, {'text': 123}, {'text': True}, {'text': []}, {'text': {}}])
def test_invalid_input_error_schema_matches_v0_v1(service, body):
    from src.app import create_app as v0
    from src.app_v1 import create_app as v1
    with TestClient(v0(backend_factory=object)) as a, TestClient(v1(backend_factory=object)) as b:
        actual = service[0].post('/classify', json=body)
        previous = [a.post('/classify', json=body), b.post('/classify', json=body)]
    assert actual.status_code == 422
    assert all(r.status_code == 422 and r.json() == actual.json() for r in previous)


@pytest.mark.parametrize('count,truncated', [(256, False), (257, True)])
def test_exact_token_boundary_and_full_length_truncation_flag(service, count, truncated):
    text = ' '.join(['good'] * (count - 2))
    encoded = service[1].encoder.encode([text], truncation=True)
    assert encoded['actual_token_counts'] == [count]
    assert len(encoded['input_ids'][0]) == 256
    assert encoded['input_ids'][0][0] == 101 and encoded['input_ids'][0][-1] == 102
    assert check_response(service[0].post('/classify', json={'text': text}))['truncated'] is truncated


def test_genuine_softmax_confidence(service, cases):
    text = cases[0][0]['text']
    encoded = service[1].encoder.encode([text], truncation=True)
    inputs = {name: np.asarray(encoded[name], dtype=np.int64) for name in ('input_ids', 'attention_mask')}
    values = service[1].session.run(['logits'], inputs)[0][0].astype(np.float64)
    probabilities = np.exp(values - values.max())
    expected = float((probabilities / probabilities.sum()).max())
    assert check_response(service[0].post('/classify', json={'text': text}))['score'] == pytest.approx(expected, abs=1e-12)


def test_public_schemas_and_previous_versions_unchanged():
    from src import app as v0, app_v1 as v1
    for module in (v0, v1):
        for model in (ClassifyRequest, ClassifyResponse, Timings):
            assert model.model_json_schema() == getattr(module, model.__name__).model_json_schema()
    for module, version in ((v0, 'v0'), (v1, 'v1')):
        with TestClient(module.create_app(backend_factory=object)) as client:
            assert client.get('/health').json() == {'status': 'ok', 'version': version}


def test_no_large_libraries_or_network_in_actual_inference_path():
    code = """import importlib.abc,sys,socket
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','transformers'}: raise RuntimeError(fullname)
def deny(*a,**k): raise RuntimeError('Network prohibited')
socket.socket.connect=socket.socket.connect_ex=deny
sys.meta_path.insert(0,Deny())
from src.app_v2 import app
from src.onnx_backend_v2 import OnnxBackendV2
backend=OnnxBackendV2.from_environment()
assert backend.classify('A good movie.')['label'] in {'POSITIVE','NEGATIVE'}
assert not any(n.split('.')[0] in {'torch','transformers'} for n in sys.modules)
print('PASS: offline v2 inference without torch/transformers')
"""
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=dict(os.environ, **ENV), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('artifact', ['model.onnx', 'tokenizer.json'])
def test_changed_artifact_rejected_before_session_creation(tmp_path, artifact):
    (tmp_path / artifact).write_bytes(b'incorrect local artifact')
    paths = {'MODEL_DIR': str(tmp_path), 'TOKENIZER_PATH': str(tmp_path / 'tokenizer.json'), 'MODEL_THREADS': '4'}
    if artifact == 'tokenizer.json':
        (tmp_path / 'model.onnx').symlink_to(ROOT / 'model-cache/v2/model.onnx')
        # A symlink is independently rejected; use a mocked digest to reach the tokenizer check.
        with patch('src.onnx_backend_v2.Path.is_symlink', return_value=False), \
             patch('src.onnx_backend_v2.hashlib.file_digest') as digest:
            digest.return_value.hexdigest.side_effect = [V2_MODEL_SHA256, 'wrong']
            with patch.dict(os.environ, paths), patch.object(ort, 'InferenceSession') as initialize:
                with pytest.raises(RuntimeError, match='SHA-256 mismatch: tokenizer.json'):
                    OnnxBackendV2.from_environment()
                initialize.assert_not_called()
    else:
        with patch.dict(os.environ, paths), patch.object(ort, 'InferenceSession') as initialize:
            with pytest.raises(RuntimeError, match='SHA-256 mismatch: model.onnx'):
                OnnxBackendV2.from_environment()
            initialize.assert_not_called()


def test_missing_local_artifacts_and_wrong_threads_stop(tmp_path):
    with patch.dict(os.environ, dict(ENV, MODEL_DIR=str(tmp_path))):
        with pytest.raises(RuntimeError, match='Missing regular local artifact'):
            OnnxBackendV2.from_environment()
    with patch.dict(os.environ, dict(ENV, MODEL_THREADS='2')):
        with pytest.raises(RuntimeError, match='Frozen MODEL_THREADS=4'):
            OnnxBackendV2.from_environment()


def test_pinned_identities_preserved():
    for path, expected in [(ROOT / 'model-cache/v2/model.onnx', V2_MODEL_SHA256),
                           (ROOT / 'model-cache/v1/tokenizer.json', ARTIFACT_HASHES['tokenizer.json'])]:
        with path.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected
