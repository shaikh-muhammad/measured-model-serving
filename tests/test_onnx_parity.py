"""Offline correctness tests; full dataset inference runs only in the export CLI.

Saved evidence is independently recalculated here. A module-scoped real-model
fixture exercises dynamic shapes and padding without repeating 1000 sentences.
"""

import copy
import json
import socket
from unittest.mock import patch

import numpy as np
import pytest
import torch

from scripts import export_onnx as parity


@pytest.fixture(scope='module')
def real():
    with parity.no_network():
        tokenizer, model = parity.load_local()
        session, schema = parity.session_and_schema()
        yield tokenizer, model, session, schema


@pytest.fixture(scope='module')
def evidence():
    assert parity.REPORT.is_file(), 'Run the documented export CLI first; no evidence regeneration in tests'
    return json.loads(parity.REPORT.read_text())


def test_local_only_loading_configuration():
    with patch.object(parity, 'sha256', side_effect=lambda p: parity.SOURCE_HASHES[p.name]), \
         patch.object(parity.AutoTokenizer, 'from_pretrained') as tokenizer, \
         patch.object(parity.AutoModelForSequenceClassification, 'from_pretrained') as model:
        parity.load_local()
        expected = dict(local_files_only=True, token=False, trust_remote_code=False)
        tokenizer.assert_called_once_with(str(parity.SOURCE), **expected)
        model.assert_called_once_with(str(parity.SOURCE), **expected)
        model.return_value.to.assert_called_once_with('cpu')
        model.return_value.to.return_value.eval.assert_called_once_with()


def test_missing_cache_stops_without_loading():
    with patch.object(parity, 'SOURCE_HASHES', {'missing-file': 'unused'}), \
         patch.object(parity.AutoTokenizer, 'from_pretrained') as load:
        with pytest.raises(RuntimeError, match='Missing regular local file'):
            parity.load_local()
        load.assert_not_called()


def test_legacy_export_configuration():
    arrays = {name: np.ones((1, 5), dtype=np.int64) for name in ('input_ids', 'attention_mask')}
    with patch.object(torch.onnx, 'export') as export:
        parity.export(torch.nn.Identity(), arrays)
    args, options = export.call_args
    assert options == dict(dynamo=False, opset_version=17, input_names=['input_ids', 'attention_mask'],
                           output_names=['logits'], dynamic_axes=parity.AXES, external_data=False)
    assert args[2] == str(parity.OUTPUT)
    assert all(np.array_equal(t.numpy(), arrays[n]) for t, n in zip(args[1], arrays))


def test_real_opset_dynamic_axes_schema_and_cpu_threads(real, evidence):
    tokenizer, model, session, schema = real
    assert schema == evidence['schema']
    assert session.get_providers() == ['CPUExecutionProvider']
    assert session.get_session_options().intra_op_num_threads == torch.get_num_threads() == 4
    assert not model.training and next(model.parameters()).device.type == 'cpu'
    assert evidence['opset'] == 17 and evidence['dynamo'] is False
    assert evidence['dynamic_axes'] == {n: {str(a): s for a, s in axes.items()} for n, axes in parity.AXES.items()}


@pytest.mark.parametrize('target,index', [(5, 0), (20, 1), (100, 2), (200, 3)])
@pytest.mark.parametrize('size', [1, 4])
def test_real_lengths_shapes_shared_arrays_and_padding(real, target, index, size):
    tokenizer, model, session, _ = real
    texts = parity.representative_texts()
    selected = [texts[index]] if size == 1 else [texts[index]] + [texts[i] for i in range(4) if i != index]
    arrays = parity.encode(tokenizer, selected)
    before = {n: a.copy() for n, a in arrays.items()}
    assert arrays['attention_mask'][0].sum() == target
    for i, text in enumerate(selected):
        plain = tokenizer(text, truncation=False, add_special_tokens=True)['input_ids']
        mask = arrays['attention_mask'][i].astype(bool)
        assert arrays['input_ids'][i][mask].tolist() == plain
        assert np.all(arrays['input_ids'][i][~mask] == tokenizer.pad_token_id)
    if size == 4:
        assert (arrays['attention_mask'] == 0).any()
    rows = [dict(id=str(i), length_bucket='representative') for i in range(size)]
    seen = {}
    def observe(module, args, kwargs):
        seen.update({n: t.numpy().copy() for n, t in kwargs.items()})
    hook = model.register_forward_pre_hook(observe, with_kwargs=True)
    class ObservedSession:
        def run(self, outputs, inputs):
            assert inputs is arrays
            assert all(np.array_equal(inputs[n], seen[n]) for n in inputs)
            return session.run(outputs, inputs)
    try:
        records, batch = parity.compare(model, ObservedSession(), arrays, rows)
    finally:
        hook.remove()
    assert all(np.array_equal(arrays[n], before[n]) for n in arrays)
    assert batch['output_shape'] == [size, 2]
    assert all(r['finite_logits'] and r['labels_agree'] and
               r['max_absolute_logit_difference'] <= 0.001 for r in records)


def test_failure_details_are_preserved():
    row = dict(sentence_id='bad-case', length_bucket='long', numerical_parity='FAIL',
               max_absolute_logit_difference=0.002, labels_agree=False, baseline_agrees=False)
    result = parity.summarize([row])
    assert result['numerical_parity'] == result['label_parity'] == result['baseline_parity'] == 'FAIL'
    assert result['count_exceeding_tolerance'] == result['total_label_mismatches'] == 1
    assert result['disagreements'] == result['numerical_failures'] == result['baseline_mismatches'] == [row]
    assert result['disagreements_by_bucket'] == {'long': 1}


@pytest.mark.parametrize('size', ['1', '4'])
def test_complete_saved_evidence_recalculated(evidence, size):
    dataset = parity.read_dataset()
    expected = {r['id']: r for r in dataset}
    labels, hashes = parity.baseline_labels(dataset)
    assert hashes == evidence['baseline_source_sha256']
    result = evidence['dataset_batches'][size]
    records = result['records']
    assert [r['sentence_id'] for r in records] == [r['id'] for r in dataset]
    assert len(records) == len({r['sentence_id'] for r in records}) == 1000
    for record in records:
        pt, rt = np.array(record['pytorch_logits']), np.array(record['onnx_logits'])
        assert pt.shape == rt.shape == (2,) and np.isfinite(pt).all() and np.isfinite(rt).all()
        assert record['max_absolute_logit_difference'] == float(np.max(np.abs(pt - rt)))
        assert record['max_absolute_logit_difference'] <= 0.001
        assert record['length_bucket'] == expected[record['sentence_id']]['length_bucket']
        mapping = {0: 'NEGATIVE', 1: 'POSITIVE'}
        assert record['pytorch_label'] == mapping[int(pt.argmax())]
        assert record['onnx_label'] == mapping[int(rt.argmax())]
        assert record['labels_agree'] == (record['pytorch_label'] == record['onnx_label'])
        assert record['baseline_label'] == labels[record['sentence_id']]
        assert record['baseline_agrees'] == (record['onnx_label'] == record['baseline_label'])
    assert parity.summarize(records) == result['summary']
    assert result['coverage'] == 'PASS'
    assert [i for b in result['batches'] for i in b['sentence_ids']] == [r['id'] for r in dataset]
    for batch in result['batches']:
        assert batch['output_shape'] == [int(size), 2]
        assert batch['input_shape'][0] == int(size)
        assert set(batch['shared_input_sha256']) == {'input_ids', 'attention_mask'}


def test_saved_lengths_and_hashes(real, evidence):
    tokenizer = real[0]
    lengths = {row['id']: len(tokenizer(row['text'], truncation=False)['input_ids'])
               for row in parity.read_dataset()}
    for result in evidence['dataset_batches'].values():
        assert all(r['actual_token_length'] == lengths[r['sentence_id']] for r in result['records'])
        texts = {row['id']: row['text'] for row in parity.read_dataset()}
        for batch in result['batches']:
            arrays = parity.encode(tokenizer, [texts[i] for i in batch['sentence_ids']])
            assert batch['input_shape'] == list(arrays['input_ids'].shape)
            assert batch['padding_tokens'] == int((arrays['attention_mask'] == 0).sum())
            assert batch['shared_input_sha256'] == {
                name: parity.hashlib.sha256(value.tobytes()).hexdigest()
                for name, value in arrays.items()
            }
    assert evidence['dataset_sha256'] == parity.DATA_HASH
    assert evidence['dataset_seed'] == 20261010
    assert evidence['onnx_sha256'] == parity.sha256(parity.OUTPUT)
    assert evidence['onnx_size_bytes'] == parity.OUTPUT.stat().st_size
    assert evidence['git_head'] == parity.HEAD
    assert evidence['status'] == 'PASS'
    assert len(evidence['representative_cases']) == 8
    for case in evidence['representative_cases']:
        assert case['actual_lengths'][0] == case['target_tokens']
        assert case['batch']['output_shape'] == [case['batch_size'], 2]
        assert case['summary'] == parity.summarize(case['records'])


def test_report_deterministic_and_refuses_overwrite(tmp_path):
    report = {'z': [1, 2], 'a': {'status': 'FAIL'}}
    first, second = tmp_path / 'first.json', tmp_path / 'second.json'
    parity.write_report(report, first)
    parity.write_report(copy.deepcopy(report), second)
    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(FileExistsError):
        parity.write_report({'unrelated': True}, first)
    assert json.loads(first.read_text()) == report


def test_no_network_or_downloads():
    with parity.no_network():
        with pytest.raises(RuntimeError, match='Network connections are prohibited'):
            socket.socket.connect(None, ('127.0.0.1', 9))
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY',
                'HF_HUB_DISABLE_IMPLICIT_TOKEN'):
        assert parity.os.environ[key] == '1'
