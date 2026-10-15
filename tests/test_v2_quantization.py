"""Signed dynamic quantization correctness, using saved full-dataset evidence."""

import copy
import json
from unittest.mock import patch

import numpy as np
import pytest

from scripts import quantize_onnx as q


@pytest.fixture(scope='module')
def evidence():
    assert q.REPORT.is_file(), 'Run the documented quantization CLI first'
    return json.loads(q.REPORT.read_text())


@pytest.fixture(scope='module')
def runtimes():
    with q.no_network():
        yield q.session(q.SOURCE), q.session(q.OUTPUT), q.StandaloneEncoder(q.TOKENIZER)


def test_exact_dynamic_qint8_call_once_without_fallback(tmp_path):
    output = tmp_path / 'unused-model.onnx'
    model = object()
    with patch.object(q, 'OUTPUT', output), patch.object(q, 'sha256', return_value=q.SOURCE_HASH), \
         patch.object(q.onnx, 'load', return_value=model), patch.object(q, 'quantize_dynamic') as quantize:
        q.quantize_once()
    quantize.assert_called_once_with(model_input=model, model_output=str(output),
                                    weight_type=q.QuantType.QInt8, op_types_to_quantize=['MatMul'])


def test_quantization_error_stops_without_retry(tmp_path):
    with patch.object(q, 'OUTPUT', tmp_path / 'unused-model.onnx'), \
         patch.object(q, 'sha256', return_value=q.SOURCE_HASH), patch.object(q.onnx, 'load'), \
         patch.object(q, 'quantize_dynamic', side_effect=RuntimeError('genuine failure')) as quantize:
        with pytest.raises(RuntimeError, match='genuine failure'):
            q.quantize_once()
    assert quantize.call_count == 1


def test_actual_signed_weights_quantized_nodes_checker_and_hash(evidence):
    assert q.graph_evidence() == evidence['graph']
    assert evidence['graph']['signed_weights'] == evidence['graph']['dynamic_quantization'] == 'PASS'
    assert all(row['dtype'] == 'INT8' for row in evidence['graph']['quantized_weights'])
    assert evidence['graph']['node_counts']['MatMulInteger'] > 0
    assert evidence['graph']['node_counts']['DynamicQuantizeLinear'] > 0
    assert evidence['graph']['signed_weight_count'] == evidence['graph']['quantized_matmul_count'] == 38
    assert evidence['graph']['floating_embedding_count'] == 2
    assert all(row['dtype'] == 'FLOAT' and row['original_array_sha256'] == row['current_array_sha256']
               for row in evidence['graph']['floating_embeddings'])
    assert all(row['dtype'] == 'UINT8' and row['role'] in {'activation', 'activation_zero_point'}
               for row in evidence['graph']['uint8_activation_tensors'])
    assert evidence['v2_sha256'] == q.sha256(q.OUTPUT)
    assert evidence['source_v1_sha256'] == q.sha256(q.SOURCE) == q.SOURCE_HASH
    assert evidence['tokenizer_sha256'] == q.sha256(q.TOKENIZER) == q.TOKENIZER_HASH
    assert evidence['v2_size_bytes'] == q.OUTPUT.stat().st_size
    assert evidence['quantization_options']['weight_type'] == 'QuantType.QInt8'
    assert evidence['quantization_options']['op_types_to_quantize'] == ['MatMul']


def test_archived_failure_remains_rejected_and_byte_identical(evidence):
    assert q.sha256(q.FAILED_MODEL) == q.FAILED_MODEL_HASH
    assert q.sha256(q.FAILED_REPORT) == q.FAILED_REPORT_HASH
    failed = json.loads(q.FAILED_REPORT.read_text())
    assert failed['quantization_requirement'] == 'FAIL' and failed['quality'] == 'NOT_EVALUATED'
    assert failed['graph']['quantized_weight_type_counts'] == {'INT8': 38, 'UINT8': 2}
    assert evidence['archived_first_failure']['model_sha256'] == q.FAILED_MODEL_HASH
    assert evidence['archived_first_failure']['evidence_sha256'] == q.FAILED_REPORT_HASH
    assert evidence['archived_first_failure']['configuration']['op_types_to_quantize'] is None


def test_unsigned_embedding_weights_cannot_pass_corrected_verifier():
    with pytest.raises(RuntimeError, match='Signed 8-bit quantized weights could not be verified'):
        q.graph_evidence(q.FAILED_MODEL)


@pytest.mark.parametrize('length', [5, 20, 100, 200])
@pytest.mark.parametrize('size', [1, 4])
def test_real_cpu_dynamic_shapes_finite_logits_and_identical_inputs(runtimes, length, size):
    left, right, encoder = runtimes
    texts = [' '.join(['good'] * (length - 2))] if size == 1 else [
        ' '.join(['good'] * (count - 2)) for count in (length, 5, 20, 200)]
    rows = [dict(id=str(i), text=text) for i, text in enumerate(texts)]
    seen = {}
    class Observe:
        def __init__(self, session, name):
            self.session, self.name = session, name
        def run(self, outputs, inputs):
            seen[self.name] = inputs
            return self.session.run(outputs, inputs)
    result = q.compare(Observe(left, 'v1'), Observe(right, 'v2'), encoder, rows)
    assert seen['v1'] is seen['v2']
    assert result['output_shape'] == [size, 2]
    assert result['records'][0]['actual_token_length'] == length
    assert result['finite_logits'] == result['dynamic_shapes'] == 'PASS'
    assert right.get_providers() == ['CPUExecutionProvider']
    assert right.get_session_options().intra_op_num_threads == 4
    if size == 4:
        assert result['padded_tokens'] > 0


@pytest.mark.parametrize('size', ['1', '4'])
def test_saved_complete_mapping_logits_baseline_labels_and_bucket_counts(evidence, size):
    rows = q.read_dataset()
    buckets = {row['id']: row['length_bucket'] for row in rows}
    baseline, hashes = q.baseline_labels(rows)
    assert hashes == evidence['baseline_source_sha256']
    result = evidence['full_dataset'][size]
    assert q.summarize(result['batches']) == result['summary']
    assert result['summary']['labels_checked'] == result['summary']['unique_ids'] == 1000
    assert [row['sentence_id'] for batch in result['batches'] for row in batch['records']] == [row['id'] for row in rows]
    tokenizer_report = json.loads((q.ROOT / 'results/parity/v1-tokenizer.json').read_text())
    token_rows = {row['sentence_id']: row for batch in tokenizer_report['dataset'][size]['batches'] for row in batch['records']}
    for batch in result['batches']:
        assert batch['output_shape'] == [int(size), 2]
        assert batch['sentence_ids'] == [row['sentence_id'] for row in batch['records']]
        assert batch['shared_array_sha256'] == {
            name: q.array_hash([row[name] for row in batch['records']]) for name in ('input_ids', 'attention_mask')}
        for i, row in enumerate(batch['records']):
            assert row['batch_position'] == i and row['v0_label'] == baseline[row['sentence_id']]
            assert row['length_bucket'] == buckets[row['sentence_id']]
            assert row['actual_token_length'] == sum(row['attention_mask'])
            assert row['input_ids'] == token_rows[row['sentence_id']]['input_ids']
            assert row['attention_mask'] == token_rows[row['sentence_id']]['attention_mask']
            left, right = np.array(row['v1_logits']), np.array(row['v2_logits'])
            assert np.isfinite(left).all() and np.isfinite(right).all() and left.shape == right.shape == (2,)
            assert row['v1_label'] == q.LABELS[int(left.argmax())]
            assert row['v2_label'] == q.LABELS[int(right.argmax())]
            assert row['v2_matches_v0'] == (row['v2_label'] == row['v0_label'])
            assert row['v2_matches_v1'] == (row['v2_label'] == row['v1_label'])
            assert row['max_absolute_logit_difference'] == float(np.max(np.abs(left - right)))
    assert result['summary']['quality'] == ('PASS' if result['summary']['v2_vs_v0_agreement_percentage'] >= 99 else 'FAIL')
    assert sum(result['summary']['mismatch_counts_by_bucket'].values()) == result['summary']['v2_vs_v0_mismatches']


def test_batch_consistency_and_representative_evidence(evidence):
    assert evidence['batch_size_disagreements'] == q.batch_consistency(evidence['full_dataset'])
    assert len(evidence['representative_cases']) == 8
    for case in evidence['representative_cases']:
        assert case['batch']['records'][0]['actual_token_length'] == case['target_tokens']
        assert case['batch']['output_shape'] == [case['batch_size'], 2]
        assert case['summary'] == q.summarize([case['batch']])
        assert case['summary']['v0_labels_checked'] == 0
        assert case['summary']['v2_vs_v0_agreement_percentage'] is None


def test_quality_failure_and_all_disagreements_preserved():
    row = dict(sentence_id='actual-id', length_bucket='long', v0_label='NEGATIVE', v1_label='NEGATIVE',
               v2_label='POSITIVE', v1_matches_v0=True, v2_matches_v0=False, v2_matches_v1=False,
               max_absolute_logit_difference=0.5)
    summary = q.summarize([dict(records=[row])])
    assert summary['quality'] == 'FAIL'
    assert summary['v0_disagreements'] == summary['v1_disagreements'] == [row]
    assert summary['mismatch_counts_by_bucket'] == {'long': 1}
    assert summary['v2_vs_v0_agreement_percentage'] == 0
    assert q.summarize([dict(records=[copy.deepcopy(row)])]) == summary


@pytest.mark.parametrize('matches,outcome', [(99, 'PASS'), (98, 'FAIL')])
def test_original_quality_floor_is_applied_without_relaxation(matches, outcome):
    rows = [dict(sentence_id=str(i), length_bucket='short', v0_label='NEGATIVE',
                 v1_label='NEGATIVE', v2_label='NEGATIVE' if i < matches else 'POSITIVE',
                 v1_matches_v0=True, v2_matches_v0=i < matches, v2_matches_v1=i < matches,
                 max_absolute_logit_difference=0.5) for i in range(100)]
    summary = q.summarize([dict(records=rows)])
    assert summary['quality'] == outcome and q.QUALITY_FLOOR == 0.99
    assert summary['v2_vs_v0_agreement_percentage'] == matches
    assert len(summary['v0_disagreements']) == 100 - matches
