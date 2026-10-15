"""Offline tokenizer correctness. No model inference or timing experiments.

The CLI checks the full dataset once. These tests validate its saved comparisons
and exercise real local tokenization on a small set of representative inputs.
"""

import copy
import json
from pathlib import Path
import socket
from unittest.mock import patch

import pytest

from scripts import verify_tokenizer as verify


@pytest.fixture(scope='module')
def real():
    with verify.no_network():
        yield verify.load_reference(), verify.StandaloneEncoder()


@pytest.fixture(scope='module')
def evidence():
    assert verify.REPORT.is_file(), 'Run the documented verification CLI first'
    return json.loads(verify.REPORT.read_text())


def test_local_offline_fast_reference_loading():
    with patch('transformers.AutoTokenizer.from_pretrained') as load:
        load.return_value.is_fast = True
        verify.load_reference()
        load.assert_called_once_with(str(verify.SOURCE), local_files_only=True,
                                     token=False, trust_remote_code=False, use_fast=True)
    assert verify.os.environ['HF_HUB_OFFLINE'] == verify.os.environ['TRANSFORMERS_OFFLINE'] == '1'


def test_missing_local_cache_stops_without_download():
    with patch.object(verify, 'LOCAL_HASHES', {'missing-tokenizer-file': 'unused'}):
        with pytest.raises(RuntimeError, match='Missing regular local tokenizer file'):
            verify.load_reference()


def test_standalone_load_and_dependency_independence(evidence):
    assert verify.Tokenizer.from_file(str(verify.ARTIFACT)).token_to_id('[CLS]') == 101
    result = verify.prove_independence()
    assert result['exit_status'] == 0, result
    assert result['stderr'] == ''
    assert evidence['dependency_independence']['outcome'] == 'PASS'


@pytest.mark.parametrize('padding', [False, True])
def test_real_exact_ids_masks_order_special_tokens_and_padding(real, padding):
    reference, encoder = real
    texts = ['A good movie.', 'good', 'A very bad movie, indeed!', '  CAFÉ 🙂 \n']
    rows = [dict(id=f'case-{i}', text=text) for i, text in enumerate(texts)]
    batch = verify.compare_batch(reference, encoder, rows, padding=padding)
    assert batch['outcome'] == 'PASS'
    assert batch['sentence_ids'] == ['case-0', 'case-1', 'case-2', 'case-3']
    for record in batch['records']:
        assert record['batch_position'] == int(record['sentence_id'][-1])
        assert record['input_ids'] == record['reference_input_ids']
        assert record['attention_mask'] == record['reference_attention_mask']
        nonpad = [identity for identity, mask in zip(record['input_ids'], record['attention_mask']) if mask]
        assert nonpad[0] == reference.cls_token_id and nonpad[-1] == reference.sep_token_id
        assert all(identity == reference.pad_token_id for identity, mask in
                   zip(record['input_ids'], record['attention_mask']) if not mask)
    if padding:
        assert len(set(batch['row_lengths'])) == 1 and batch['padded_token_count'] > 0
    else:
        assert len(set(batch['row_lengths'])) > 1 and batch['padded_token_count'] == 0


@pytest.mark.parametrize('identity', ['empty', 'whitespace', 'over_character_limit', 'emoji', 'accented',
                                    'over_token_limit', 'digits', 'plain', 'punctuation', 'uppercase',
                                    'lowercase', 'spaced', 'unicode'])
def test_real_edge_exact_equality_without_silent_truncation(real, identity):
    reference, encoder = real
    row = next(row for row in verify.edge_rows() if row['id'] == identity)
    record = verify.compare_batch(reference, encoder, [row])['records'][0]
    assert record['outcome'] == 'PASS'
    assert record['actual_token_count'] == len(record['input_ids'])
    assert record['truncated'] is False


@pytest.mark.parametrize('count', [256, 257, 402])
@pytest.mark.parametrize('truncation', [False, True])
def test_real_boundary_and_truncation_metadata(real, count, truncation):
    reference, encoder = real
    row = dict(id=f'boundary-{count}', text=' '.join(['good'] * (count - 2)))
    record = verify.compare_batch(reference, encoder, [row], truncation=truncation)['records'][0]
    assert record['outcome'] == 'PASS'
    assert record['actual_token_count'] == count
    assert len(record['input_ids']) == (min(256, count) if truncation else count)
    assert record['exceeds_token_limit'] is (count > 256)
    assert record['truncated'] is (truncation and count > 256)
    assert record['input_ids'][0] == 101 and record['input_ids'][-1] == 102


def test_encoding_options_do_not_leak_between_calls(real):
    reference, encoder = real
    texts = [' '.join(['good'] * 300), 'good']
    assert encoder.encode(texts, padding=True, truncation=True)['truncated'] == [True, False]
    plain = encoder.encode(texts)
    assert list(map(len, plain['input_ids'])) == [302, 3]
    assert plain['truncated'] == [False, False]
    assert all(all(mask == 1 for mask in row) for row in plain['attention_mask'])
    assert encoder.encode(['A GREAT Movie'])['input_ids'] == encoder.encode(['a great movie'])['input_ids']


@pytest.mark.parametrize('size', ['1', '4'])
def test_saved_all_1000_rows_array_hashes_exact_differences_and_coverage(evidence, size):
    dataset = verify.read_dataset()
    rows = {row['id']: row for row in dataset}
    result = evidence['dataset'][size]
    for batches, summary in [(result['batches'], result['summary'])] + (
            [(result['unpadded_batches'], result['unpadded_summary'])] if size == '4' else []):
        assert verify.summarize(batches) == summary
        assert summary['rows'] == summary['unique_ids'] == summary['exact_matches'] == 1000
        assert summary['mismatches'] == 0
        assert summary['mismatches_by_bucket'] == {'short': 0, 'medium': 0, 'long': 0}
        assert summary['truncated_ids'] == summary['exceeds_token_limit_ids'] == []
        assert [r['sentence_id'] for batch in batches for r in batch['records']] == [row['id'] for row in dataset]
        for batch in batches:
            assert len(batch['records']) == int(size)
            assert batch['sentence_ids'] == [r['sentence_id'] for r in batch['records']]
            for position, record in enumerate(batch['records']):
                row = rows[record['sentence_id']]
                assert record['batch_position'] == position
                assert record['length_bucket'] == row['length_bucket']
                assert record['text_sha256'] == verify.hashlib.sha256(row['text'].encode()).hexdigest()
                for name in ('input_ids', 'attention_mask'):
                    assert record[name] == record['reference_' + name]
                    assert record['array_sha256'][name] == verify.array_hash(record[name])
                    assert record['reference_array_sha256'][name] == verify.array_hash(record['reference_' + name])
                    assert record['differences'][name] == []
                assert len(record['input_ids']) == len(record['attention_mask'])
                assert record['actual_token_count'] == sum(record['attention_mask'])
                assert record['metadata_match'] and record['outcome'] == 'PASS'


def test_saved_edge_boundary_configuration_and_integrity(evidence, real):
    assert evidence['outcome'] == evidence['padding_exercised'] == 'PASS'
    assert all(outcome == 'PASS' for outcome in evidence['boundary_checks'].values())
    assert evidence['tokenizer_sha256'] == verify.sha256(verify.ARTIFACT)
    assert evidence['tokenizer_size_bytes'] == verify.ARTIFACT.stat().st_size
    assert evidence['git_head'] == verify.HEAD and evidence['pinned_revision'] == verify.REVISION
    assert evidence['dataset_sha256'] == verify.EXPECTED_HASHES['data/sentences.jsonl']
    config = evidence['configuration']
    assert config['padding']['direction'] == 'right' and config['padding']['pad_id'] == 0
    assert config['special_tokens'] == {'[CLS]': 101, '[MASK]': 103, '[PAD]': 0, '[SEP]': 102, '[UNK]': 100}
    assert config['outcome'] == 'PASS'
    expected_ids = {row['id'] for row in verify.edge_rows()}
    for mode, result in evidence['edge_cases'].items():
        assert result['summary'] == verify.summarize(result['batches'])
        assert result['summary']['mismatches'] == 0
        assert {r['sentence_id'] for b in result['batches'] for r in b['records']} == expected_ids
        for batch in result['batches']:
            for record in batch['records']:
                assert record['input_ids'] == record['reference_input_ids']
                assert record['attention_mask'] == record['reference_attention_mask']
                assert record['truncated'] is (mode == 'service_256' and record['actual_token_count'] > 256)
    assert {case['sentence_id'] for case in evidence['api_validation_cases']} == {'empty', 'whitespace', 'over_character_limit'}
    for name, expected in verify.EXPECTED_HASHES.items():
        assert verify.sha256(verify.ROOT / name) == expected


def test_hashes_are_deterministic_and_keep_array_boundaries():
    values = [[101, 2204, 102], [101, 102, 0]]
    assert verify.array_hash(values) == verify.array_hash(copy.deepcopy(values))
    assert verify.array_hash([1, 23]) != verify.array_hash([12, 3])
    assert verify.array_hash([[1], [2]]) != verify.array_hash([[1, 2]])


def test_all_mismatch_positions_and_metadata_are_reported(real):
    reference, encoder = real
    class CorruptEncoder:
        def encode(self, texts, **options):
            result = encoder.encode(texts, **options)
            result['input_ids'][0][1] = 999
            result['attention_mask'][0].pop()
            result['actual_token_counts'][0] += 1
            return result
    batch = verify.compare_batch(reference, CorruptEncoder(), [dict(id='mismatch-id', text='good')])
    record = batch['records'][0]
    assert record['sentence_id'] == 'mismatch-id' and record['batch_position'] == 0
    assert record['differences']['input_ids'] == [dict(index=1, expected=2204, actual=999)]
    assert record['differences']['attention_mask'] == [dict(index=2, expected=1, actual=None)]
    assert not record['metadata_match'] and record['outcome'] == 'FAIL'
    assert verify.summarize([batch])['disagreements'] == [record]


def test_reports_and_artifacts_refuse_overwrite(tmp_path, real):
    path = tmp_path / 'evidence.json'
    verify.write_report({'b': 2, 'a': 1}, path)
    content = path.read_bytes()
    with pytest.raises(FileExistsError):
        verify.write_report({'other': True}, path)
    assert path.read_bytes() == content
    with pytest.raises(RuntimeError, match='Refusing to overwrite'):
        verify.save_standalone(real[0], verify.ARTIFACT)


def test_network_denial_without_opening_socket():
    with verify.no_network():
        with pytest.raises(RuntimeError, match='Network connections are prohibited'):
            socket.socket.connect(None, ('127.0.0.1', 9))
