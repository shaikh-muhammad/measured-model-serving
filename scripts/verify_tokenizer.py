"""Standalone tokenizer helpers and offline reference verification.

After sourcing scripts/env.sh, run:
PYTHONDONTWRITEBYTECODE=1 python scripts/verify_tokenizer.py --create-and-verify

StandaloneEncoder imports only tokenizers and standard Python modules. Reference
Transformers is imported lazily by load_reference, never by the encoding path.
No inference, performance measurements, downloads, or API validation occur.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys

from tokenizers import Tokenizer, __version__ as TOKENIZERS_VERSION

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'model-cache/v0'
ARTIFACT = ROOT / 'model-cache/v1/tokenizer.json'
REPORT = ROOT / 'results/parity/v1-tokenizer.json'
HEAD = '27032e2231cb77e75e50c769814d760cd9451012'
MODEL = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'
REVISION = '714eb0fa89d2f80546fda750413ed43d93601a13'
MAX_TOKENS = 256
EXPECTED_HASHES = {
    'data/sentences.jsonl': 'bfe687d4cff16e7d06e8580d09b5947760095b2829fceb0ac9b5fc7f70a99be0',
    'model-cache/v1/model.onnx': 'dda17f29d4032bc0eecd4333c9be4fe82c6ea72720a770f88c8e1fbbb47779c1',
    'results/parity/v1-export.json': 'd736c4170a25aa140b2af3ce4214066aa39afd497d27a09ee9333c51b8070dab',
    'bench/thresholds.yaml': '3f31248914dcd93754995f23002a15bcc8e7329259b6787ff638a4d7bc795439',
}
LOCAL_HASHES = {
    'config.json': '582122c8f414793d131e10022ce9ba04e3811a9da6389137ee2f18665b4f4d15',
    'tokenizer_config.json': '5ab9097b4149371c5fd52b2d6e26cb6f9c07c0d19fcfdda895b1adad6b57c3e0',
    'vocab.txt': '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3',
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_hash(values):
    """SHA-256 of compact UTF-8 JSON integer arrays (including row boundaries)."""
    return hashlib.sha256(json.dumps(values, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@contextmanager
def no_network():
    original, original_ex = socket.socket.connect, socket.socket.connect_ex
    def deny(*args, **kwargs):
        raise RuntimeError('Network connections are prohibited')
    socket.socket.connect = socket.socket.connect_ex = deny
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex = original, original_ex


def load_reference():
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY',
                'HF_HUB_DISABLE_IMPLICIT_TOKEN'):
        os.environ[key] = '1'
    for name, expected in LOCAL_HASHES.items():
        path = SOURCE / name
        require(path.is_file() and not path.is_symlink(), f'Missing regular local tokenizer file: {path}')
        require(sha256(path) == expected, f'Pinned original file mismatch: {name}')
    from transformers import AutoTokenizer
    reference = AutoTokenizer.from_pretrained(str(SOURCE), local_files_only=True,
                                             token=False, trust_remote_code=False, use_fast=True)
    require(reference.is_fast, 'Original fast tokenizer required')
    return reference


def save_standalone(reference, path=ARTIFACT):
    require(not path.exists(), f'Refusing to overwrite tokenizer: {path}')
    # Clone the original backend; preserve the reference object and original files.
    native = Tokenizer.from_str(reference.backend_tokenizer.to_str())
    native.enable_padding(direction=reference.padding_side, pad_id=reference.pad_token_id,
                          pad_token=reference.pad_token, pad_type_id=reference.pad_token_type_id)
    native.enable_truncation(max_length=MAX_TOKENS, direction=reference.truncation_side)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(native.to_str(pretty=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    Tokenizer.from_file(str(path))


class StandaloneEncoder:
    """Inputs and metadata from tokenizer.json alone; each call has fresh settings.

    Padding defaults off. Explicit truncation uses the service's 256-token limit.
    Truncation flags use full pre-truncation counts including special tokens.
    This helper does not enforce HTTP text validation or the character limit.
    """
    def __init__(self, path=ARTIFACT):
        native = Tokenizer.from_file(str(path))
        self.serialized = native.to_str()
        self.padding = native.padding
        self.truncation = native.truncation
        require(self.padding is not None and self.truncation is not None,
                'Standalone artifact must embed padding and truncation configuration')
        require(self.truncation['max_length'] == MAX_TOKENS, 'Incorrect embedded service limit')

    def encode(self, texts, *, padding=False, truncation=False):
        require(bool(texts) and all(isinstance(text, str) for text in texts), 'Nonempty text list required')
        native = Tokenizer.from_str(self.serialized)
        native.no_padding()
        native.no_truncation()
        full = native.encode_batch(texts, add_special_tokens=True)
        counts = [len(row.ids) for row in full]
        if truncation:
            native.enable_truncation(max_length=MAX_TOKENS, direction=self.truncation['direction'],
                                     strategy=self.truncation['strategy'], stride=0)
        if padding:
            native.enable_padding(direction=self.padding['direction'], pad_id=self.padding['pad_id'],
                                  pad_token=self.padding['pad_token'], pad_type_id=self.padding['pad_type_id'])
        encoded = native.encode_batch(texts, add_special_tokens=True) if padding or truncation else full
        return dict(input_ids=[row.ids for row in encoded], attention_mask=[row.attention_mask for row in encoded],
                    actual_token_counts=counts, exceeds_token_limit=[count > MAX_TOKENS for count in counts],
                    truncated=[bool(truncation and count > MAX_TOKENS) for count in counts])


def exact_differences(expected, actual):
    """Preserve all differing indices, including missing or extra integer tokens."""
    return [dict(index=i, expected=expected[i] if i < len(expected) else None,
                 actual=actual[i] if i < len(actual) else None)
            for i in range(max(len(expected), len(actual)))
            if i >= len(expected) or i >= len(actual) or expected[i] != actual[i]]


def compare_batch(reference, encoder, rows, *, padding=False, truncation=False):
    texts = [row['text'] for row in rows]
    expected = reference(texts, add_special_tokens=True, padding=padding, truncation=truncation,
                         **({'max_length': MAX_TOKENS} if truncation else {}))
    actual = encoder.encode(texts, padding=padding, truncation=truncation)
    full = reference(texts, add_special_tokens=True, padding=False, truncation=False)
    require(len(expected['input_ids']) == len(actual['input_ids']) == len(rows), 'Row coverage changed')
    records = []
    for i, row in enumerate(rows):
        differences = {name: exact_differences(expected[name][i], actual[name][i])
                       for name in ('input_ids', 'attention_mask')}
        count = len(full['input_ids'][i])
        expected_truncated = bool(truncation and count > MAX_TOKENS)
        metadata_match = (actual['actual_token_counts'][i] == count and
                          actual['truncated'][i] == expected_truncated and
                          actual['exceeds_token_limit'][i] == (count > MAX_TOKENS))
        agree = not any(differences.values()) and metadata_match
        records.append(dict(sentence_id=row['id'], batch_position=i,
                            length_bucket=row.get('length_bucket', 'edge'),
                            text_sha256=hashlib.sha256(row['text'].encode()).hexdigest(),
                            actual_token_count=count, standalone_full_token_count=actual['actual_token_counts'][i],
                            exceeds_token_limit=count > MAX_TOKENS, truncation_enabled=truncation,
                            truncated=actual['truncated'][i], expected_truncated=expected_truncated,
                            input_ids=actual['input_ids'][i], attention_mask=actual['attention_mask'][i],
                            reference_input_ids=expected['input_ids'][i], reference_attention_mask=expected['attention_mask'][i],
                            array_sha256={name: array_hash(actual[name][i]) for name in ('input_ids', 'attention_mask')},
                            reference_array_sha256={name: array_hash(expected[name][i]) for name in ('input_ids', 'attention_mask')},
                            differences=differences, metadata_match=metadata_match, outcome='PASS' if agree else 'FAIL'))
    return dict(sentence_ids=[row['id'] for row in rows], padding=padding, truncation=truncation,
                row_lengths=[len(ids) for ids in actual['input_ids']],
                shape=[len(rows), len(actual['input_ids'][0])] if padding or len(rows) == 1 else None,
                padded_token_count=sum(mask.count(0) for mask in actual['attention_mask']), records=records,
                outcome='PASS' if all(record['outcome'] == 'PASS' for record in records) else 'FAIL')


def summarize(batches):
    records = [record for batch in batches for record in batch['records']]
    buckets = sorted({record['length_bucket'] for record in records})
    mismatches = [record for record in records if record['outcome'] != 'PASS']
    return dict(rows=len(records), unique_ids=len({record['sentence_id'] for record in records}),
                exact_matches=len(records) - len(mismatches), mismatches=len(mismatches),
                mismatches_by_bucket={bucket: sum(r['length_bucket'] == bucket for r in mismatches) for bucket in buckets},
                exceeds_token_limit_ids=[r['sentence_id'] for r in records if r['exceeds_token_limit']],
                truncated_ids=[r['sentence_id'] for r in records if r['truncated']],
                disagreements=mismatches, outcome='FAIL' if mismatches else 'PASS')


def read_dataset():
    path = ROOT / 'data/sentences.jsonl'
    require(sha256(path) == EXPECTED_HASHES['data/sentences.jsonl'], 'Dataset hash changed')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    require(len(rows) == len({row['id'] for row in rows}) == 1000, 'Expected 1000 unique dataset IDs')
    require(Counter(row['length_bucket'] for row in rows) == Counter(short=400, medium=400, long=200),
            'Dataset bucket counts changed')
    return rows


def edge_rows():
    rows = [json.loads(line) for line in (ROOT / 'tests/edge_cases.jsonl').read_text().splitlines()]
    require(all(set(row) == {'id', 'text'} for row in rows), 'Unexpected edge fixture structure')
    return rows + [dict(id=identity, text=text) for identity, text in (
        ('plain', 'A good movie.'), ('punctuation', "Great?! ... yes; no: isn't it?"),
        ('uppercase', 'A GREAT Movie'), ('lowercase', 'a great movie'),
        ('spaced', '  A\tgood\nmovie.  '), ('unicode', 'CAFÉ naïve 中文 🙂😞'),
        ('boundary-256', ' '.join(['good'] * 254)),
        ('boundary-257', ' '.join(['good'] * 255)),
    )]


def prove_independence():
    code = """import importlib.abc, sys
class DenyLargeLibraries(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'onnxruntime'}:
            raise RuntimeError('Forbidden runtime dependency: ' + fullname)
sys.meta_path.insert(0, DenyLargeLibraries())
from scripts.verify_tokenizer import StandaloneEncoder
encoded = StandaloneEncoder().encode(['A good movie.', 'good'], padding=True)
assert encoded['input_ids'][0] == [101, 1037, 2204, 3185, 1012, 102]
assert encoded['attention_mask'][1] == [1, 1, 1, 0, 0, 0]
assert not any(name.split('.')[0] in {'torch', 'transformers', 'onnxruntime'} for name in sys.modules)
print('PASS: standalone encoding with tokenizers and standard Python only')
"""
    command = [sys.executable, '-c', code]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    return dict(command=command, stdout=result.stdout, stderr=result.stderr, exit_status=result.returncode,
                outcome='PASS' if result.returncode == 0 else 'FAIL')


def write_report(report, path=REPORT):
    serialized = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())


def preflight():
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT).decode().strip()
    require(git('rev-parse', 'HEAD') == HEAD and git('branch', '--show-current') == 'main', 'Git identity mismatch')
    require(not git('diff', 'HEAD') and not git('diff', '--cached'), 'Tracked files changed')
    allowed = {'scripts/export_onnx.py', 'tests/test_onnx_parity.py', 'results/parity/v1-export.json',
               'scripts/verify_tokenizer.py', 'tests/test_tokenizer_parity.py'}
    lines = git('status', '--porcelain', '--untracked-files=all').splitlines()
    require(all(line.startswith('?? ') and line[3:] in allowed for line in lines), 'Unexpected pending files')
    for name, expected in EXPECTED_HASHES.items():
        require(sha256(ROOT / name) == expected, f'Precondition hash mismatch: {name}')
    previous = json.loads((ROOT / 'results/parity/v1-export.json').read_text())
    require(previous['pinned_revision'] == REVISION and previous['model_threads'] == 4 and previous['status'] == 'PASS',
            'Pinned revision or frozen thread choice mismatch')
    require(not ARTIFACT.exists() and not REPORT.exists(), 'Destination already exists')
    return previous


def verify():
    previous = preflight()
    with no_network():
        reference = load_reference()
        save_standalone(reference)
        encoder = StandaloneEncoder()
        special = {token: reference.convert_tokens_to_ids(token) for token in reference.all_special_tokens}
        native = Tokenizer.from_file(str(ARTIFACT))
        configuration = dict(special_tokens=special, padding=encoder.padding, truncation=encoder.truncation,
                             reference_padding_side=reference.padding_side, reference_truncation_side=reference.truncation_side,
                             reference_pad_token_id=reference.pad_token_id,
                             special_tokens_match=all(native.token_to_id(token) == identity for token, identity in special.items()),
                             padding_direction_match=encoder.padding['direction'] == reference.padding_side,
                             pad_token_id_match=encoder.padding['pad_id'] == reference.pad_token_id)
        configuration['outcome'] = 'PASS' if all(configuration[key] for key in
            ('special_tokens_match', 'padding_direction_match', 'pad_token_id_match')) else 'FAIL'
        import transformers
        report = dict(git_head=HEAD, model_name=MODEL, pinned_revision=REVISION,
                      source_directory=str(SOURCE), python_version=platform.python_version(),
                      tokenizers_version=TOKENIZERS_VERSION, transformers_version=transformers.__version__,
                      dataset_sha256=EXPECTED_HASHES['data/sentences.jsonl'], dataset_seed=20261010,
                      tokenizer_sha256=sha256(ARTIFACT), tokenizer_size_bytes=ARTIFACT.stat().st_size,
                      preserved_input_hashes=EXPECTED_HASHES, configuration=configuration,
                      hash_format='SHA-256 of compact UTF-8 JSON arrays: json.dumps(values, separators=(comma,colon))',
                      dataset={}, edge_cases={}, outcome='PASS')
        rows = read_dataset()
        for size in (1, 4):
            batches = [compare_batch(reference, encoder, rows[start:start + size], padding=size == 4)
                       for start in range(0, len(rows), size)]
            summary = summarize(batches)
            require([r['sentence_id'] for batch in batches for r in batch['records']] == [row['id'] for row in rows],
                    'Dataset input order changed')
            unpadded = []
            if size == 4:
                # Explicit unpadded batch comparison covers every sentence as well.
                unpadded = [compare_batch(reference, encoder, rows[start:start + size], padding=False)
                            for start in range(0, len(rows), size)]
            report['dataset'][str(size)] = dict(batches=batches, summary=summary,
                                              unpadded_batches=unpadded,
                                              unpadded_summary=summarize(unpadded) if unpadded else summary,
                                              coverage='PASS')
            print(f'Batch size {size}: {summary["exact_matches"]}/1000 exact; mismatches {summary["mismatches"]}; buckets {summary["mismatches_by_bucket"]}')
        edges = edge_rows()
        for truncation in (False, True):
            batches = [compare_batch(reference, encoder, [row], truncation=truncation) for row in edges]
            batches += [compare_batch(reference, encoder, edges[start:start + 4], padding=True, truncation=truncation)
                        for start in range(0, len(edges), 4)]
            report['edge_cases']['service_256' if truncation else 'no_truncation'] = dict(batches=batches, summary=summarize(batches))
        boundary_records = {r['sentence_id']: r for b in report['edge_cases']['service_256']['batches']
                            if len(b['records']) == 1 for r in b['records']}
        report['boundary_checks'] = {
            'exact_256': 'PASS' if boundary_records['boundary-256']['actual_token_count'] == 256 and
                         len(boundary_records['boundary-256']['input_ids']) == 256 and not boundary_records['boundary-256']['truncated'] else 'FAIL',
            'over_256': 'PASS' if boundary_records['boundary-257']['actual_token_count'] == 257 and
                        len(boundary_records['boundary-257']['input_ids']) == 256 and boundary_records['boundary-257']['truncated'] else 'FAIL',
            'truncation_detection': 'PASS' if all(r['metadata_match'] for value in report['edge_cases'].values()
                                                for b in value['batches'] for r in b['records']) else 'FAIL',
        }
        report['api_validation_cases'] = [dict(sentence_id=row['id'], characters=len(row['text']),
            reason='empty or whitespace-only' if not row['text'].strip() else 'over 2000 characters',
            note='API validation case; tokenizer encoding is not evidence of HTTP 422 behavior')
            for row in edges if not row['text'].strip() or len(row['text']) > 2000]
        report['dependency_independence'] = prove_independence()
        report['padding_exercised'] = 'PASS' if any(batch['padded_token_count'] > 0
            for batch in report['dataset']['4']['batches']) else 'FAIL'
        checks = [configuration['outcome'], *report['boundary_checks'].values(),
                  report['dependency_independence']['outcome'], report['padding_exercised']]
        checks += [result[key]['outcome'] for result in report['dataset'].values() for key in ('summary', 'unpadded_summary')]
        checks += [result['summary']['outcome'] for result in report['edge_cases'].values()]
        report['outcome'] = 'PASS' if all(check == 'PASS' for check in checks) else 'FAIL'
        report['original_source_hashes'] = previous['source_sha256']
        write_report(report)
        print('Special tokens and padding:', configuration['outcome'])
        print('Boundary checks:', json.dumps(report['boundary_checks'], sort_keys=True))
        print('Edge checks:', {name: value['summary']['mismatches'] for name, value in report['edge_cases'].items()})
        print(f'Tokenizer: {sha256(ARTIFACT)}; {ARTIFACT.stat().st_size} bytes')
        print(f'{report["outcome"]}: {REPORT.relative_to(ROOT)}; SHA-256 {sha256(REPORT)}')
        require(report['outcome'] == 'PASS', 'Exact tokenizer parity failed; genuine evidence saved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--create-and-verify', action='store_true', required=True,
                        help='Save original local fast tokenizer and check exact dataset, boundary, and edge parity')
    parser.parse_args()
    verify()
