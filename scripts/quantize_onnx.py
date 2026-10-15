"""One offline signed dynamic-int8 quantization and complete correctness check.

After sourcing scripts/env.sh:
PYTHONDONTWRITEBYTECODE=1 python scripts/quantize_onnx.py --quantize-and-check

No performance measurements. Only verified archived first-attempt output paths
may be reused. Quantization is called once with MatMul-only QInt8; any corrected
failure is preserved without tuning or retrying.
"""

from collections import Counter
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

from scripts.verify_tokenizer import (StandaloneEncoder, array_hash, no_network,
                                      read_dataset, require, sha256)

ort.disable_telemetry_events()
SOURCE = ROOT / 'model-cache/v1/model.onnx'
TOKENIZER = ROOT / 'model-cache/v1/tokenizer.json'
OUTPUT = ROOT / 'model-cache/v2/model.onnx'
REPORT = ROOT / 'results/parity/v2-quantization.json'
FAILED_MODEL = ROOT / 'model-cache/v2/default-mixed-weight.onnx'
FAILED_REPORT = ROOT / 'results/parity/v2-quantization-default-failure.json'
FAILED_MODEL_HASH = '54ba963b1ebdc0798c8386969841bc21f73f885cf57fc92643e87905a40d65ff'
FAILED_REPORT_HASH = '0a67e0f112be8f30c1aabc19a58fcf72ea637559575d2890e971f070ae09bcef'
HEAD = '27032e2231cb77e75e50c769814d760cd9451012'
MODEL = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'
REVISION = '714eb0fa89d2f80546fda750413ed43d93601a13'
SOURCE_HASH = 'dda17f29d4032bc0eecd4333c9be4fe82c6ea72720a770f88c8e1fbbb47779c1'
TOKENIZER_HASH = '01c420969d5dee75248f2c1b1bf5d834f40f1de1427aa0b9fc93c61055d809e0'
QUALITY_FLOOR = 0.99
LABELS = {0: 'NEGATIVE', 1: 'POSITIVE'}


def baseline_labels(rows):
    expected = {row['id'] for row in rows}
    reference, hashes = None, {}
    paths = sorted((ROOT / 'results/v0-baseline').glob('attempt-*.json'))
    require(len(paths) == 6, 'Exactly six official baseline repetitions required')
    for path in paths:
        relative = str(path.relative_to(ROOT))
        committed = subprocess.check_output(['git', 'show', 'HEAD:' + relative], cwd=ROOT)
        require(path.read_bytes() == committed, 'Original baseline changed: ' + relative)
        data = json.loads(committed)
        observed = {row['sentence_id']: row['predicted_label'] for row in data['request_records']}
        require(len(data['request_records']) == len(observed) == 1000 and set(observed) == expected,
                'Baseline ID coverage differs')
        require(data['validity_status'] == 'VALID' and data['actual_model_threads'] == 4,
                'Baseline validity or frozen threads differ')
        require(reference is None or reference == observed, 'Six baseline repetitions disagree')
        reference = observed
        hashes[relative] = sha256(path)
    return reference, hashes


def quantize_once():
    require(sha256(SOURCE) == SOURCE_HASH, 'Original v1 ONNX hash differs')
    require(not OUTPUT.exists(), 'Refusing to overwrite v2 model')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # ModelProto is supported by the inspected pinned API. Its own temporary
    # directory obeys scripts/env.sh; no inferred sibling is written beside v1.
    quantize_dynamic(model_input=onnx.load(str(SOURCE)), model_output=str(OUTPUT),
                     weight_type=QuantType.QInt8, op_types_to_quantize=['MatMul'])
    require(sha256(SOURCE) == SOURCE_HASH, 'Original v1 model changed during quantization')


def graph_evidence(path=OUTPUT):
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    initializers = {item.name: item for item in graph.graph.initializer}
    weights = []
    for item in graph.graph.initializer:
        if item.name.endswith('_quantized'):
            values = numpy_helper.to_array(item)
            weights.append(dict(name=item.name, dtype=onnx.TensorProto.DataType.Name(item.data_type),
                                shape=list(item.dims), minimum=int(values.min()), maximum=int(values.max())))
    nodes = [dict(name=node.name, operation=node.op_type, inputs=list(node.input))
             for node in graph.graph.node if node.op_type in
             {'MatMulInteger', 'DynamicQuantizeLinear', 'QLinearMatMul', 'ConvInteger'}]
    require(bool(weights) and all(item['dtype'] == 'INT8' for item in weights),
            'Signed 8-bit quantized weights could not be verified')
    require(any(node['operation'] == 'DynamicQuantizeLinear' for node in nodes), 'Dynamic quantization absent')
    integer_nodes = [node for node in nodes if node['operation'] == 'MatMulInteger']
    require(bool(integer_nodes), 'No quantized matrix operations')
    require(all(node['inputs'][1] in initializers and
                initializers[node['inputs'][1]].data_type == onnx.TensorProto.INT8 for node in integer_nodes),
            'Integer MatMul weight input is not signed int8')
    connected_weights = {node['inputs'][1] for node in integer_nodes}
    require(connected_weights == {weight['name'] for weight in weights},
            'Quantized weights are not exclusively connected MatMul weights')
    # Dynamic UINT8 activation zero points are not model weights. Reject every
    # unsigned initializer that cannot be identified as an activation zero point.
    activation_zero_points = {node['inputs'][2] for node in integer_nodes if len(node['inputs']) > 2}
    uint8_initializers = [item for item in graph.graph.initializer if item.data_type == onnx.TensorProto.UINT8]
    require(all(item.name in activation_zero_points for item in uint8_initializers),
            'Unsigned quantized model weight detected')
    original = onnx.load(str(SOURCE))
    original_initializers = {item.name: item for item in original.graph.initializer}
    embeddings = []
    for name in ('model.distilbert.embeddings.word_embeddings.weight',
                 'model.distilbert.embeddings.position_embeddings.weight'):
        require(name in initializers and name in original_initializers, 'Original floating embedding is missing')
        source, current = original_initializers[name], initializers[name]
        require(source.data_type == current.data_type == onnx.TensorProto.FLOAT and source.dims == current.dims,
                'Embedding dtype or dimensions changed')
        source_values, current_values = numpy_helper.to_array(source), numpy_helper.to_array(current)
        source_hash = hashlib.sha256(source_values.tobytes()).hexdigest()
        current_hash = hashlib.sha256(current_values.tobytes()).hexdigest()
        require(source_hash == current_hash, 'Original floating embedding values changed')
        require(any(node.op_type == 'Gather' and node.input[0] == name for node in graph.graph.node),
                'Floating embedding is no longer connected to Gather')
        embeddings.append(dict(name=name, dtype='FLOAT', shape=list(current.dims),
                               original_array_sha256=source_hash, current_array_sha256=current_hash,
                               unchanged='PASS'))
    activation_tensors = [dict(name=node.output[index], dtype='UINT8',
                              role='activation' if index == 0 else 'activation_zero_point',
                              evidence='DynamicQuantizeLinear ONNX schema')
                          for node in graph.graph.node if node.op_type == 'DynamicQuantizeLinear'
                          for index in (0, 2)]
    return dict(checker='PASS', signed_weights='PASS', dynamic_quantization='PASS',
                quantized_weights=weights, quantized_nodes=nodes,
                signed_weight_count=len(connected_weights), quantized_matmul_count=len(integer_nodes),
                unsigned_model_weights='PASS: none', floating_embeddings=embeddings,
                floating_embedding_count=len(embeddings), uint8_activation_tensors=activation_tensors,
                uint8_activation_initializers=[dict(name=item.name, role='activation_zero_point')
                                               for item in uint8_initializers],
                node_counts=dict(sorted(Counter(node.op_type for node in graph.graph.node).items())),
                opsets=[dict(domain=item.domain, version=item.version) for item in graph.opset_import],
                dynamic_activation_type='UINT8; only connected MatMul weights are quantized to INT8; embeddings remain FLOAT')


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    value = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    value.disable_fallback()
    require(value.get_providers() == ['CPUExecutionProvider'], 'Unexpected execution provider')
    inputs = [(node.name, node.type, node.shape) for node in value.get_inputs()]
    outputs = [(node.name, node.type, node.shape) for node in value.get_outputs()]
    require(inputs == [(name, 'tensor(int64)', ['batch', 'sequence']) for name in ('input_ids', 'attention_mask')],
            'Incorrect or non-dynamic ONNX inputs')
    require(outputs == [('logits', 'tensor(float)', ['batch', 2])], 'Incorrect or non-dynamic ONNX logits')
    return value


def encoded_inputs(encoder, rows):
    encoded = encoder.encode([row['text'] for row in rows], padding=len(rows) > 1, truncation=False)
    require(not any(encoded['exceeds_token_limit']), 'Dataset or representative unexpectedly exceeds 256 tokens')
    return encoded, {name: np.asarray(encoded[name], dtype=np.int64)
                     for name in ('input_ids', 'attention_mask')}


def compare(v1, v2, encoder, rows, baseline=None):
    encoded, arrays = encoded_inputs(encoder, rows)
    # The same arrays object is passed to both sessions, without retokenizing.
    left = v1.run(['logits'], arrays)[0]
    right = v2.run(['logits'], arrays)[0]
    require(left.shape == right.shape == (len(rows), 2), 'Invalid classifier output shape')
    require(np.isfinite(left).all() and np.isfinite(right).all(), 'Nonfinite logits')
    records = []
    for position, row in enumerate(rows):
        v1_label, v2_label = LABELS[int(left[position].argmax())], LABELS[int(right[position].argmax())]
        record = dict(sentence_id=row['id'], batch_position=position,
                      length_bucket=row.get('length_bucket', 'representative'),
                      actual_token_length=encoded['actual_token_counts'][position], truncated=False,
                      input_ids=encoded['input_ids'][position], attention_mask=encoded['attention_mask'][position],
                      v1_logits=left[position].tolist(), v2_logits=right[position].tolist(),
                      v1_label=v1_label, v2_label=v2_label, v2_matches_v1=v1_label == v2_label,
                      max_absolute_logit_difference=float(np.max(np.abs(left[position].astype(np.float64) - right[position]))))
        if baseline is not None:
            record.update(v0_label=baseline[row['id']], v2_matches_v0=v2_label == baseline[row['id']],
                          v1_matches_v0=v1_label == baseline[row['id']])
        records.append(record)
    return dict(sentence_ids=[row['id'] for row in rows], records=records,
                input_shape=list(arrays['input_ids'].shape), output_shape=list(left.shape),
                shared_array_sha256={name: array_hash(encoded[name]) for name in ('input_ids', 'attention_mask')},
                padded_tokens=sum(mask.count(0) for mask in encoded['attention_mask']),
                finite_logits='PASS', dynamic_shapes='PASS')


def summarize(batches):
    records = [row for batch in batches for row in batch['records']]
    mismatches_v1 = [row for row in records if not row['v2_matches_v1']]
    mismatches_v0 = [row for row in records if row.get('v2_matches_v0') is False]
    v1_baseline = [row for row in records if row.get('v1_matches_v0') is False]
    buckets = sorted({row['length_bucket'] for row in records})
    has_baseline = all('v0_label' in row for row in records)
    agreement = (len(records) - len(mismatches_v0)) / len(records) if has_baseline else None
    return dict(labels_checked=len(records), unique_ids=len({row['sentence_id'] for row in records}),
                v0_labels_checked=len(records) if has_baseline else 0,
                v2_vs_v0_agreement_percentage=100 * agreement if has_baseline else None,
                v2_vs_v1_agreement_percentage=100 * (len(records) - len(mismatches_v1)) / len(records),
                v1_vs_v0_agreement_percentage=100 * (len(records) - len(v1_baseline)) / len(records) if has_baseline else None,
                v2_vs_v0_mismatches=len(mismatches_v0), v2_vs_v1_mismatches=len(mismatches_v1),
                v0_disagreements=mismatches_v0, v1_disagreements=mismatches_v1,
                mismatch_counts_by_bucket={bucket: sum(row['length_bucket'] == bucket for row in mismatches_v0) for bucket in buckets},
                v1_mismatch_counts_by_bucket={bucket: sum(row['length_bucket'] == bucket for row in mismatches_v1) for bucket in buckets},
                worst_logit_difference=max(row['max_absolute_logit_difference'] for row in records),
                quality=('PASS' if agreement >= QUALITY_FLOOR else 'FAIL') if has_baseline else 'NOT_APPLICABLE')


def batch_consistency(results):
    groups = {size: {row['sentence_id']: row for batch in value['batches'] for row in batch['records']}
              for size, value in results.items()}
    return {version: [dict(sentence_id=identity, length_bucket=groups['1'][identity]['length_bucket'],
                          batch_1_label=groups['1'][identity][version + '_label'],
                          batch_4_label=groups['4'][identity][version + '_label'])
                      for identity in groups['1'] if groups['1'][identity][version + '_label'] !=
                      groups['4'][identity][version + '_label']] for version in ('v1', 'v2')}


def write_report(report):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open('x') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def run():
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip() == HEAD,
            'Git HEAD differs')
    require(sha256(SOURCE) == SOURCE_HASH and sha256(TOKENIZER) == TOKENIZER_HASH, 'Source artifact differs')
    require(sha256(FAILED_MODEL) == FAILED_MODEL_HASH and sha256(FAILED_REPORT) == FAILED_REPORT_HASH,
            'Failed attempt archives are missing or changed')
    require(OUTPUT.is_file() and REPORT.is_file() and
            sha256(OUTPUT) == FAILED_MODEL_HASH and sha256(REPORT) == FAILED_REPORT_HASH,
            'Only the verified archived first-attempt output paths may be reused')
    rows = read_dataset()
    labels, hashes = baseline_labels(rows)
    report = dict(git_head=HEAD, model_name=MODEL, pinned_revision=REVISION,
                  dataset_sha256=sha256(ROOT / 'data/sentences.jsonl'), dataset_seed=20261010,
                  dataset_ids=[row['id'] for row in rows], tokenizer_sha256=TOKENIZER_HASH,
                  tokenizer_path=str(TOKENIZER), source_v1_sha256=SOURCE_HASH, source_v1_size_bytes=SOURCE.stat().st_size,
                  quantization_api='onnxruntime.quantization.quantize_dynamic',
                  quantization_options=dict(weight_type='QuantType.QInt8', op_types_to_quantize=['MatMul'],
                                            other_arguments='pinned API defaults'),
                  quantization_signature=str(inspect.signature(quantize_dynamic)),
                  baseline_source_sha256=hashes, model_threads=4, inter_op_threads=1,
                  requested_provider='CPUExecutionProvider', provider=None, quality_floor=QUALITY_FLOOR,
                  versions=dict(python=platform.python_version(), onnx=onnx.__version__, onnxruntime=ort.__version__),
                  representative_cases=[], full_dataset={}, errors=[], quality='NOT_EVALUATED', correctness='FAIL',
                  archived_first_failure=dict(model_path=str(FAILED_MODEL.relative_to(ROOT)), model_sha256=FAILED_MODEL_HASH,
                    evidence_path=str(FAILED_REPORT.relative_to(ROOT)), evidence_sha256=FAILED_REPORT_HASH,
                    configuration=dict(weight_type='QuantType.QInt8', op_types_to_quantize=None),
                    outcome='REJECTED: signed-weight verification failed; label quality was not evaluated',
                    quantized_weight_types=dict(INT8=38, UINT8=2)),
                  report_kind='Correctness evidence, not an official benchmark')
    # Reuse is explicitly authorized only after both byte-identical archives and
    # original output identities have been checked. Corrected outputs are refused.
    OUTPUT.unlink()
    REPORT.unlink()
    try:
        with no_network():
            quantize_once()
            report.update(v2_sha256=sha256(OUTPUT), v2_size_bytes=OUTPUT.stat().st_size, graph=graph_evidence())
            v1, v2 = session(SOURCE), session(OUTPUT)
            report['provider'] = v2.get_providers()[0]
            encoder = StandaloneEncoder(TOKENIZER)
            texts = [' '.join((['good', 'bad', 'movie'] * 66)[:length - 2]) for length in (5, 20, 100, 200)]
            for index, length in enumerate((5, 20, 100, 200)):
                for size in (1, 4):
                    selected = [texts[index]] if size == 1 else [texts[index]] + [texts[i] for i in range(4) if i != index]
                    batch = compare(v1, v2, encoder, [dict(id=f'representative-{length}-{i}', text=text) for i, text in enumerate(selected)])
                    report['representative_cases'].append(dict(target_tokens=length, batch_size=size, batch=batch,
                                                              summary=summarize([batch])))
            for size in (1, 4):
                batches = []
                for start in range(0, len(rows), size):
                    batches.append(compare(v1, v2, encoder, rows[start:start + size], labels))
                    if (start + size) % 100 == 0:
                        print(f'Correctness batch size {size}: {start + size}/1000 IDs checked', flush=True)
                require([row['sentence_id'] for batch in batches for row in batch['records']] == report['dataset_ids'],
                        'Dataset ID mapping differs')
                summary = summarize(batches)
                report['full_dataset'][str(size)] = dict(batches=batches, summary=summary, coverage='PASS')
                print(f'Batch size {size}: v2/v0 {summary["v2_vs_v0_agreement_percentage"]}%; v2/v1 {summary["v2_vs_v1_agreement_percentage"]}%; mismatches {summary["mismatch_counts_by_bucket"]}; worst logit difference {summary["worst_logit_difference"]}', flush=True)
            report['batch_size_disagreements'] = batch_consistency(report['full_dataset'])
            report['quality'] = 'PASS' if all(value['summary']['quality'] == 'PASS' for value in report['full_dataset'].values()) else 'FAIL'
            report['correctness'] = 'PASS'
            require(sha256(SOURCE) == SOURCE_HASH and sha256(TOKENIZER) == TOKENIZER_HASH, 'Original artifacts changed')
            require(sha256(FAILED_MODEL) == FAILED_MODEL_HASH and sha256(FAILED_REPORT) == FAILED_REPORT_HASH,
                    'Archived failure changed')
    except Exception as error:
        report['errors'].append(repr(error))
        if OUTPUT.is_file():
            report.update(v2_sha256=sha256(OUTPUT), v2_size_bytes=OUTPUT.stat().st_size)
        write_report(report)
        raise
    write_report(report)
    print(f'{report["quality"]}: v2 SHA-256 {report["v2_sha256"]}; size {report["v2_size_bytes"]} bytes', flush=True)
    print(f'Evidence: {REPORT.relative_to(ROOT)}; SHA-256 {sha256(REPORT)}', flush=True)
    require(report['quality'] == 'PASS', 'Registered 99% quality floor failed; evidence preserved; stop before image build')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quantize-and-check', action='store_true', required=True,
                        help='Quantize once with QInt8, then compare all 1000 IDs at batch sizes 1 and 4')
    parser.parse_args()
    run()
