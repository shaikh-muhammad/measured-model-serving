"""Offline correctness export; no performance measurements.

Run after sourcing scripts/env.sh:
PYTHONDONTWRITEBYTECODE=1 python scripts/export_onnx.py --export-and-check
Writes only model-cache/v1/model.onnx and results/parity/v1-export.json.
Existing destinations are refused. At most two legacy export attempts occur.
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

for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN"):
    os.environ[key] = "1"

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ort.disable_telemetry_events()

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "model-cache/v0"
OUTPUT = ROOT / "model-cache/v1/model.onnx"
REPORT = ROOT / "results/parity/v1-export.json"
HEAD = "27032e2231cb77e75e50c769814d760cd9451012"
MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
REVISION = "714eb0fa89d2f80546fda750413ed43d93601a13"
DATA_HASH = "bfe687d4cff16e7d06e8580d09b5947760095b2829fceb0ac9b5fc7f70a99be0"
TOLERANCE = 0.001
AXES = {"input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"}, "logits": {0: "batch"}}
SOURCE_HASHES = {
    "config.json": "582122c8f414793d131e10022ce9ba04e3811a9da6389137ee2f18665b4f4d15",
    "model.safetensors": "7c3919835e442510166d267fe7cbe847e0c51cd26d9ba07b89a57b952b49b8aa",
    "tokenizer_config.json": "5ab9097b4149371c5fd52b2d6e26cb6f9c07c0d19fcfdda895b1adad6b57c3e0",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()


@contextmanager
def no_network():
    """Deny Python socket connections as an additional offline safeguard."""
    original = socket.socket.connect
    original_ex = socket.socket.connect_ex
    def deny(*args, **kwargs):
        raise RuntimeError("Network connections are prohibited for parity checks")
    socket.socket.connect = deny
    socket.socket.connect_ex = deny
    try:
        yield
    finally:
        socket.socket.connect = original
        socket.socket.connect_ex = original_ex


def load_local():
    for name, expected in SOURCE_HASHES.items():
        path = SOURCE / name
        require(path.is_file() and not path.is_symlink(), f"Missing regular local file: {path}")
        require(sha256(path) == expected, f"Pinned model hash mismatch: {name}")
    require(torch.version.cuda is None, "CPU-only PyTorch required")
    torch.set_num_threads(4)
    options = dict(local_files_only=True, token=False, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(str(SOURCE), **options)
    model = AutoModelForSequenceClassification.from_pretrained(str(SOURCE), **options)
    model = model.to("cpu").eval()
    return tokenizer, model


def read_dataset():
    path = ROOT / "data/sentences.jsonl"
    require(sha256(path) == DATA_HASH, "Dataset hash changed")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    require(len(rows) == len({r['id'] for r in rows}) == 1000, "Dataset ID coverage failed")
    require(Counter(r['length_bucket'] for r in rows) ==
            Counter(short=400, medium=400, long=200), "Dataset buckets changed")
    return rows


def baseline_labels(rows):
    expected_ids = {r['id'] for r in rows}
    reference, hashes = None, {}
    paths = sorted((ROOT / "results/v0-baseline").glob("*.json"))
    require(len(paths) == 8, "Expected six official attempts and two summaries")
    for path in paths:
        relative = str(path.relative_to(ROOT))
        committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
        require(path.read_bytes() == committed, f"Baseline changed: {relative}")
        hashes[relative] = sha256(path)
        data = json.loads(committed)
        if "request_records" not in data:
            continue
        records = data['request_records']
        labels = {r['sentence_id']: r['predicted_label'] for r in records}
        require(len(records) == len(labels) == 1000 and set(labels) == expected_ids,
                "Baseline ID coverage failed")
        require(data['validity_status'] == 'VALID' and data['actual_model_threads'] == 4,
                "Baseline validity/thread count failed")
        require(all(r['success'] for r in records), "Baseline has failed requests")
        require(reference is None or labels == reference, "STOP: baseline repetitions disagree")
        reference = labels
    require(reference is not None, "No official baseline labels")
    return reference, hashes


def encode(tokenizer, texts):
    encoded = tokenizer(texts, padding=True, truncation=False, add_special_tokens=True,
                        return_tensors="np")
    arrays = {name: np.ascontiguousarray(encoded[name], dtype=np.int64)
              for name in ("input_ids", "attention_mask")}
    require(arrays['input_ids'].shape[1] <= 256, "Unexpected input over 256 tokens")
    return arrays


class LogitsOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def export(model, arrays, output=OUTPUT):
    torch.onnx.export(
        LogitsOnly(model).eval(),
        tuple(torch.from_numpy(arrays[name]) for name in ("input_ids", "attention_mask")),
        str(output), dynamo=False, opset_version=17,
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes=AXES, external_data=False,
    )


def session_and_schema(path=OUTPUT):
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    require([(x.domain, x.version) for x in graph.opset_import] == [("", 17)], "Wrong opset")
    for value in list(graph.graph.input) + list(graph.graph.output):
        dims = value.type.tensor_type.shape.dim
        for axis, symbol in AXES[value.name].items():
            require(dims[axis].dim_param == symbol and not dims[axis].HasField('dim_value'),
                    f"Non-dynamic axis: {value.name}/{axis}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    session.disable_fallback()
    require(session.get_providers() == ['CPUExecutionProvider'], "Unexpected execution provider")
    inputs = [{'name': v.name, 'type': v.type, 'shape': v.shape} for v in session.get_inputs()]
    outputs = [{'name': v.name, 'type': v.type, 'shape': v.shape} for v in session.get_outputs()]
    require(inputs == [{'name': name, 'type': 'tensor(int64)', 'shape': ['batch', 'sequence']}
                       for name in ('input_ids', 'attention_mask')], "Invalid input schema")
    require(outputs == [{'name': 'logits', 'type': 'tensor(float)', 'shape': ['batch', 2]}],
            "Invalid output schema")
    return session, {'inputs': inputs, 'outputs': outputs, 'providers': session.get_providers()}


def compare(model, session, arrays, rows, baseline=None):
    # Both runtimes consume these same arrays; no second tokenization occurs.
    with torch.inference_mode():
        pt = model(**{name: torch.from_numpy(value) for name, value in arrays.items()}).logits.numpy()
    rt = session.run(['logits'], arrays)[0]
    require(pt.shape == rt.shape == (len(rows), 2), "Invalid logits shape")
    finite = bool(np.isfinite(pt).all() and np.isfinite(rt).all())
    records = []
    for i, row in enumerate(rows):
        left = model.config.id2label[int(pt[i].argmax())]
        right = model.config.id2label[int(rt[i].argmax())]
        difference = float(np.max(np.abs(pt[i].astype(np.float64) - rt[i]))) if finite else None
        record = dict(sentence_id=row['id'], length_bucket=row['length_bucket'],
                      actual_token_length=int(arrays['attention_mask'][i].sum()),
                      pytorch_logits=pt[i].tolist() if finite else None,
                      onnx_logits=rt[i].tolist() if finite else None,
                      max_absolute_logit_difference=difference, pytorch_label=left, onnx_label=right,
                      labels_agree=left == right, finite_logits=finite,
                      numerical_parity='PASS' if finite and difference <= TOLERANCE else 'FAIL',
                      label_parity='PASS' if left == right else 'FAIL')
        if baseline is not None:
            record['baseline_label'] = baseline[row['id']]
            record['baseline_agrees'] = right == baseline[row['id']]
        records.append(record)
    fingerprint = {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in arrays.items()}
    return records, {'input_shape': list(arrays['input_ids'].shape), 'output_shape': list(rt.shape),
                     'shared_input_sha256': fingerprint, 'padding_tokens': int((arrays['attention_mask'] == 0).sum())}


def summarize(records):
    mismatches = [r for r in records if not r['labels_agree']]
    failures = [r for r in records if r['numerical_parity'] == 'FAIL']
    baseline = [r for r in records if r.get('baseline_agrees') is False]
    buckets = {bucket: sum(r['length_bucket'] == bucket for r in mismatches)
               for bucket in sorted({r['length_bucket'] for r in records})}
    return dict(count=len(records), unique_sentence_ids=len({r['sentence_id'] for r in records}),
                largest_absolute_logit_difference=max((r['max_absolute_logit_difference'] or 0) for r in records),
                count_exceeding_tolerance=len(failures), total_label_mismatches=len(mismatches),
                label_agreement_percentage=100 * (len(records) - len(mismatches)) / len(records),
                disagreements_by_bucket=buckets, disagreements=mismatches,
                numerical_failures=failures, baseline_mismatches=baseline,
                baseline_agreement_percentage=100 * (len(records) - len(baseline)) / len(records),
                numerical_parity='FAIL' if failures else 'PASS',
                label_parity='FAIL' if mismatches else 'PASS',
                baseline_parity='FAIL' if baseline else 'PASS')


def representative_texts():
    # Single-vocabulary words produce varied sentiment with exact requested lengths.
    return [' '.join((['good', 'bad', 'movie'] * 66)[:length - 2])
            for length in (5, 20, 100, 200)]


def write_report(data, path=REPORT):
    # Exclusive creation prevents overwriting unrelated evidence. One complete serialization.
    serialized = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())


def run():
    require(git('rev-parse', 'HEAD') == HEAD and git('branch', '--show-current') == 'main',
            'Unexpected Git identity')
    require(not git('diff', 'HEAD') and not git('diff', '--cached'), 'Committed files changed')
    require(not OUTPUT.exists() and not REPORT.exists(), 'Destination already exists; refusing overwrite')
    rows = read_dataset()
    baseline, baseline_hashes = baseline_labels(rows)
    require(sha256(ROOT / 'bench/thresholds.yaml') ==
            '3f31248914dcd93754995f23002a15bcc8e7329259b6787ff638a4d7bc795439', 'Thresholds changed')
    with no_network():
        tokenizer, model = load_local()
        texts = representative_texts()
        sample = encode(tokenizer, texts[:1])
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        errors = []
        for attempt in range(2):
            try:
                export(model, sample)
                break
            except Exception as error:
                errors.append(repr(error))
                print(f'Export attempt {attempt + 1} failed: {error}', flush=True)
        else:
            raise RuntimeError(f'Both legacy exports failed: {errors}')
        session, schema = session_and_schema()
        report = dict(git_head=HEAD, model_name=MODEL, pinned_revision=REVISION,
                      source_model_directory=str(SOURCE), source_sha256=SOURCE_HASHES,
                      exporter='torch.onnx.export (legacy)', dynamo=False, opset=17, dynamic_axes=AXES,
                      onnx_sha256=sha256(OUTPUT), onnx_size_bytes=OUTPUT.stat().st_size,
                      model_threads=4, inter_op_threads=1,
                      versions=dict(python=platform.python_version(), torch=torch.__version__,
                                    transformers=__import__('transformers').__version__, onnx=onnx.__version__,
                                    onnxruntime=ort.__version__, pytest=__import__('pytest').__version__),
                      dataset_sha256=DATA_HASH, dataset_seed=20261010, tolerance=TOLERANCE,
                      schema=schema, export_errors=errors, baseline_source_sha256=baseline_hashes,
                      baseline_repetitions_agree='PASS', export_check='PASS', dynamic_axes_check='PASS',
                      representative_cases=[], dataset_batches={}, status='PASS')
        for index, target in enumerate((5, 20, 100, 200)):
            for size in (1, 4):
                selected = [texts[index]] if size == 1 else [texts[index]] + [texts[i] for i in range(4) if i != index]
                case_rows = [dict(id=f'representative-{target}-{i}', length_bucket='representative') for i in range(size)]
                arrays = encode(tokenizer, selected)
                records, batch = compare(model, session, arrays, case_rows)
                case = dict(target_tokens=target, batch_size=size, texts=selected, records=records,
                            actual_lengths=[r['actual_token_length'] for r in records], batch=batch,
                            summary=summarize(records))
                report['representative_cases'].append(case)
                if any(case['summary'][key] == 'FAIL' for key in ('numerical_parity', 'label_parity')):
                    report['status'] = 'FAIL'
                    write_report(report)
                    raise RuntimeError('Representative parity failed; genuine evidence saved')
        for size in (1, 4):
            records, batches = [], []
            for start in range(0, len(rows), size):
                group = rows[start:start + size]
                arrays = encode(tokenizer, [r['text'] for r in group])
                observed, batch = compare(model, session, arrays, group, baseline)
                records.extend(observed)
                batch['sentence_ids'] = [r['id'] for r in group]
                batches.append(batch)
                if (start + size) % 100 == 0:
                    print(f'Correctness batch size {size}: {start + size}/1000 sentences checked', flush=True)
            summary = summarize(records)
            require([r['sentence_id'] for r in records] == [r['id'] for r in rows], 'ID mapping failed')
            report['dataset_batches'][str(size)] = dict(records=records, batches=batches, summary=summary, coverage='PASS')
            print(f'Batch size {size}: {json.dumps({k: v for k, v in summary.items() if not isinstance(v, (list, dict))})}', flush=True)
            if any(summary[key] == 'FAIL' for key in ('numerical_parity', 'label_parity', 'baseline_parity')):
                report['status'] = 'FAIL'
                write_report(report)
                raise RuntimeError('Full-dataset parity failed; genuine evidence saved')
        write_report(report)
        print(f'PASS: ONNX SHA-256 {report["onnx_sha256"]}; size {report["onnx_size_bytes"]} bytes')
        print(f'Evidence: {REPORT.relative_to(ROOT)}; SHA-256 {sha256(REPORT)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-and-check', action='store_true', required=True,
                        help='Export locally and verify representatives plus all 1000 IDs at batch sizes 1 and 4')
    parser.parse_args()
    run()
