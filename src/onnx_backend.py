"""Pinned local SST-2 logits with the verified standalone tokenizer helper."""

import hashlib
import os
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import onnxruntime as ort

from scripts.verify_tokenizer import StandaloneEncoder

ort.disable_telemetry_events()

ARTIFACT_HASHES = {
    'model.onnx': 'dda17f29d4032bc0eecd4333c9be4fe82c6ea72720a770f88c8e1fbbb47779c1',
    'tokenizer.json': '01c420969d5dee75248f2c1b1bf5d834f40f1de1427aa0b9fc93c61055d809e0',
}
# Original pinned config and saved parity evidence establish this class order.
LABELS = {0: 'NEGATIVE', 1: 'POSITIVE'}


class OnnxBackend:
    def __init__(self, encoder, session):
        self.encoder = encoder
        self.session = session

    @classmethod
    def from_environment(cls):
        directory = os.environ.get('MODEL_DIR')
        if not directory:
            raise RuntimeError('MODEL_DIR must explicitly name the bundled local artifacts')
        if os.environ.get('MODEL_THREADS', '4') != '4':
            raise RuntimeError('Frozen MODEL_THREADS=4 is required')
        path = Path(directory).resolve()
        for name, expected in ARTIFACT_HASHES.items():
            artifact = path / name
            if not artifact.is_file() or artifact.is_symlink():
                raise RuntimeError(f'Missing regular local artifact: {artifact}; downloads are disabled')
            digest = hashlib.sha256()
            with artifact.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise RuntimeError(f'Pinned artifact SHA-256 mismatch: {name}')
        encoder = StandaloneEncoder(path / 'tokenizer.json')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(path / 'model.onnx'), sess_options=options,
                                       providers=['CPUExecutionProvider'])
        session.disable_fallback()
        if session.get_providers() != ['CPUExecutionProvider']:
            raise RuntimeError('CPU execution provider required')
        inputs = [(item.name, item.type, item.shape) for item in session.get_inputs()]
        outputs = [(item.name, item.type, item.shape) for item in session.get_outputs()]
        if inputs != [(name, 'tensor(int64)', ['batch', 'sequence']) for name in ('input_ids', 'attention_mask')]:
            raise RuntimeError('Unexpected ONNX input schema')
        if outputs != [('logits', 'tensor(float)', ['batch', 2])]:
            raise RuntimeError('Unexpected ONNX output schema')
        return cls(encoder, session)

    def classify(self, text):
        start = perf_counter_ns()
        encoded = self.encoder.encode([text], truncation=True)
        inputs = {name: np.asarray(encoded[name], dtype=np.int64)
                  for name in ('input_ids', 'attention_mask')}
        if not 2 <= inputs['input_ids'].shape[1] <= 256:
            raise RuntimeError('Invalid model sequence length')
        tokenized = perf_counter_ns()
        logits = self.session.run(['logits'], inputs)[0]
        modeled = perf_counter_ns()
        if logits.shape != (1, 2) or not np.isfinite(logits).all():
            raise RuntimeError('Invalid or nonfinite ONNX logits')
        values = logits[0].astype(np.float64)
        probabilities = np.exp(values - values.max())
        probabilities /= probabilities.sum()
        index = int(probabilities.argmax())
        score = float(probabilities[index])
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise RuntimeError('Invalid softmax confidence')
        processed = perf_counter_ns()
        return dict(label=LABELS[index], score=score, truncated=encoded['truncated'][0],
                    timings_ms=dict(tokenize=(tokenized - start) / 1_000_000,
                                    model=(modeled - tokenized) / 1_000_000,
                                    postprocess=(processed - modeled) / 1_000_000))
