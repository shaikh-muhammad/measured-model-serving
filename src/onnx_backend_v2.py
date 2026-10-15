"""V2 artifact identity adapter; inference remains OnnxBackend.classify."""

import hashlib
import os
from pathlib import Path

from src.onnx_backend import ARTIFACT_HASHES, OnnxBackend, StandaloneEncoder, ort

V2_MODEL_SHA256 = 'c488916e50d340cd86ee5740bdf1522ec0fc7245a51f8755f50334398363ec51'


class OnnxBackendV2(OnnxBackend):
    @classmethod
    def from_environment(cls):
        model_dir = os.environ.get('MODEL_DIR')
        if not model_dir:
            raise RuntimeError('MODEL_DIR must explicitly name the bundled local artifacts')
        if os.environ.get('MODEL_THREADS', '4') != '4':
            raise RuntimeError('Frozen MODEL_THREADS=4 is required')
        directory = Path(model_dir).resolve()
        tokenizer = Path(os.environ.get('TOKENIZER_PATH', str(directory / 'tokenizer.json')))
        for name, expected in {'model.onnx': V2_MODEL_SHA256,
                               'tokenizer.json': ARTIFACT_HASHES['tokenizer.json']}.items():
            artifact = tokenizer if name == 'tokenizer.json' else directory / name
            if not artifact.is_file() or artifact.is_symlink():
                raise RuntimeError(f'Missing regular local artifact: {artifact}; downloads are disabled')
            with artifact.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != expected:
                raise RuntimeError(f'Pinned artifact SHA-256 mismatch: {name}')
        encoder = StandaloneEncoder(tokenizer)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(directory / 'model.onnx'), sess_options=options,
                                       providers=['CPUExecutionProvider'])
        session.disable_fallback()
        if session.get_providers() != ['CPUExecutionProvider']:
            raise RuntimeError('CPU execution provider required')
        if [(i.name, i.type, i.shape) for i in session.get_inputs()] != [
                (name, 'tensor(int64)', ['batch', 'sequence'])
                for name in ('input_ids', 'attention_mask')]:
            raise RuntimeError('Unexpected ONNX input schema')
        if [(o.name, o.type, o.shape) for o in session.get_outputs()] != [
                ('logits', 'tensor(float)', ['batch', 2])]:
            raise RuntimeError('Unexpected ONNX output schema')
        return cls(encoder, session)
