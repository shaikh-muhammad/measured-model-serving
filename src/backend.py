"""CPU v0 inference; model preparation is a separate, explicitly authorized step."""

import os
from pathlib import Path
from time import perf_counter_ns

import torch

MAX_TOKENS = 256


class V0Backend:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model.to("cpu").eval()
        self.labels = {int(index): label for index, label in model.config.id2label.items()}
        if set(self.labels) != {0, 1} or set(self.labels.values()) != {"POSITIVE", "NEGATIVE"}:
            raise ValueError("Model configuration must map two classes to POSITIVE and NEGATIVE")

    @classmethod
    def from_environment(cls):
        directory = os.environ.get("MODEL_DIR")
        if not directory:
            raise RuntimeError("MODEL_DIR must explicitly name a prepared local model directory")
        path = Path(directory).resolve()
        if not path.is_dir():
            raise RuntimeError(f"Local MODEL_DIR does not exist: {path}; downloads are disabled")
        try:
            threads = int(os.environ.get("MODEL_THREADS", "4"))
            if threads < 1:
                raise ValueError("MODEL_THREADS must be a positive integer")
            torch.set_num_threads(threads)
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                str(path), local_files_only=True, token=False, trust_remote_code=False,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                str(path), local_files_only=True, token=False, trust_remote_code=False,
            )
            if model.config.model_type != "distilbert":
                raise ValueError("MODEL_DIR must contain the prepared DistilBERT sentiment model")
            return cls(tokenizer, model)
        except Exception as error:
            raise RuntimeError(f"Cannot initialize local v0 model from {path}: {error}") from error

    def classify(self, text):
        start = perf_counter_ns()
        full_ids = self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        truncated = len(full_ids) > MAX_TOKENS
        inputs = self.tokenizer(
            text, add_special_tokens=True, truncation=True,
            max_length=MAX_TOKENS, return_tensors="pt",
        )
        if inputs["input_ids"].shape[1] > MAX_TOKENS:
            raise RuntimeError("Tokenizer exceeded the 256-token model input limit")
        tokenized = perf_counter_ns()
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        modeled = perf_counter_ns()
        probabilities = torch.softmax(logits, dim=-1)[0]
        index = int(torch.argmax(probabilities).item())
        label = self.labels[index]
        score = float(probabilities[index].item())
        processed = perf_counter_ns()
        return {
            "label": label, "score": score, "truncated": truncated,
            "timings_ms": {
                "tokenize": (tokenized - start) / 1_000_000,
                "model": (modeled - tokenized) / 1_000_000,
                "postprocess": (processed - modeled) / 1_000_000,
            },
        }
