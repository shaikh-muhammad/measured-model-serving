"""Prepare the pinned public model anonymously; never download other files."""

import argparse
import hashlib
import os
from pathlib import Path
import shutil

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from huggingface_hub import hf_hub_download

REPOSITORY = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
REVISION = "714eb0fa89d2f80546fda750413ed43d93601a13"
ROOT = Path(__file__).resolve().parents[1]
# Original file identities verified from the official pinned-revision metadata.
IDENTITIES = {
    "config.json": ("git", "b57fe5dfcb8ec3f9bab35ed427c3434e3c7dd1ba"),
    "model.safetensors": ("sha256", "7c3919835e442510166d267fe7cbe847e0c51cd26d9ba07b89a57b952b49b8aa"),
    "tokenizer_config.json": ("git", "3ed34255a7cb8e6706a8bb21993836e99e7b959f"),
    "vocab.txt": ("git", "fb140275c155a9c7c5a3b3e0e77a9e839594a938"),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(local_only=False):
    ready = ROOT / "model-cache" / "v0"
    ready.mkdir(parents=True, exist_ok=True)
    for name, (kind, expected) in IDENTITIES.items():
        cached = Path(hf_hub_download(
            repo_id=REPOSITORY, filename=name, revision=REVISION,
            cache_dir=str(ROOT / "model-cache" / "hub"),
            token=False, local_files_only=local_only,
        ))
        digest = sha256(cached)
        if kind == "git":
            content = cached.read_bytes()
            identity = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        else:
            identity = digest
        if identity != expected:
            raise RuntimeError(f"Pinned original-file identity mismatch: {name}")
        destination = ready / name
        if destination.is_symlink():
            raise RuntimeError(f"Ready model file must not be a symlink: {destination}")
        if not destination.exists() or sha256(destination) != digest:
            shutil.copyfile(cached, destination)
        if not destination.is_file() or sha256(destination) != digest:
            raise RuntimeError(f"Prepared SHA-256 mismatch: {name}")
        print(f"{digest}  model-cache/v0/{name}")
    print(f"Prepared {REPOSITORY}@{REVISION}; four regular files; no tokenizer.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-only", action="store_true", help="Verify using the existing cache only")
    args = parser.parse_args()
    prepare(local_only=args.local_only)
