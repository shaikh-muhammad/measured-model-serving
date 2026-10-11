"""Generate synthetic feedback; final generation requires a pinned local tokenizer.

Offline tests inject a test tokenizer. Their token counts are not DistilBERT counts.
No dataset is validated against the real model until final generation succeeds.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re

SEED = 20261010
MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
TOKENIZER_REPOSITORY = "distilbert/" + MODEL_ID
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUCKETS = {"short": (5, 20, 400), "medium": (21, 80, 400), "long": (81, 200, 200)}
CLAUSE_COUNTS = {"short": (1, 1), "medium": (3, 5), "long": (9, 12)}
MAX_ATTEMPTS = 100
SUBJECTS = ("support reply", "help page", "service update", "support process")
POSITIVE = ("clear", "helpful", "useful", "excellent")
NEGATIVE = ("confusing", "unhelpful", "unclear", "disappointing")
CONTEXTS = ("during setup", "after the update", "during checkout", "after a restart")
CONNECTORS = (" Also, ", " Meanwhile, ", " In addition, ")


def token_count(tokenizer, text):
    """Count the complete input, including special tokens, without truncation."""
    return len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])


def make_clause(rng, tone):
    subject = rng.choice(SUBJECTS)
    context = rng.choice(CONTEXTS)
    positive = rng.choice(POSITIVE)
    negative = rng.choice(NEGATIVE)
    if tone == "mixed":
        return f"The {subject} was {positive} but also {negative} {context}."
    adjective = positive if tone == "positive" else negative
    return f"The {subject} was {adjective} and easy to review {context}."


def generate_records(tokenizer, seed=SEED):
    """Return 1,000 records; rejection sampling has a fixed attempt limit."""
    rng = random.Random(seed)
    records = []
    for bucket, (minimum, maximum, required) in BUCKETS.items():
        for index in range(required):
            tone = ("positive", "negative", "mixed")[index % 3]
            for _ in range(MAX_ATTEMPTS):
                low, high = CLAUSE_COUNTS[bucket]
                clauses = [make_clause(rng, tone) for _ in range(rng.randint(low, high))]
                text = rng.choice(CONNECTORS).join(clauses)
                count = token_count(tokenizer, text)
                if len(text) <= 2000 and minimum <= count <= maximum and count <= 256:
                    records.append({"id": f"sentence-{len(records) + 1:04d}",
                                    "text": text, "length_bucket": bucket})
                    break
            else:
                raise ValueError(f"Cannot fill {bucket} bucket at record {index + 1} "
                                 f"after {MAX_ATTEMPTS} attempts")
    return records


def jsonl_bytes(records):
    """Use stable UTF-8 encoding, field order, separators, and LF newlines."""
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                   for row in records).encode("utf-8")


def validate_records(records, tokenizer):
    """Validate with the supplied tokenizer; the CLI supplies the real tokenizer."""
    if len(records) != 1000 or len({row["id"] for row in records}) != 1000:
        raise ValueError("Expected exactly 1,000 records with unique IDs")
    counts = dict.fromkeys(BUCKETS, 0)
    for row in records:
        if set(row) != {"id", "text", "length_bucket"}:
            raise ValueError("Invalid record fields")
        bucket = row["length_bucket"]
        if bucket not in BUCKETS:
            raise ValueError("Unknown length bucket")
        minimum, maximum, _ = BUCKETS[bucket]
        count = token_count(tokenizer, row["text"])
        if len(row["text"]) > 2000 or not minimum <= count <= maximum or count > 256:
            raise ValueError(f"Invalid input limits for {row['id']}")
        counts[bucket] += 1
    if counts != {bucket: limits[2] for bucket, limits in BUCKETS.items()}:
        raise ValueError("Incorrect length-bucket distribution")


def load_local_tokenizer(revision, cache_dir):
    """Require an explicit immutable revision and a project-local HF cache."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Tokenizer revision must be a verified 40-character commit SHA")
    cache_dir = Path(cache_dir).resolve()
    if not cache_dir.is_relative_to(PROJECT_ROOT) or not cache_dir.is_dir():
        raise ValueError("Required local tokenizer cache is absent or outside the project; "
                         "downloads are disabled")
    from transformers import DistilBertTokenizer
    try:
        return DistilBertTokenizer.from_pretrained(
            TOKENIZER_REPOSITORY, revision=revision, cache_dir=str(cache_dir),
            local_files_only=True, token=False,
        )
    except (OSError, ValueError, TypeError) as error:
        raise ValueError("Required pinned local DistilBERT tokenizer is unavailable; "
                         "downloads are disabled") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Verified model commit SHA")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "model-cache" / "hub")
    args = parser.parse_args()
    try:
        tokenizer = load_local_tokenizer(args.revision, args.cache_dir)
        records = generate_records(tokenizer)
        validate_records(records, tokenizer)
        payload = jsonl_bytes(records)
        output = PROJECT_ROOT / "data" / "sentences.jsonl"
        if output.exists():
            if output.read_bytes() != payload:
                raise ValueError("Existing dataset differs; refusing to overwrite")
        else:
            with output.open("xb") as stream:
                stream.write(payload)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Generation failed: {error}\n")
    print(f"Validated {len(records)} records with {MODEL_ID} at {args.revision}")
    print(f"seed={SEED}; sha256={hashlib.sha256(payload).hexdigest()}")


if __name__ == "__main__":
    main()
