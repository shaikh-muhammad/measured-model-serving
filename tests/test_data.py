"""Offline structural tests only: fake counts do not validate DistilBERT buckets."""

from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_sentences", ROOT / "data/make_sentences.py")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class TestTokenizer:
    """Deterministic whitespace test double, with two synthetic special tokens."""

    __test__ = False

    def __call__(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens is True
        assert truncation is False
        return {"input_ids": [0] * (len(text.split()) + 2)}


@pytest.fixture(scope="module")
def records():
    return generator.generate_records(TestTokenizer())


def test_same_seed_is_deterministic(records):
    assert records == generator.generate_records(TestTokenizer(), seed=generator.SEED)


def test_exact_count_and_unique_ids(records):
    assert len(records) == 1000
    assert len({row["id"] for row in records}) == 1000
    assert [row["id"] for row in records] == [f"sentence-{i:04d}" for i in range(1, 1001)]


def test_bucket_distribution_and_test_token_counts(records):
    assert Counter(row["length_bucket"] for row in records) == {
        "short": 400, "medium": 400, "long": 200,
    }
    for row in records:
        low, high, _ = generator.BUCKETS[row["length_bucket"]]
        assert low <= generator.token_count(TestTokenizer(), row["text"]) <= high
        assert generator.token_count(TestTokenizer(), row["text"]) <= 256
    generator.validate_records(records, TestTokenizer())


def test_jsonl_structure_and_character_limit(records):
    payload = generator.jsonl_bytes(records)
    assert payload.endswith(b"\n")
    assert b"\r" not in payload
    decoded = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    assert decoded == records
    for row in decoded:
        assert set(row) == {"id", "text", "length_bucket"}
        assert all(isinstance(value, str) for value in row.values())
        assert 0 < len(row["text"]) <= 2000


def test_repeat_output_bytes_and_sha256(records, tmp_path):
    first = generator.jsonl_bytes(records)
    second = generator.jsonl_bytes(generator.generate_records(TestTokenizer()))
    first_path, second_path = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    first_path.write_bytes(first)
    second_path.write_bytes(second)
    assert first_path.read_bytes() == second_path.read_bytes()
    assert hashlib.sha256(first_path.read_bytes()).digest() == hashlib.sha256(second_path.read_bytes()).digest()


def test_impossible_tokenizer_fails_with_bounded_attempts():
    calls = 0

    def impossible(text, *, add_special_tokens, truncation):
        nonlocal calls
        calls += 1
        return {"input_ids": [0] * 300}

    with pytest.raises(ValueError, match="Cannot fill short bucket"):
        generator.generate_records(impossible)
    assert calls == generator.MAX_ATTEMPTS


def test_absent_local_cache_fails_without_import_or_download():
    with pytest.raises(ValueError, match="local tokenizer cache is absent"):
        generator.load_local_tokenizer("0" * 40, ROOT / "data/make_sentences.py")


def test_unpinned_revision_is_rejected():
    with pytest.raises(ValueError, match="verified 40-character commit SHA"):
        generator.load_local_tokenizer("main", ROOT)


def test_edge_case_fixture_completeness():
    rows = [json.loads(line) for line in (ROOT / "tests/edge_cases.jsonl").read_text().splitlines()]
    assert len(rows) == 7
    cases = {row["id"]: row["text"] for row in rows}
    assert set(cases) == {"empty", "whitespace", "over_character_limit", "emoji",
                          "accented", "over_token_limit", "digits"}
    assert all(set(row) == {"id", "text"} for row in rows)
    assert all(isinstance(row["id"], str) and isinstance(row["text"], str) for row in rows)
    assert cases["empty"] == ""
    assert cases["whitespace"] and cases["whitespace"].isspace()
    assert len(cases["over_character_limit"]) == 5000
    assert "🙂" in cases["emoji"]
    assert "é" in cases["accented"]
    assert cases["over_token_limit"].split() == ["help"] * 300
    assert len(cases["over_token_limit"]) <= 2000
    assert cases["digits"].isascii() and cases["digits"].isdigit()
