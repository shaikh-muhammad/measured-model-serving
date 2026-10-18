"""Offline saved-evidence verification, without Docker, models or network."""

import copy
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("report", ROOT / "bench/report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    original = subprocess.Popen

    def guarded(argv, *args, **kwargs):
        assert argv[0] == "git" and argv[1] in ("show", "ls-tree"), argv
        return original(argv, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Network, Docker or model operation forbidden")

    monkeypatch.setattr(subprocess, "Popen", guarded)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(report.runner, "docker", forbidden)
    monkeypatch.setattr(report.runner, "main", forbidden)


@pytest.fixture(scope="module")
def evidence():
    return report.load_evidence()


def test_discovery_inventory_and_groups(evidence, tmp_path):
    files = report.discover()
    assert list(files) == ["v0", "v1", "v2"]
    assert all(len(a) == 6 and len(s) == 2 for a, s in files.values())
    assert set(evidence["groups"]) == {(v, c) for v in files for c in (1, 4)}
    for version, directory in report.DIRECTORIES.items():
        destination = tmp_path / directory
        destination.mkdir(parents=True)
        for group in files[version]:
            for source in group:
                (destination / source.name).write_bytes(source.read_bytes())
    missing = next((tmp_path / report.DIRECTORIES["v0"]).glob("attempt-*.json"))
    missing.unlink()
    with pytest.raises(ValueError, match="six attempts and two summaries"):
        report.discover(tmp_path)


@pytest.mark.parametrize("mutation, message", [
    ("invalid", "validity_status"), ("missing", "Incomplete dataset coverage"),
    ("duplicate", "dataset IDs"), ("failure", "Failed HTTP request"),
    ("warmup", "Failed HTTP request"), ("timing", "Conflicting saved HTTP"),
    ("memory", "Conflicting peak memory"), ("version", "Invalid version"),
    ("threads", "actual_model_threads"), ("metric", "differs from original"),
])
def test_corrupt_attempt_fails(evidence, mutation, message):
    a = copy.deepcopy(evidence["groups"]["v0", 1]["runs"][0])
    if mutation == "invalid":
        a["validity_status"] = "INVALID"
    elif mutation == "missing":
        a["request_records"].pop()
    elif mutation == "duplicate":
        a["request_records"][0]["sentence_id"] = a["request_records"][1]["sentence_id"]
    elif mutation == "failure":
        a["request_records"][0]["success"] = False
    elif mutation == "warmup":
        a["warmup_records"][0]["success"] = False
    elif mutation == "timing":
        del a["request_records"][0]["model_ms"]
    elif mutation == "memory":
        a["peak_container_memory"] += 1
    elif mutation == "version":
        a["version"] = "v1"
    elif mutation == "threads":
        a["actual_model_threads"] = 2
    elif mutation == "metric":
        a["p99"] += 1
    with pytest.raises(ValueError, match=message):
        report.validate_attempt(a, "v0", evidence["plan"], report.runner.load_dataset(evidence["plan"]))


def test_summary_consistency_and_clear_cli_failure(monkeypatch, capsys):
    original = report.read_json

    def corrupted(path):
        value = original(path)
        if path.name.startswith("summary-"):
            value["statistics"]["p99"]["median"] += 1
        return value

    monkeypatch.setattr(report, "read_json", corrupted)
    assert report.main([]) == 1
    streams = capsys.readouterr()
    assert streams.out == ""
    assert "Summary-versus-attempt mismatch" in streams.err


def test_nearest_rank_and_three_run_aggregation(evidence):
    assert [report.runner.nearest_rank(list(range(1, 1001)), p) for p in (50, 95, 99)] == [500, 950, 990]
    for group in evidence["groups"].values():
        runs, summary = group["runs"], group["statistics"]
        for key in report.METRICS:
            values = sorted(a[key] for a in runs)
            assert summary[key] == {"median": values[1], "range": [values[0], values[2]]}


def test_registered_arithmetic_quality_and_precedence(evidence):
    plan = evidence["plan"]
    baseline = {"median": 100, "range": [98, 102]}
    d = report.decision(baseline, {"median": 75}, .99, plan)
    assert d["improvement"] == .25 and d["baseline_spread"] == .04
    assert d["outcome"] == "KEEP"
    assert report.decision(baseline, {"median": 75}, .989, plan)["outcome"] == "REJECT"
    noisy = {"median": 100, "range": [80, 105]}
    assert report.decision(noisy, {"median": 75}, 1, plan)["outcome"] == "INCONCLUSIVE"
    assert report.decision(noisy, {"median": 75}, .98, plan)["outcome"] == "REJECT"
    assert report.decision(baseline, {"median": 80}, 1, plan)["outcome"] == "KEEP"
    assert evidence["decisions"]["v1"]["outcome"] == "REJECT"
    assert evidence["decisions"]["v2"]["outcome"] == "KEEP"
    # Earlier values are regression checks only, never generator inputs.
    assert [evidence["groups"][v, 4]["statistics"]["p99"]["median"] for v in report.DIRECTORIES] == [987.331188, 1233.355484, 625.352030]


def test_labels_from_each_saved_http_run(evidence):
    for q in evidence["quality"].values():
        assert (q["unique_ids"], q["matching"], q["total"]) == (1000, 6000, 6000)
        assert q["fraction"] == 1 and q["mismatching_ids"] == []
        assert q["mismatch_counts"] == {"long": 0, "medium": 0, "short": 0}
    rows = [{"id": "a", "length_bucket": "short"}, {"id": "b", "length_bucket": "long"}]
    result = report.compare_labels([{"request_records": [
        {"sentence_id": "a", "predicted_label": "POSITIVE"},
        {"sentence_id": "b", "predicted_label": "NEGATIVE"}]}], {"a": "POSITIVE", "b": "POSITIVE"}, rows)
    assert result["fraction"] == .5 and result["mismatching_ids"] == ["b"]
    assert result["mismatch_counts"] == {"long": 1, "short": 0}


def test_outputs_values_units_xml_and_determinism(evidence):
    before = {p: p.read_bytes() for pair in report.discover().values() for group in pair for p in group}
    first = report.outputs(evidence)
    assert first == report.outputs(report.load_evidence())
    md, svg = first["docs/benchmark_report.md"], first["docs/p99_by_version.svg"]
    xml = ET.fromstring(svg)
    assert xml.tag == "{http://www.w3.org/2000/svg}svg"
    assert "cgroup memory.peak" in md and "MiB = bytes / 1,048,576" in md
    assert "Image size (bytes)" in md and "Startup (s)" in md
    texts = [node.text for node in xml.iter("{http://www.w3.org/2000/svg}text")]
    for v in report.DIRECTORIES:
        stat = evidence["groups"][v, 4]["statistics"]["p99"]
        assert f"{stat['median']:.6f}" in texts
        for c in (1, 4):
            s = evidence["groups"][v, c]["statistics"]
            assert report.interval(s["p99"]) in md
            assert report.interval(s["peak_container_memory"], 1024**2) in md
            assert report.interval(s["docker_image_size_bytes"], digits=0) in md
    assert all(p.read_bytes() == data for p, data in before.items())
    assert not {"torch", "onnxruntime", "transformers"}.intersection(sys.modules)


def test_integrity_rejects_changed_input(monkeypatch):
    monkeypatch.setattr(report.runner, "sha256", lambda p: "changed")
    with pytest.raises(ValueError, match="Frozen SHA-256 mismatch"):
        report.load_evidence()


def test_check_mode_and_make_target(evidence, monkeypatch, capsys):
    rendered = report.outputs(evidence)
    original = Path.read_bytes

    def fresh(path):
        name = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else ""
        return rendered[name].encode() if name in rendered else original(path)

    monkeypatch.setattr(report, "load_evidence", lambda: evidence)
    monkeypatch.setattr(Path, "is_file", lambda p: True)
    monkeypatch.setattr(Path, "read_bytes", fresh)
    assert report.main(["--check"]) == 0
    monkeypatch.setattr(Path, "read_bytes", lambda p: b"stale")
    assert report.main(["--check"]) == 1
    assert "Stale or missing output" in capsys.readouterr().err
    recipe = (ROOT / "Makefile").read_text().split("\nreport:\n")[1]
    assert recipe.strip() == "source scripts/env.sh && PYTHONDONTWRITEBYTECODE=1 python bench/report.py"
