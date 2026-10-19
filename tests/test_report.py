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
REPORT_IMPORT_CHECK = [
    sys.executable, "-I", "-B", "-c",
    """import importlib.util
import sys

spec = importlib.util.spec_from_file_location("report", sys.argv[1])
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
evidence = report.load_evidence()
report.outputs(evidence)
loaded = sorted({"torch", "onnxruntime", "transformers"}.intersection(sys.modules))
if loaded:
    raise SystemExit("Report import loaded forbidden dependencies: " + ", ".join(loaded))
""",
    str(ROOT / "bench/report.py"),
]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    original = subprocess.Popen

    def guarded(argv, *args, **kwargs):
        assert argv == REPORT_IMPORT_CHECK or (
            argv[0] == "git" and argv[1] in ("show", "ls-tree")
        ), argv
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
    result = subprocess.run(REPORT_IMPORT_CHECK, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"Isolated report import failed ({result.returncode})\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


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


def original_attempts():
    """Independent test inputs: do not use the generator's aggregates."""
    result = {}
    for version, directory in (("v0", "results/v0-baseline"), ("v1", "results/v1-official"),
                                ("v2", "results/v2-official")):
        for path in sorted((ROOT / directory).glob("attempt-*.json")):
            attempt = json.loads(path.read_text())
            assert attempt["validity_status"] == "VALID"
            result.setdefault((version, attempt["concurrency"]), []).append((path, attempt))
    for values in result.values():
        values.sort(key=lambda pair: pair[1]["run_slot"])
        assert [a["run_slot"] for _, a in values] == [1, 2, 3]
    return result


def test_portfolio_data_independently_matches_attempts():
    import hashlib
    import statistics
    data = json.loads((ROOT / "docs/chart_data.json").read_text())
    attempts = original_attempts()
    definitions = {"p50": (1, 0), "p95": (1, 0), "p99": (1, 0),
                   "throughput_requests_per_second": (1, 1),
                   "peak_container_memory": (1000000, 0),
                   "docker_image_size_bytes": (1000000000, 1),
                   "startup_time_seconds": (1, 9)}
    for workload in data["workloads"]:
        c = workload["concurrency"]
        for row in workload["versions"]:
            v = row["version"]
            for key, (divisor, digits) in definitions.items():
                values = [a[key] for _, a in attempts[v, c]]
                median = statistics.median(values)
                baseline = statistics.median(a[key] for _, a in attempts["v0", c])
                point = row["metrics"][key]
                assert point["run_values"] == values
                assert point["raw_median"] == median
                assert point["baseline"] == baseline
                assert point["change_percent"] == (median - baseline) / baseline * 100
                assert point["display_number"] == f"{median/divisor:.{digits}f}"
                assert point["display"] == f"{median/divisor:.{digits}f} {point['unit']}"
                expected_change = "original" if v == "v0" else f"{(median-baseline)/baseline*100:+.0f}% vs original"
                assert point["change_label"] == expected_change
                optimum = max if key == "throughput_requests_per_second" else min
                expected_best = median == optimum(statistics.median(a[key] for _, a in attempts[other, c])
                                              for other in ("v0", "v1", "v2"))
                assert point["best"] == expected_best
            largest = max(statistics.median(a["p99"] for _, a in attempts[other, c])
                          for other in ("v0", "v1", "v2"))
            assert row["progress_fraction"] == statistics.median(a["p99"] for _, a in attempts[v,c])/largest
    assert len(data["sources"]) == 18
    assert {s["path"] for s in data["sources"]} == {
        p.relative_to(ROOT).as_posix() for values in attempts.values() for p, _ in values}
    for source in data["sources"]:
        path = ROOT/source["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
        assert path.read_bytes() == subprocess.check_output(["git", "show", "HEAD:"+source["path"]], cwd=ROOT)
    # Recompute run percentiles from individual saved latencies, not summary fields.
    for values in attempts.values():
        for _, attempt in values:
            latencies = sorted(r["client_observed_request_latency_ms"] for r in attempt["request_records"])
            assert len(latencies) == 1000
            for key, index in (("p50",499),("p95",949),("p99",989)):
                assert attempt[key] == latencies[index]


def test_portfolio_labels_and_decisions_from_saved_predictions():
    import statistics
    attempts = original_attempts()
    reference = {r["sentence_id"]: r["predicted_label"] for r in attempts["v0",1][0][1]["request_records"]}
    assert len(reference) == 1000
    for values in attempts.values():
        for _, attempt in values:
            actual = {r["sentence_id"]: r["predicted_label"] for r in attempt["request_records"]}
            assert actual == reference
    data = json.loads((ROOT/"docs/chart_data.json").read_text())
    baseline = statistics.median(a["p99"] for _,a in attempts["v0",4])
    spread = (max(a["p99"] for _,a in attempts["v0",4])-min(a["p99"] for _,a in attempts["v0",4]))/baseline
    for row in data["workloads"][0]["versions"]:
        v=row["version"]
        if v=="v0":
            assert row["decision_label"]=="Starting point"
            continue
        improvement=(baseline-statistics.median(a["p99"] for _,a in attempts[v,4]))/baseline
        # Original registered 20% and noise gates; all saved quality checks pass.
        expected="KEEP" if improvement>=0.20 and improvement>spread else "REJECT" if improvement<0.20 else "INCONCLUSIVE"
        assert row["decision"]==expected
        badge=f"{abs(improvement)*100:.0f}% "+("faster" if improvement>0 else "slower")
        assert row["badge"]==badge
        assert row["decision_label"]=={"KEEP":"Kept","REJECT":"Rejected","INCONCLUSIVE":"Inconclusive"}[expected]+": "+badge
        assert row["agreement"]["unique_ids"]==1000
        assert row["agreement"]["matching"]==row["agreement"]["total"]==6000


def test_readme_numeric_tables_independent_of_generator():
    import statistics
    attempts=original_attempts()
    text=(ROOT/"README.md").read_text()
    human=[line.split("|")[1:-1] for line in text.splitlines() if line.startswith(("| v0 |","| v1 |","| v2 |"))]
    assert len(human)==6
    for i,v in enumerate(("v0","v1","v2")):
        values={k:statistics.median(a[k] for _,a in attempts[v,4]) for k in
                ("p50","p95","p99","throughput_requests_per_second","peak_container_memory",
                 "docker_image_size_bytes","startup_time_seconds")}
        row=[cell.strip() for cell in human[i]]
        assert row[2]==f"{values['p99']:.0f} ms"
        assert row[3]==f"{values['throughput_requests_per_second']:.1f}"
        assert row[4]==f"{values['peak_container_memory']/1000000:.0f} MB"
        assert row[5]==f"{values['docker_image_size_bytes']/1000000000:.1f} GB"
        exact=[cell.strip() for cell in human[i+3]]
        assert exact[1:5]==[f"{values[k]:.6f}" for k in ("p50","p95","p99")]+[str(values["throughput_requests_per_second"])]
        assert exact[5:9]==[f"{values['peak_container_memory']:.0f}",
                            str(values["peak_container_memory"]/1048576),
                            f"{values['docker_image_size_bytes']:.0f}",
                            f"{values['startup_time_seconds']:.9f}"]
        assert exact[9]=="6000/6000"
    b=statistics.median(a["p99"] for _,a in attempts["v0",4])
    v2=statistics.median(a["p99"] for _,a in attempts["v2",4])
    assert f"a {(b-v2)/b*100:.0f}% reduction" in text
    memory=[statistics.median(a["peak_container_memory"] for _,a in attempts[v,4]) for v in ("v0","v2")]
    assert f"memory use fell by {(memory[0]-memory[1])/memory[0]*100:.0f}%" in text
    for c in (1,4):
        for v in ("v0","v1","v2"):
            for key in ("p50","p95","p99"):
                # Both images expose the independent median as a recorded label.
                number=f"{statistics.median(a[key] for _,a in attempts[v,c]):.0f} ms"
                data=json.loads((ROOT/"docs/chart_data.json").read_text())
                assert number in data["labels"]

    # Also check narrative values and decision labels directly from the attempts.
    b0=statistics.median(a["p99"] for _,a in attempts["v0",4])
    for i,v in enumerate(("v0","v1","v2")):
        cells=[cell.strip() for cell in human[i]]
        if v=="v0":
            assert cells[6:]==["Original reference","Starting point"]
        else:
            candidate=statistics.median(a["p99"] for _,a in attempts[v,4])
            change=(candidate-b0)/b0*100
            expected=("Rejected: " if change>0 else "Kept: ")+f"{abs(change):.0f}% "+("slower" if change>0 else "faster")
            assert cells[6:]==["Yes, all 1,000 sentences",expected]
    serial=[statistics.median(a["p50"] for _,a in attempts[v,1]) for v in ("v0","v1")]
    tails=[statistics.median(a["p99"] for _,a in attempts[v,1]) for v in ("v0","v1")]
    assert f"from {serial[0]:.0f} to {serial[1]:.0f} milliseconds" in text
    assert f"unchanged: {tails[0]:.0f} versus {tails[1]:.0f} milliseconds" in text
    assert all(statistics.median(a[key] for _,a in attempts["v1",4]) >
               statistics.median(a[key] for _,a in attempts["v0",4]) for key in ("p50","p95","p99"))


def test_portfolio_saved_graphics_and_real_freshness():
    import hashlib
    import struct
    data=json.loads((ROOT/"docs/chart_data.json").read_text())
    text=(ROOT/"README.md").read_text()
    assert data["labels"]==[label["text"] for graphic in data["graphics"].values() for label in graphic["texts"]]
    for name in ("docs/result-cards.png","docs/typical-vs-slowest.png"):
        raw=(ROOT/name).read_bytes()
        assert raw[:8]==b"\x89PNG\r\n\x1a\n"
        assert struct.unpack(">II",raw[16:24])==(2200,1120)
        assert data["image_sha256"][name]==hashlib.sha256(raw).hexdigest()
        assert re_search_link(name,text)
    assert report.main(["--check"])==0


def re_search_link(path,text):
    import re
    return re.search(r"!\[[^]]+\]\("+re.escape(path)+r"\)",text)


def test_readme_structure_language_and_preserved_archive():
    import re
    text=(ROOT/"README.md").read_text()
    assert re.findall(r"^## (.+)$",text,re.M)==[
        "What I measured and why","Results","What did not help","What I learned",
        "How to reproduce it","Limitations","Licences and provenance","Technical details"]
    human=text.split("## Technical details")[0]
    banned=r"MatMul|nearest-rank|cgroup|quantize_dynamic|opset|INT8|UINT8|tokenizer artifact|preregistered|concurrency"
    assert not re.findall(banned,human,re.I)
    original=subprocess.check_output(["git","show","4d5a44a65141d1427adab4bc94ee4c85300de5f5:README.md"],cwd=ROOT).decode()
    def rebase(match):
        target=match[1]
        if target.startswith(("http:","https:","#","mailto:")):return match[0]
        return "]("+("../"+target if not target.startswith("docs/") else target[5:])+")"
    expected=re.sub(r"\]\(([^)]+)\)",rebase,original)
    archive=(ROOT/"docs/engineering-log.md").read_text().split("<!-- original-readme-archive -->\n",1)[1]
    assert archive==expected


def test_readme_relative_links_and_png_paths():
    import re
    from urllib.parse import unquote
    text=(ROOT/"README.md").read_text()
    for target in re.findall(r"\]\(([^)]+)\)",text):
        if target.startswith(("https:","http:")):continue
        path,_,anchor=target.partition("#")
        destination=ROOT/unquote(path) if path else ROOT/"README.md"
        assert destination.exists(),target
        if anchor:
            content=destination.read_text()
            headings=re.findall(r"^#{1,6}\s+(.+)$",content,re.M)
            slugs=[re.sub(r"[^\w -]","",heading.lower()).replace(" ","-") for heading in headings]
            assert unquote(anchor) in slugs,target
