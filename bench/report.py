"""Deterministic saved-evidence report. Run: python bench/report.py [--check]."""

import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys

# Import only the frozen runner's definitions; never invoke its CLI or runtime.
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("frozen_runner", ROOT / "bench/run_load.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
DIRECTORIES = {"v0": "results/v0-baseline", "v1": "results/v1-official", "v2": "results/v2-official"}
NAMES = {"v0": "v0 PyTorch", "v1": "v1 ONNX", "v2": "v2 signed INT8"}
FROZEN = {
    "bench/run_load.py": "f61e09b2f0af90d6e64bcda44844dc87024197f354e01a5e0f8ee21f93d6ac33",
    "bench/thresholds.yaml": "3f31248914dcd93754995f23002a15bcc8e7329259b6787ff638a4d7bc795439",
    "data/sentences.jsonl": "bfe687d4cff16e7d06e8580d09b5947760095b2829fceb0ac9b5fc7f70a99be0",
}
METRICS = ("p50", "p95", "p99", "throughput_requests_per_second", "peak_container_memory",
           "startup_time_seconds", "docker_image_size_bytes")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def committed(path):
    name = path.relative_to(ROOT).as_posix()
    require(path.read_bytes() == git("show", f"HEAD:{name}"), f"Changed committed input: {name}")


def discover(root=ROOT):
    """Require exactly six original attempts and two summaries in each directory."""
    result = {}
    for version, directory in DIRECTORIES.items():
        paths = sorted((root / directory).iterdir())
        attempts = [p for p in paths if re.fullmatch(r"attempt-[0-9a-f]{32}\.json", p.name)]
        summaries = [p for p in paths if re.fullmatch(r"summary-[0-9a-f]{32}\.json", p.name)]
        require(len(attempts) == 6 and len(summaries) == 2 and len(paths) == 8,
                f"{directory}: expected exactly six attempts and two summaries")
        result[version] = (attempts, summaries)
    return result


def integrity(files):
    for name, digest in FROZEN.items():
        require(runner.sha256(ROOT / name) == digest, f"Frozen SHA-256 mismatch: {name}")
        committed(ROOT / name)
    manifest = re.findall(r"^([0-9a-f]{64})  (results/v[12]-official/\S+)",
                          (ROOT / "README.md").read_text(), re.M)
    optimized = {p.relative_to(ROOT).as_posix() for v in ("v1", "v2") for group in files[v] for p in group}
    require(len(manifest) == 16 and {n for _, n in manifest} == optimized, "README result manifest mismatch")
    for digest, name in manifest:
        require(runner.sha256(ROOT / name) == digest, f"Manifest SHA-256 mismatch: {name}")
    for version, groups in files.items():
        actual = {p.relative_to(ROOT).as_posix() for group in groups for p in group}
        tracked = set(git("ls-tree", "-r", "--name-only", "HEAD", DIRECTORIES[version]).decode().splitlines())
        require(actual == tracked, f"Original file inventory mismatch: {version}")
        for group in groups:
            for path in group:
                committed(path)


def validate_attempt(a, version, plan, rows):
    """Check saved HTTP observations, telemetry and provenance before aggregation."""
    for key in plan["mandatory_measurements"]["per_run"]:
        require(a.get(key) is not None, f"Missing mandatory metadata: {key}")
    expected = dict(version=version, mode="official", validity_status="VALID", error_count=0,
                    failure_details=[], warmup_failures=[], measured_request_failures=[],
                    warmup_request_count=100, measured_request_count=1000, actual_model_threads=4,
                    server_workers=1, cpu_limit=4, memory_limit=2147483648, attempt_number=1,
                    seed=plan["identity"]["dataset"]["seed"], dataset_sha256=FROZEN["data/sentences.jsonl"],
                    model_revision=plan["identity"]["model_revision"], cpu_architecture="aarch64",
                    load_runner_file_sha256=FROZEN["bench/run_load.py"],
                    peak_memory_measurement_method="cgroup memory.peak")
    for key, value in expected.items():
        require(a.get(key) == value, f"Invalid {key}: expected {value!r}, got {a.get(key)!r}")
    require(a["concurrency"] in (1, 4) and a["run_slot"] in (1, 2, 3), "Invalid concurrency or run slot")
    ordered = runner.seeded_order(rows, a["seed"])
    for records, expected_rows in ((a["warmup_records"], ordered[:100]), (a["request_records"], ordered)):
        require(len(records) == len(expected_rows), "Incomplete dataset coverage or warm-up count")
        for index, (record, row) in enumerate(zip(records, expected_rows)):
            require(record and record.get("sentence_id") == row["id"] and record.get("order_index") == index,
                    "Missing, duplicate or out-of-order dataset IDs")
            require(record.get("success") is True and record.get("http_status") == 200, "Failed HTTP request")
            require(runner.finite_number(record.get("client_observed_request_latency_ms")), "Incomplete latency")
            parsed = runner.response_record(json.loads(record["response_body"]))
            require(all(record.get(k) == v for k, v in parsed.items()), "Conflicting saved HTTP response or timing")
    require(a["measurement_methods"] == plan["measurement_definitions"], "Conflicting measurement definitions")
    require(a["measurement_units"] == dict(latency="milliseconds", memory="bytes", image_size="bytes", startup="seconds"),
            "Conflicting measurement units")
    require(a["runtime_identity"] == {"uid": 10001, "gid": 10001}, "Invalid runtime identity")
    require(len(a["server_processes"]) == 1 and a["server_processes"][0][-2:] == ["--workers", "1"], "Invalid workers")
    require(a["hardware"]["os_release"]["ID"] == "ubuntu" and a["hardware"]["cpuinfo"].count("processor\t:") == 4,
            "Invalid VM hardware")
    packages = {"python", "fastapi", "uvicorn", "torch", "transformers"} if version == "v0" else {
        "python", "fastapi", "uvicorn", "onnxruntime", "tokenizers"}
    require(packages <= a["relevant_software_versions"].keys(), "Incomplete software versions")
    require(re.fullmatch(r"[0-9a-f]{64}", a["container_id"]), "Invalid container identity")
    require(re.fullmatch(r"sha256:[0-9a-f]{64}", a["image_identity"]), "Invalid image identity")
    samples = a["memory_samples"]
    require(samples and all(s["method"] == "cgroup memory.peak" and runner.finite_number(s["bytes"])
                            and runner.finite_number(s["monotonic_seconds"]) for s in samples), "Incomplete memory measurements")
    require(a["peak_container_memory"] == max(s["bytes"] for s in samples), "Conflicting peak memory")
    for key in ("peak_container_memory", "startup_time_seconds", "docker_image_size_bytes"):
        require(runner.finite_number(a[key]) and a[key] > 0, f"Invalid resource measurement: {key}")
    elapsed = a["run_timing"]["measured_elapsed_seconds"]
    require(runner.finite_number(elapsed) and elapsed > 0 and a["run_timing"]["start_utc"] < a["run_timing"]["end_utc"],
            "Incomplete run timing")
    for key, value in runner.metrics(a["request_records"], elapsed).items():
        require(a[key] == value, f"Saved {key} differs from original observations")
    # The runner commit may differ by version, but must contain the frozen source.
    require(re.fullmatch(r"[0-9a-f]{40}", a["load_runner_git_commit"]), "Invalid runner Git provenance")
    require(git("show", f"{a['load_runner_git_commit']}:bench/run_load.py") == (ROOT / "bench/run_load.py").read_bytes(),
            "Runner Git source mismatch")


def compare_labels(attempts, reference, rows):
    buckets = {r["id"]: r["length_bucket"] for r in rows}
    mismatch = []
    counts = Counter({b: 0 for b in sorted(set(buckets.values()))})
    matches = total = 0
    ids = set()
    for a in attempts:
        for r in a["request_records"]:
            sid = r["sentence_id"]
            ids.add(sid)
            total += 1
            if r["predicted_label"] == reference[sid]:
                matches += 1
            else:
                mismatch.append(sid)
                counts[buckets[sid]] += 1
    return dict(unique_ids=len(ids), matching=matches, total=total, fraction=matches / total,
                mismatching_ids=sorted(set(mismatch)), mismatch_counts=dict(counts))


def spread(stat):
    return (stat["range"][1] - stat["range"][0]) / stat["median"]


def decision(baseline, candidate, quality, plan):
    rules = plan["acceptance"]
    improvement = (baseline["median"] - candidate["median"]) / baseline["median"]
    minimum = improvement >= rules["performance"]["improvement_threshold_fraction"]
    noise = improvement > spread(baseline)
    quality_pass = quality >= rules["quality"]["agreement_threshold_fraction"]
    performance = "FAIL" if not minimum else "PASS" if noise else "INCONCLUSIVE"
    outcome = "REJECT" if performance == "FAIL" or not quality_pass else "KEEP" if performance == "PASS" else "INCONCLUSIVE"
    return dict(improvement=improvement, baseline_spread=spread(baseline), minimum=minimum,
                noise=noise, quality=quality_pass, performance=performance, outcome=outcome)


def load_evidence():
    files = discover()
    integrity(files)
    plan = runner.load_plan()
    rows = runner.load_dataset(plan)
    groups, sources = {}, []
    for version, (attempt_paths, summary_paths) in files.items():
        attempts = {}
        for path in attempt_paths:
            a = read_json(path)
            try:
                validate_attempt(a, version, plan, rows)
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{path.relative_to(ROOT)}: {error}") from error
            attempts[path.relative_to(ROOT).as_posix()] = a
        for concurrency in (1, 4):
            runs = sorted((a for a in attempts.values() if a["concurrency"] == concurrency), key=lambda a: a["run_slot"])
            require([a["run_slot"] for a in runs] == [1, 2, 3], f"Missing/conflicting run slots: {version}/{concurrency}")
            groups[version, concurrency] = dict(runs=runs, statistics=runner.summary(runs, plan))
        seen = set()
        for path in summary_paths:
            s = read_json(path)
            c = s["concurrency"]
            require(c in (1, 4) and c not in seen, f"Conflicting summary: {path.name}")
            seen.add(c)
            names = [n for n, a in sorted(attempts.items(), key=lambda item: item[1]["run_slot"]) if a["concurrency"] == c]
            expected = dict(mode="official", version=version, concurrency=c, model_threads=4,
                            valid_attempt_files=names, statistics=groups[version, c]["statistics"])
            require(s == expected, f"Summary-versus-attempt mismatch: {path.relative_to(ROOT)}")
        sources.extend(p.relative_to(ROOT).as_posix() for p in attempt_paths + summary_paths)
    all_runs = {v: [a for c in (1, 4) for a in groups[v, c]["runs"]] for v in DIRECTORIES}
    reference = {r["sentence_id"]: r["predicted_label"] for r in all_runs["v0"][0]["request_records"]}
    quality = {v: compare_labels(runs, reference, rows) for v, runs in all_runs.items()}
    for version, q in quality.items():
        require(q["matching"] == q["total"], f"{version}: HTTP labels disagree with v0: {q}")
    # Check existing mandatory correctness/parity evidence; no inference is rerun.
    checks = {"results/parity/v1-tokenizer.json": "outcome", "results/parity/v1-export.json": "status",
              "results/parity/v2-quantization.json": "correctness",
              "results/checks/v1-container.json": "integration", "results/checks/v2-container.json": "integration"}
    for name, key in checks.items():
        committed(ROOT / name)
        require(read_json(ROOT / name)[key] == "PASS", f"Mandatory correctness evidence failed: {name}")
    v1_image = read_json(ROOT / "results/checks/v1-container.json")
    v2_image = read_json(ROOT / "results/checks/v2-container.json")["image"]
    for v, runs in all_runs.items():
        first = runs[0]
        for a in runs:
            for key in ("hardware", "image_identity", "docker_image_size_bytes", "relevant_software_versions", "load_runner_git_commit"):
                require(a[key] == first[key], f"Conflicting {v} metadata: {key}")
            require(a["hardware"] == all_runs["v0"][0]["hardware"], "VM changed between versions")
        if v != "v0":
            image, size = (v1_image["image_id"], v1_image["image_size_bytes"]) if v == "v1" else (v2_image["id"], v2_image["size_bytes"])
            require((first["image_identity"], first["docker_image_size_bytes"]) == (image, size), f"Wrong official image: {v}")
    decisions = {v: decision(groups["v0", 4]["statistics"]["p99"], groups[v, 4]["statistics"]["p99"],
                             quality[v]["fraction"], plan) for v in ("v1", "v2")}
    require(decisions["v1"]["outcome"] == "REJECT" and decisions["v2"]["outcome"] == "KEEP", "Unexpected registered outcomes")
    return dict(plan=plan, groups=groups, quality=quality, decisions=decisions, sources=sorted(sources), checks=checks)


def interval(stat, divisor=1, digits=6):
    values = [stat["median"], *stat["range"]]
    m, lo, hi = [f"{v / divisor:.{digits}f}" for v in values]
    return f"{m} ({lo}–{hi})"


def markdown(e):
    plan, groups = e["plan"], e["groups"]
    first = groups["v0", 1]["runs"][0]
    lines = ["# Official benchmark results", "",
             "Saved-data comparison of v0 PyTorch, v1 ONNX Runtime, and corrected v2 MatMul-only signed INT8 dynamic quantization.",
             "No new measurements or inference are performed. Diagnostic and feasibility runs are excluded.", "",
             "Regenerate from the project root: `source scripts/env.sh && PYTHONDONTWRITEBYTECODE=1 python bench/report.py`",
             "or `make report`. Verify without writing: `python bench/report.py --check` (after sourcing the environment).", "",
             "## Environment and frozen configuration", "",
             f"ARM64 Ubuntu VM: {first['hardware']['os_release']['PRETTY_NAME']}; {first['hardware']['uname'][2]}; "
             f"{first['cpu_limit']:g} CPUs, {first['memory_limit']} bytes (2 GiB) container limit, "
             f"{first['server_workers']} worker, {first['actual_model_threads']} model threads, selected by the v0-only diagnostic and frozen across versions.",
             f"Dataset: `{plan['identity']['dataset']['path']}`, seed {first['seed']}, SHA-256 `{first['dataset_sha256']}`.",
             f"Model revision: `{first['model_revision']}`. Frozen runner SHA-256: `{first['load_runner_file_sha256']}`.",
             f"Thresholds SHA-256: `{FROZEN['bench/thresholds.yaml']}`.", ""]
    for v in DIRECTORIES:
        a = groups[v, 1]["runs"][0]
        lines.append(f"- {NAMES[v]}: image `{a['image_identity']}`; runner Git commit `{a['load_runner_git_commit']}`; "
                     + ", ".join(f"{k} {val}" for k, val in sorted(a['relevant_software_versions'].items())) + ".")
    lines += ["", "## Client latency and throughput", "",
              "Every cell is the three-run median (minimum–maximum) of run-level statistics. Latency is milliseconds; throughput is requests/s."]
    for c in (4, 1):
        lines += ["", f"### Concurrency {c}" + (" — primary comparison" if c == 4 else ""), "",
                  "| Version | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (requests/s) |",
                  "| --- | --- | --- | --- | --- |"]
        for v in DIRECTORIES:
            s = groups[v, c]["statistics"]
            lines.append("| " + " | ".join([NAMES[v], *(interval(s[k]) for k in METRICS[:4])]) + " |")
    lines += ["", "[Official p99 chart](p99_by_version.svg)", "", "## Resources", "",
              "Peak container memory uses **cgroup memory.peak**; MiB = bytes / 1,048,576. "
              "Docker image size is its actual built-image size in bytes. Startup is seconds from issuing Docker start until the first correct healthy response.", "",
              "| Version | Concurrency | Peak memory (MiB) | Startup (s) | Image size (bytes) |",
              "| --- | --- | --- | --- | --- |"]
    for v, c in groups:
        s = groups[v, c]["statistics"]
        lines.append(f"| {NAMES[v]} | {c} | {interval(s['peak_container_memory'], 1024**2)} | "
                     f"{interval(s['startup_time_seconds'])} | {interval(s['docker_image_size_bytes'], digits=0)} |")
    lines += ["", "## Server-stage timings", "",
              "Median across three run-level medians (minimum–maximum of those medians), in milliseconds. "
              "server_total spans handler entry through inference, before response serialization; stages need not sum to it.", "",
              "| Version | Concurrency | tokenize | model | postprocess | server_total |",
              "| --- | --- | --- | --- | --- | --- |"]
    for v, c in groups:
        stages = groups[v, c]["statistics"]["server_stage_timings"]
        cells = [interval(dict(median=stages[k]["median_ms"], range=stages[k]["range_ms"])) for k in runner.STAGES]
        lines.append(f"| {NAMES[v]} | {c} | " + " | ".join(cells) + " |")
    lines += ["", 'Both ONNX implementations reconstruct the tokenizer for every request in `StandaloneEncoder.encode()` ([scripts/verify_tokenizer.py:124](../scripts/verify_tokenizer.py#L124)), called by [src/onnx_backend.py:65](../src/onnx_backend.py#L65); v2 uses the same encoder and inherited inference path ([src/onnx_backend_v2.py:13](../src/onnx_backend_v2.py#L13), [line 31](../src/onnx_backend_v2.py#L31)). v0 initializes its tokenizer once ([src/backend.py:34](../src/backend.py#L34)). At concurrency 4, v1 had a faster model stage than v0 (235.3 vs 345.7 ms), while the ONNX tokenize-stage medians (89.1 ms for v1 and 82.3 ms for v2, versus 0.61 ms for v0) are consistent with this per-request overhead. Its contribution to the end-to-end p99 regression was not isolated by a separate measurement.']
    lines += ["", "## Label agreement", "",
              "All six original v0 HTTP runs agree for every dataset ID. Each version is compared against these original labels across both concurrency settings.", "",
              "| Version | Unique IDs | Matching HTTP labels | Agreement | Mismatching IDs | Mismatch records by bucket |",
              "| --- | --- | --- | --- | --- | --- |"]
    for v, q in e["quality"].items():
        lines.append(f"| {v} | {q['unique_ids']} | {q['matching']}/{q['total']} | {q['fraction'] * 100:.6f}% | "
                     f"{', '.join(q['mismatching_ids']) or 'none'} | " + ", ".join(f"{k}: {n}" for k, n in q['mismatch_counts'].items()) + " |")
    lines += ["", "## Registered decisions", "",
              "Using concurrency 4, B = baseline median p99, C = candidate median p99, I = (B − C) / B, "
              "S = (baseline maximum − baseline minimum) / B. Performance requires I ≥ "
              f"{plan['acceptance']['performance']['improvement_threshold_fraction'] * 100:g}% and I > S; quality requires ≥ "
              f"{plan['acceptance']['quality']['agreement_threshold_fraction'] * 100:g}% label agreement. "
              "Performance failure or quality failure takes REJECT precedence. Meeting the minimum improvement without exceeding S is INCONCLUSIVE. "
              "KEEP also requires mandatory validity, correctness and parity checks.", ""]
    b = groups["v0", 4]["statistics"]["p99"]
    lines += [f"Baseline spread S = ({b['range'][1]:.6f} − {b['range'][0]:.6f}) / {b['median']:.6f} = {spread(b) * 100:.6f}%.", "",
              "| Candidate | Improvement calculation (ms) | I | Minimum improvement | I > S | Quality | Performance | Decision |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for v, d in e["decisions"].items():
        c = groups[v, 4]["statistics"]["p99"]["median"]
        lines.append(f"| {v} | ({b['median']:.6f} − {c:.6f}) / {b['median']:.6f} | {d['improvement'] * 100:.6f}% | "
                     f"{'PASS' if d['minimum'] else 'FAIL'} | {'PASS' if d['noise'] else 'FAIL'} | "
                     f"{'PASS' if d['quality'] else 'FAIL'} | {d['performance']} | **{d['outcome']}** |")
    lines += ["", "Mandatory saved HTTP validity, summary consistency, tokenizer/export parity, corrected signed quantization, "
              "and container integration evidence passed. The archived failed quantization experiment is not the official v2 input.", "",
              "What did not help: v1 became slower under the primary concurrency-4 workload despite preserving predictions. "
              "Its smaller image does not establish faster inference. Concurrency 1 is disclosed fully but does not decide acceptance.", "",
              "## Variation disclosures", "",
              "Relative spread = (maximum − minimum) / median. The registered disclosure threshold is strictly above "
              f"{plan['official_benchmarks']['same_version_variation']['threshold_fraction'] * 100:g}%. "
              "Valid runs are retained. The following run metrics and stage medians exceed it:", ""]
    for (v, c), group in groups.items():
        stats = group["statistics"]
        values = [(k, spread(stats[k])) for k in METRICS]
        values += [(f"{k} stage median", spread(dict(median=s['median_ms'], range=s['range_ms'])))
                   for k, s in stats['server_stage_timings'].items()]
        for key, value in values:
            if value > plan['official_benchmarks']['same_version_variation']['threshold_fraction']:
                lines.append(f"- {v}, concurrency {c}, {key}: {value * 100:.6f}%.")
    lines += ["", "## Methodology and limitations", "",
              "One ARM64 VM, a fixed generated text dataset, and three repetitions per setting. "
              "The baseline was measured earlier; virtual machine scheduling noise and changing host conditions limit comparisons. "
              "Results may differ on x86 or another system. Label agreement is not human-labeled accuracy.", "",
              "Client latency covers request issuance through receipt of the complete response. The registered nearest-rank method "
              "uses sorted observations 500, 950 and 990 for p50, p95 and p99 of each 1,000-request run. "
              "The 100 separate warm-ups are excluded. Throughput is successful measured requests / measured elapsed seconds. "
              "Server stages use saved per-request timings. Memory, startup, image size and client p99 describe different measurements.", "",
              "## Original evidence", "",
              "All 18 VALID attempts and six summaries are checked against committed bytes; summaries are checked against original observations. "
              "Each table row aggregates the three matching version/concurrency run slots from these records:", ""]
    lines.extend(f"- [{name}](../{name})" for name in e['sources'])
    lines += ["", "Existing mandatory correctness evidence:", ""]
    lines.extend(f"- [{name}](../{name})" for name in e['checks'])
    return "\n".join(lines) + "\n"


def svg(e):
    stats = {v: e['groups'][v, 4]['statistics']['p99'] for v in DIRECTORIES}
    top = math.ceil(max(s['range'][1] for s in stats.values()) / 200) * 200
    def y(value):
        return 400 - value / top * 290
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="840" height="510" viewBox="0 0 840 510" role="img" aria-labelledby="title description">',
             '<title id="title">Official p99 latency — concurrency 4</title>',
             '<desc id="description">Three-run median client p99 latency in milliseconds. Whiskers show original three-run minimum to maximum. Lower latency is better.</desc>',
             '<rect width="840" height="510" fill="white"/>',
             '<g font-family="sans-serif" fill="#172b4d">',
             '<text x="420" y="35" text-anchor="middle" font-size="23">Official p99 latency — concurrency 4</text>',
             '<text x="420" y="62" text-anchor="middle" font-size="15">Three-run medians; whiskers: min–max. Lower is better.</text>',
             '<text transform="translate(25 260) rotate(-90)" text-anchor="middle" font-size="16">p99 latency (ms)</text>']
    for tick in range(0, top + 1, 200):
        yy = y(tick)
        lines += [f'<path d="M85 {yy:.3f} H800" stroke="#d6dce5"/>',
                  f'<text x="75" y="{yy + 5:.3f}" text-anchor="end" font-size="14">{tick}</text>']
    for x, (v, s), color in zip((205, 440, 675), stats.items(), ('#315b8a', '#85531a', '#21705b')):
        mid, lo, hi = s['median'], *s['range']
        lines += [f'<rect x="{x - 60}" y="{y(mid):.3f}" width="120" height="{400-y(mid):.3f}" fill="{color}"/>',
                  f'<path d="M{x} {y(lo):.3f} V{y(hi):.3f} M{x-15} {y(lo):.3f} H{x+15} M{x-15} {y(hi):.3f} H{x+15}" stroke="#101820" stroke-width="2"/>',
                  f'<text x="{x}" y="{y(hi)-14:.3f}" text-anchor="middle" font-size="16">{mid:.6f}</text>',
                  f'<text x="{x}" y="430" text-anchor="middle" font-size="17">{NAMES[v]}</text>',
                  f'<text x="{x}" y="454" text-anchor="middle" font-size="13">{lo:.6f}–{hi:.6f} ms</text>']
    lines += ['</g>', '</svg>']
    return '\n'.join(lines) + '\n'



# Portfolio outputs remain derived from the validated original attempts.
# The renderer is imported only by render_graphics(), never by outputs()/--check.
PALETTE = {"background": "#f6f5f2", "dark": "#14151a", "secondary": "#5b5d66",
           "teal": "#0b8f80", "teal_tint": "#e6f5f2", "orange": "#c96a0a",
           "orange_tint": "#fdf0e2", "grey": "#6b6e7a", "grey_tint": "#eeeef1"}
GRAPHICS = ("docs/result-cards.png", "docs/typical-vs-slowest.png")
FONT = "DejaVu Sans"  # Inter is not installed on the authoring machine.
RENDERER_VERSION = "3.11.2"


def portfolio_data(e):
    """JSON-ready provenance, unrounded medians, display arithmetic and layout."""
    import hashlib
    workloads = []
    definitions = {
        "p50": ("ms", 1, 0), "p95": ("ms", 1, 0), "p99": ("ms", 1, 0),
        "throughput_requests_per_second": ("requests/s", 1, 1),
        "peak_container_memory": ("MB", 1_000_000, 0),
        "docker_image_size_bytes": ("GB", 1_000_000_000, 1),
        "startup_time_seconds": ("s", 1, 9),
    }
    names = {"v0": "Original", "v1": "New format", "v2": "New format + compression"}
    descriptions = {"v0": "The starting model", "v1": "Converted, not compressed",
                    "v2": "Converted and compressed"}
    sources = []
    source_index = {}
    for name in e['sources']:
        if '/attempt-' in name:
            attempt = read_json(ROOT / name)
            source_index[attempt['version'], attempt['concurrency'], attempt['run_slot']] = name
    for c, label in ((4, "Four requests at the same time"), (1, "One request at a time")):
        versions = []
        for v in DIRECTORIES:
            runs = e["groups"][v, c]["runs"]
            require(len(runs) == 3 and all(a["validity_status"] == "VALID" for a in runs),
                    "Portfolio requires three VALID runs")
            metrics = {}
            for key, (unit, divisor, digits) in definitions.items():
                values = [a[key] for a in runs]
                raw = sorted(values)[1]
                baseline = sorted(a[key] for a in e["groups"]["v0", c]["runs"])[1]
                change = (raw - baseline) / baseline * 100
                display_number = f"{raw / divisor:.{digits}f}"
                optimum = max if key == "throughput_requests_per_second" else min
                best = raw == optimum(sorted(a[key] for a in e["groups"][other, c]["runs"])[1]
                                  for other in DIRECTORIES)
                good = change > 0 if key == "throughput_requests_per_second" else change < 0
                metrics[key] = dict(run_values=values, raw_median=raw, baseline=baseline,
                                    divisor=divisor, digits=digits, unit=unit,
                                    display_number=display_number, display=f"{display_number} {unit}",
                                    change_percent=change,
                                    change_label="original" if v == "v0" else f"{change:+.0f}% vs original",
                                    change_color="grey" if v == "v0" else "teal" if good else "orange",
                                    best=best, formula="median of three VALID run-level statistics; no pooling",
                                    rounding="format fixed point, round half to even; convert units after median")
            improvement = -metrics["p99"]["change_percent"]
            badge = "baseline" if v == "v0" else f"{abs(improvement):.0f}% " + (
                "faster" if improvement > 0 else "slower")
            decision_name = "BASELINE" if v == "v0" else e["decisions"][v]["outcome"]
            category = {"BASELINE": "STARTING POINT", "REJECT": "TRIED, REJECTED",
                        "KEEP": "KEPT", "INCONCLUSIVE": "INCONCLUSIVE"}[decision_name]
            decision_label = "Starting point" if v == "v0" else (
                {"KEEP": "Kept", "REJECT": "Rejected", "INCONCLUSIVE": "Inconclusive"}[decision_name]
                + ": " + badge)
            versions.append(dict(version=v, name=names[v], description=descriptions[v],
                                 category=category, badge=badge, decision=decision_name,
                                 decision_label=decision_label, metrics=metrics,
                                 agreement=e["quality"][v]))
            for a in runs:
                path = source_index[v, c, a["run_slot"]]
                sources.append(dict(path=path, sha256=hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
                                    version=v, concurrency=c, run_slot=a["run_slot"]))
        maximum = max(row["metrics"]["p99"]["raw_median"] for row in versions)
        for row in versions:
            row["progress_fraction"] = row["metrics"]["p99"]["raw_median"] / maximum
        workloads.append(dict(concurrency=c, label=label, versions=versions))
    data = dict(schema_version=1, sources=sources, workloads=workloads,
                palette=PALETTE, rendering=dict(font=FONT, font_source="Matplotlib bundled regular and bold fonts",
                matplotlib=RENDERER_VERSION, dpi=100, png_metadata={"Software": "Measured Model Serving"},
                dimensions={name: [2200, 1120] for name in GRAPHICS},
                units={"MB": 1_000_000, "GB": 1_000_000_000, "MiB": 1_048_576},
                progress_formula="version median p99 / largest version median p99 at four requests"))
    data["readme"] = portfolio_readme(data)
    data["graphics"] = portfolio_layout(data)
    data["labels"] = [label["text"] for graphic in data["graphics"].values() for label in graphic["texts"]]
    data["image_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                            for name in GRAPHICS if (ROOT / name).is_file()}
    return data


def portfolio_readme(data):
    busy, serial = [w["versions"] for w in data["workloads"]]
    v0, v1, v2 = busy
    def metric(row, key): return row["metrics"][key]
    memory_saving = -metric(v2, "peak_container_memory")["change_percent"]
    outcome = (f'I reduced the delay reached by 99 in 100 requests from about '
               f'{metric(v0, "p99")["display_number"]} to {metric(v2, "p99")["display_number"]} milliseconds: '
               f'a {-metric(v2, "p99")["change_percent"]:.0f}% reduction.\n'
               f'Its measured memory use fell by {memory_saving:.0f}%. These results apply to this computer and workload.')
    columns = ["Version", "What it is", "Slowest 1% of requests", "Requests per second",
               "Memory", "Size on disk", "Same answers as original", "Decision"]
    table = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    exact = ["| Version | p50 ms | p95 ms | p99 ms | Requests/s | Peak memory bytes | Peak memory MiB | Image bytes | Startup seconds | Matching HTTP labels | Decision |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in busy:
        m = row["metrics"]; v = row["version"]; quality = row["agreement"]
        same = "Original reference" if v == "v0" else f'Yes, all {quality["unique_ids"]:,} sentences'
        table.append("| " + " | ".join([v, row["description"], m["p99"]["display"],
            m["throughput_requests_per_second"]["display_number"], m["peak_container_memory"]["display"],
            m["docker_image_size_bytes"]["display"], same, row["decision_label"]]) + " |")
        exact.append("| " + " | ".join([v, *(f'{m[k]["raw_median"]:.6f}' for k in ("p50","p95","p99")), str(m["throughput_requests_per_second"]["raw_median"]),
            f'{m["peak_container_memory"]["raw_median"]:.0f}', str(m["peak_container_memory"]["raw_median"] / 1048576),
            f'{m["docker_image_size_bytes"]["raw_median"]:.0f}', f'{m["startup_time_seconds"]["raw_median"]:.9f}',
            f'{quality["matching"]}/{quality["total"]}', row["decision"]]) + " |")
    results = (outcome + "\n\n" + "\n".join(table)
        + "\n\n<details>\n<summary>Exact numbers and units</summary>\n\n"
        + "\n".join(exact)
        + "\n\nEach value is the median of three valid run-level values at four simultaneous requests. "
          "Memory includes startup and warm-ups. MiB means bytes / 1,048,576. "
          "Size is Docker's built-image size. The full report retains all ranges and timing definitions.\n\n</details>\n\n"
          "The slowest-1% column is the 99-in-100 timing threshold. Percent faster means a reduction in that delay.\n"
          "Both other versions matched the original answers for all "
        + f'{v2["agreement"]["unique_ids"]:,} fixed test sentences across all six runs each. '
          "Matching answers does not measure accuracy against human labels.")
    s0, s1, _ = serial
    comparison = (f'With one request at a time, v1 improved the typical wait from '
                 f'{metric(s0, "p50")["display_number"]} to {metric(s1, "p50")["display_number"]} milliseconds.\n'
                 f'Its slowest-1% time was almost unchanged: {metric(s0, "p99")["display_number"]} versus '
                 f'{metric(s1, "p99")["display_number"]} milliseconds.\n\n'
                 'With four requests at the same time, v1 was slower at every reported delay point.\n'
                 'The ONNX version helped when requests came one by one, but not when several arrived together.\n'
                 'I decided in advance that the busy case decides, so I rejected v1 as built.')
    rejection = (f'As built, the ONNX FP32 version made the slowest requests about '
                 f'{metric(v1, "p99")["change_percent"]:.0f}% slower in the busy test, so I rejected it.\n'
                 'Compressing the converted model produced the observed improvement.\n'
                 'A smaller download did not guarantee a shorter wait.')
    return dict(results=results, findings=comparison, rejection=rejection)


def portfolio_layout(data):
    """All visible text lives in chart_data.json, with explicit allotted regions."""
    images = {name: dict(size=[2200, 1120], texts=[], cards=[], tracks=[]) for name in GRAPHICS}
    def text(image, ident, value, x, y, size, region, color="dark", weight="normal", align="left"):
        images[image]["texts"].append(dict(id=ident, text=value, x=x, y=y, size=size,
                                          region=region, color=color, weight=weight, align=align))
    d1, d3 = GRAPHICS
    header = [50, 20, 2150, 195]
    text(d1, "title", "Evaluating CPU Inference Optimizations", 60, 45, 34, header, weight="bold")
    text(d1, "subtitle", "Four requests at the same time · Middle result of three runs on one computer",
         60, 125, 22, header, color="secondary")
    for i, row in enumerate(data["workloads"][0]["versions"]):
        x = 60 + i * 705
        region = [x + 22, 250, x + 648, 952]
        accent = {"v0": "grey", "v1": "orange", "v2": "teal"}[row["version"]]
        images[d1]["cards"].append(dict(rect=[x, 230, 670, 750], accent=accent,
                                       border=3 if row["version"] == "v2" else 1))
        text(d1, f"{i}-category", row["category"], x+28, 265, 13, region, accent, "bold")
        text(d1, f"{i}-badge", row["badge"], x+626, 265, 14, region, accent, "bold", "right")
        text(d1, f"{i}-name", row["version"]+" · "+row["name"], x+28, 335, 24, region, weight="bold")
        text(d1, f"{i}-description", row["description"], x+28, 397, 18, region, "secondary")
        text(d1, f"{i}-p99", row["metrics"]["p99"]["display"], x+28, 475, 52, region, accent, "bold")
        text(d1, f"{i}-caption", "slowest 1% of requests", x+28, 575, 18, region, "secondary")
        images[d1]["tracks"].append(dict(rect=[x+28, 625, 614, 8],
                                        fraction=row["progress_fraction"], color=accent))
        for j, (key, label) in enumerate((("throughput_requests_per_second", "Requests per second"),
                                          ("peak_container_memory", "Memory"),
                                          ("docker_image_size_bytes", "Size on disk"))):
            m=row["metrics"][key]; y=710+j*78
            value=m["display_number"] if key=="throughput_requests_per_second" else m["display"]
            text(d1, f"{i}-{key}-label", label, x+28, y, 16, region, "secondary")
            text(d1, f"{i}-{key}-value", value, x+626, y-4, 23, region, weight="bold", align="right")
            text(d1, f"{i}-{key}-delta", m["change_label"], x+626, y+43, 13, region, m["change_color"], align="right")
    count=data["workloads"][0]["versions"][2]["agreement"]["unique_ids"]
    text(d1, "agreement", f"Both other versions gave the same answers as the original on all {count:,} test sentences.",
         60, 1035, 19, [50, 1010, 2150, 1100], "secondary")
    text(d3, "title", "Typical, slow and slowest", 60, 45, 34, header, weight="bold")
    text(d3, "subtitle", "Typical = middle request · Slow = 95 in 100 · Slowest = 99 in 100",
         60, 125, 22, header, "secondary")
    for i, workload in enumerate(data["workloads"]):
        x=60+i*1080; region=[x+22, 250, x+998, 960]
        images[d3]["cards"].append(dict(rect=[x,230,1020,750],accent="grey",border=1))
        text(d3,f"{i}-workload",workload["label"],x+30,270,24,region,weight="bold")
        text(d3,f"{i}-qualifier","the main test" if i==0 else "the quieter test",
             x+30,322,17,region,"secondary")
        for col,(key,label) in enumerate((("p50","Typical\nrequest"),("p95","Slow\nrequest"),("p99","Slowest 1%\nof requests"))):
            cx=x+255+col*290
            text(d3,f"{i}-{key}-heading",label,cx,405,18,region,"secondary",align="center")
            for j,row in enumerate(workload["versions"]):
                y=535+j*140; m=row["metrics"][key]
                if col==0: text(d3,f"{i}-{j}-version",row["version"],x+32,y+6,22,region,weight="bold")
                text(d3,f"{i}-{j}-{key}-value",m["display"],cx,y,32,region,
                     "teal" if m["best"] else "dark","bold" if m["best"] else "normal","center")
                if m["best"]:text(d3,f"{i}-{j}-{key}-best","best",cx,y+68,13,region,"teal","bold","center")
    text(d3,"method","Each cell is the middle of three run-level measurements · Lower times are better",
         60,1035,19,[50,1010,2150,1100],"secondary")
    return images


def render_graphics(data):
    """Development-only, deterministic PNG rendering and measured text geometry."""
    import io
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties, findfont
    from matplotlib.patches import FancyBboxPatch
    require(matplotlib.__version__ == RENDERER_VERSION, "Install the pinned development renderer")
    images, geometry = {}, {}
    with matplotlib.rc_context({"font.family": FONT, "text.antialiased": True,
                                "text.usetex": False, "svg.hashsalt": "measured-model-serving"}):
        for name, layout in data["graphics"].items():
            w,h=layout["size"]
            figure=Figure(figsize=(w/100,h/100),dpi=100,facecolor=PALETTE["background"])
            canvas=FigureCanvasAgg(figure)
            ax=figure.add_axes([0,0,1,1]);ax.set(xlim=(0,w),ylim=(h,0));ax.axis("off")
            for card in layout["cards"]:
                x,y,cw,ch=card["rect"];accent=card["accent"]
                ax.add_patch(FancyBboxPatch((x,y),cw,ch,boxstyle="round,pad=0,rounding_size=22",
                             facecolor="white",edgecolor=PALETTE[accent] if card["border"]>1 else "#dedee2",
                             linewidth=card["border"]))
                if name==GRAPHICS[0]:
                    ax.plot([x+28,x+cw-28],[665,665],color="#e4e4e8",linewidth=1)
                    ax.add_patch(FancyBboxPatch((x+440,251),195,45,
                                 boxstyle="round,pad=0,rounding_size=18",
                                 facecolor=PALETTE[accent+"_tint"],edgecolor="none"))
            for track in layout["tracks"]:
                x,y,cw,ch=track["rect"]
                for fraction,color in ((1,"grey_tint"),(track["fraction"],track["color"])):
                    ax.add_patch(FancyBboxPatch((x,y),cw*fraction,ch,boxstyle="round,pad=0,rounding_size=4",
                                              facecolor=PALETTE[color],edgecolor="none"))
            artists=[]
            for label in layout["texts"]:
                font=FontProperties(family=FONT,weight=label["weight"],size=label["size"])
                # Resolve exactly the bundled requested family; never silently substitute.
                path=findfont(font,fallback_to_default=False)
                require(Path(path).name in ("DejaVuSans.ttf","DejaVuSans-Bold.ttf"),"Unexpected font substitution")
                artist=ax.text(label["x"],label["y"],label["text"],fontproperties=font,
                               color=PALETTE[label["color"]],ha=label["align"],va="top",linespacing=1.3)
                artists.append((label,artist,path))
            canvas.draw(); renderer=canvas.get_renderer();boxes=[]
            for label,artist,path in artists:
                box=artist.get_window_extent(renderer)
                boxes.append(dict(id=label["id"],text=label["text"],
                                  bbox=[box.x0,h-box.y1,box.x1,h-box.y0],region=label["region"],
                                  font=Path(path).name,font_sha256=runner.sha256(Path(path))))
            buffer=io.BytesIO()
            canvas.print_png(buffer,metadata=data["rendering"]["png_metadata"])
            images[name]=buffer.getvalue();geometry[name]=boxes
            figure.clear()
    return images,geometry


def editorial_readme(data):
    """Render current README prose while retaining chart JSON's legacy text."""
    busy, serial = [w["versions"] for w in data["workloads"]]
    v0, v1, v2 = busy
    def metric(row, key): return row["metrics"][key]
    outcome = (f'Under four concurrent requests, v2 reduced median run-level p99 latency from '
               f'{metric(v0, "p99")["display_number"]} to {metric(v2, "p99")["display_number"]} milliseconds: '
               f'a {-metric(v2, "p99")["change_percent"]:.0f}% reduction.\n'
               f'Peak container memory use fell by {-metric(v2, "peak_container_memory")["change_percent"]:.0f}%. '
               'These results apply to this computer and workload.')
    columns = ["Version", "Implementation", "p99 latency", "Throughput (requests/s)",
               "Peak container memory", "Container image size", "Prediction-label agreement", "Decision"]
    table = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    descriptions = {"v0": "PyTorch / Transformers", "v1": "ONNX Runtime FP32",
                    "v2": "ONNX Runtime, dynamic INT8"}
    for row in busy:
        m = row["metrics"]; v = row["version"]
        change = m["p99"]["change_percent"]
        decision = "Baseline" if v == "v0" else (
            {"KEEP": "Kept", "REJECT": "Rejected", "INCONCLUSIVE": "Inconclusive"}[row["decision"]]
            + f' — approximately {abs(change):.0f}% ' + ("higher" if change > 0 else "lower") + " p99 latency")
        agreement = "Original reference" if v == "v0" else f'Yes, all {row["agreement"]["unique_ids"]:,} sentences'
        table.append("| " + " | ".join([v, descriptions[v], m["p99"]["display"],
            m["throughput_requests_per_second"]["display_number"], m["peak_container_memory"]["display"],
            m["docker_image_size_bytes"]["display"], agreement, decision]) + " |")
    # Preserve the complete exact-numbers table and its methodological notes.
    exact = data["readme"]["results"].split("<details>", 1)[1].split("</details>", 1)[0]
    results = (outcome + "\n\n" + "\n".join(table) + "\n\n<details>" + exact + "</details>\n\n"
        "p99 latency is the response-time threshold within which 99% of measured requests complete. "
        "Lower values indicate better performance near the slow end of the distribution.\n"
        "The published p99 is the median of three run-level p99 values, calculated separately using the nearest-rank method; "
        "requests were not pooled across runs.\n"
        "Throughput measures completed requests per second. Peak container memory includes startup and warm-ups; "
        "container image size measures storage, not running memory.\n"
        f'Both optimized implementations matched baseline prediction labels for all {v2["agreement"]["unique_ids"]:,} '
        "fixed test sentences across all six runs each. Prediction-label agreement does not measure accuracy against human labels.")
    s0, s1, _ = serial
    findings = (f'With individual requests, v1 reduced median run-level p50 latency from '
        f'{metric(s0, "p50")["display_number"]} to {metric(s1, "p50")["display_number"]} milliseconds.\n'
        f'Its p99 latency was almost unchanged: {metric(s0, "p99")["display_number"]} versus '
        f'{metric(s1, "p99")["display_number"]} milliseconds.\n\n'
        "With four concurrent requests, v1 had higher p50, p95, and p99 latency than v0.\n"
        "The conversion helped typical single-request latency, but regressed under the primary workload.\n"
        "I rejected v1 under the predefined acceptance criteria; the single-request improvement did not override the concurrent-workload result.")
    rejection = (f'I rejected ONNX Runtime FP32 because it increased p99 latency by approximately '
        f'{metric(v1, "p99")["change_percent"]:.0f}% under the primary concurrent workload.\n'
        "A smaller container image did not necessarily mean lower inference latency.\n"
        "The corrected dynamic INT8 implementation met the performance and prediction-label agreement criteria and was selected for this experiment.")
    return dict(results=results, findings=findings, rejection=rejection)


def readme_output(data):
    text = (ROOT / "README.md").read_text()
    for key, value in editorial_readme(data).items():
        opening = "<!-- portfolio:" + key + " -->"
        closing = "<!-- /portfolio:" + key + " -->"
        require(text.count(opening) == text.count(closing) == 1,
                f"Missing README generated block: {key}")
        before, tail = text.split(opening, 1)
        _, after = tail.split(closing, 1)
        text = before + opening + "\n" + value + "\n" + closing + after
    return text


def check_graphics():
    """Check saved image content hashes via chart JSON, plus PNG type/dimensions."""
    import struct
    for name in GRAPHICS:
        path=ROOT/name
        require(path.is_file(),f"Missing graphic: {name}; render with pinned development dependencies")
        raw=path.read_bytes()
        require(raw[:8]==b"\x89PNG\r\n\x1a\n" and len(raw)>24,f"Invalid PNG: {name}")
        require(struct.unpack(">II",raw[16:24])==(2200,1120),f"Wrong PNG dimensions: {name}")


def outputs(e):
    data = portfolio_data(e)
    return {'docs/benchmark_report.md': markdown(e), 'docs/p99_by_version.svg': svg(e),
            'docs/chart_data.json': json.dumps(data, indent=2, ensure_ascii=False) + '\n',
            'README.md': readme_output(data)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='validate evidence and check output freshness without writing')
    args = parser.parse_args(argv)
    try:
        evidence = load_evidence()
        if not args.check:
            images, _ = render_graphics(portfolio_data(evidence))
            for name, content in images.items():
                (ROOT / name).write_bytes(content)
        rendered = outputs(evidence)
        for name, content in rendered.items():
            path = ROOT / name
            if args.check:
                require(path.is_file() and path.read_bytes() == content.encode('utf-8'), f'Stale or missing output: {name}')
            else:
                path.write_bytes(content.encode('utf-8'))
        check_graphics()
        print('PASS: 18 VALID attempts, six consistent summaries, HTTP label agreement, registered decisions; '
              + ('outputs fresh' if args.check else 'wrote ' + ', '.join((*GRAPHICS, *rendered))))
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as error:
        print(f'Report evidence error: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
