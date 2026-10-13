"""Registered load runner. Importing this module performs no measurements."""

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import uuid

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve()
STAGES = ("tokenize", "model", "postprocess", "server_total")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_plan():
    return yaml.safe_load((ROOT / "bench/thresholds.yaml").read_text())


def load_dataset(plan):
    definition = plan["identity"]["dataset"]
    path = (ROOT / definition["path"]).resolve()
    if not path.is_relative_to(ROOT) or sha256(path) != definition["sha256"]:
        raise ValueError("Dataset path or SHA-256 mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        if not isinstance(row, dict) or not {"id", "text", "length_bucket"} <= row.keys():
            raise ValueError("Missing dataset fields")
        if not all(isinstance(row[key], str) and row[key].strip()
                   for key in ("id", "text", "length_bucket")) or len(row["text"]) > 2000:
            raise ValueError("Invalid dataset fields")
    if len(rows) != definition["sentences"] or len({r["id"] for r in rows}) != definition["sentences"]:
        raise ValueError("Dataset must have exactly 1000 distinct IDs")
    if Counter(r["length_bucket"] for r in rows) != definition["buckets"]:
        raise ValueError("Dataset bucket counts mismatch")
    return rows


def seeded_order(rows, seed):
    result = list(rows)
    random.Random(seed).shuffle(result)
    return result


def parser(plan):
    rules = plan["common_run_rules"]
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--version", choices=plan["official_benchmarks"]["versions"], required=True)
    cli.add_argument("--mode", choices=("diagnostic", "official"), required=True)
    cli.add_argument("--concurrency", type=int, required=True)
    cli.add_argument("--model-threads", type=int, required=True)
    cli.add_argument("--image", required=True)
    cli.add_argument("--warmup", type=int, default=rules["warmup_requests"])
    cli.add_argument("--requests", type=int, default=rules["measured_requests"])
    cli.add_argument("--seed", type=int, default=plan["identity"]["dataset"]["seed"])
    cli.add_argument("--runs", type=int, default=plan["official_benchmarks"]["valid_runs_per_version_and_concurrency"])
    cli.add_argument("--output-dir", type=Path, default=ROOT / "results")
    return cli


def validate_config(args, plan):
    rules = plan["common_run_rules"]
    expected = {"warmup": rules["warmup_requests"], "requests": rules["measured_requests"],
                "seed": plan["identity"]["dataset"]["seed"],
                "runs": plan["official_benchmarks"]["valid_runs_per_version_and_concurrency"]}
    for key, value in expected.items():
        if getattr(args, key) != value:
            raise ValueError(f"--{key} must equal registered value {value}")
    if args.version not in plan["official_benchmarks"]["versions"]:
        raise ValueError("Unsupported version")
    if args.mode == "diagnostic":
        rule = plan["thread_diagnostic"]
        if (args.version != rule["selection_version"] or args.concurrency != rule["concurrency"]
                or args.model_threads not in rule["candidate_model_threads"]):
            raise ValueError("Diagnostic requires v0, concurrency 4, threads 4, 2 or 1")
    elif args.mode != "official" or args.concurrency not in plan["official_benchmarks"]["concurrency"]:
        raise ValueError("Official concurrency must be 1 or 4")
    if args.model_threads < 1 or not args.image.strip() or args.image.startswith("-"):
        raise ValueError("Positive model threads and a Docker image are required")
    if not args.output_dir.resolve().is_relative_to(ROOT):
        raise ValueError("Results directory must be inside the project")
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise ValueError("Results path is not a directory")


def command(argv):
    try:
        result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"Command {argv!r} timed out; stdout={error.stdout!r}; stderr={error.stderr!r}") from error
    if result.returncode:
        message = f"Command {argv!r} exited {result.returncode}; stdout={result.stdout!r}; stderr={result.stderr!r}"
        if "permission denied" in result.stderr.lower():
            message += " Docker socket access denied: ask the administrator for access, then rerun; no sudo or group changes were attempted."
        raise RuntimeError(message)
    return result.stdout


async def await_completion(task):
    """Defer caller cancellation until owned work finishes, then propagate it."""
    cancellation = None
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as error:
            if task.cancelled():
                raise
            cancellation = cancellation or error
        except Exception as error:
            if cancellation is not None:
                raise cancellation from error
            raise
    if cancellation is not None:
        raise cancellation
    return result


async def docker(*args):
    # A cancelled to_thread await cannot stop its subprocess. Retain and drain it
    # before the caller can enter cleanup; command() still enforces its timeout.
    task = asyncio.create_task(asyncio.to_thread(command, ["docker", *args]))
    return await await_completion(task)


def provenance(mode):
    head = command(["git", "rev-parse", "HEAD"]).strip()
    if mode == "official":
        committed = command(["git", "show", "HEAD:bench/run_load.py"])
        if committed.encode("utf-8") != SOURCE.read_bytes():
            raise ValueError("Official runs require this runner frozen and committed with separate explicit user authorization")
    return {"load_runner_git_commit": head, "load_runner_file_sha256": sha256(SOURCE),
            "load_runner_software_versions": {"python": platform.python_version(),
                                              "httpx": importlib.metadata.version("httpx"),
                                              "PyYAML": importlib.metadata.version("PyYAML")},
            "cpu_architecture": platform.machine(),
            "hardware": {"uname": list(os.uname()),
                         "cpuinfo": (Path("/proc/cpuinfo").read_text()),
                         "os_release": platform.freedesktop_os_release()}}


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def response_record(payload):
    if not isinstance(payload, dict) or set(payload) != {"label", "score", "truncated", "timings_ms"}:
        raise ValueError("Response fields do not match API contract")
    if payload["label"] not in ("POSITIVE", "NEGATIVE"):
        raise ValueError("Invalid label")
    if not finite_number(payload["score"]) or payload["score"] > 1 or type(payload["truncated"]) is not bool:
        raise ValueError("Invalid score or truncated flag")
    if payload["truncated"]:
        raise ValueError("Measured dataset input was unexpectedly truncated")
    timings = payload["timings_ms"]
    if not isinstance(timings, dict) or set(timings) != set(STAGES) or not all(finite_number(v) for v in timings.values()):
        raise ValueError("Invalid timing fields")
    return {"predicted_label": payload["label"], "score": payload["score"],
            "truncated": payload["truncated"], **{f"{key}_ms": timings[key] for key in STAGES}}


async def phase(client, rows, concurrency, clock=None, observations=None):
    """A fixed worker pool bounds both active requests and task count."""
    clock = clock or time.perf_counter_ns
    records, failures = observations if observations is not None else [], []
    records.extend([None] * len(rows))
    iterator = iter(enumerate(rows))

    async def worker():
        for index, row in iterator:
            start = clock()
            observation = {"sentence_id": row["id"], "order_index": index}
            try:
                response = await client.post("/classify", json={"text": row["text"]})
                observation["client_observed_request_latency_ms"] = (clock() - start) / 1_000_000
                observation["http_status"] = response.status_code
                observation["response_body"] = response.text
                if response.status_code != 200:
                    raise ValueError(f"HTTP {response.status_code}")
                observation.update(response_record(response.json()))
                observation["success"] = True
            except Exception as error:
                observation.setdefault("client_observed_request_latency_ms", (clock() - start) / 1_000_000)
                observation.update(success=False, error_type=type(error).__name__, error=str(error))
                failures.append(dict(observation))
            records[index] = observation

    start = clock()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return records, failures, (clock() - start) / 1_000_000_000


def nearest_rank(values, percentile):
    if not values or not 0 < percentile <= 100:
        raise ValueError("Percentile requires data and 0 < percentile <= 100")
    return sorted(values)[math.ceil(percentile / 100 * len(values)) - 1]


def metrics(records, elapsed):
    successful = [r for r in records if r and r.get("success")]
    if not successful or elapsed <= 0:
        return {}
    values = [r["client_observed_request_latency_ms"] for r in successful]
    return {**{f"p{p}": nearest_rank(values, p) for p in (50, 95, 99)},
            "throughput_requests_per_second": len(successful) / elapsed,
            "server_stage_timings": {key: {"median_ms": statistics.median(r[f"{key}_ms"] for r in successful),
                                          "range_ms": [min(r[f"{key}_ms"] for r in successful), max(r[f"{key}_ms"] for r in successful)]}
                                     for key in STAGES}}


# Read the counter only when the mount maps to this process's own unified cgroup.
MEMORY_SCRIPT = """
import json, pathlib
group = next((x.split(':', 2)[2] for x in pathlib.Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::')), None)
result = {}
for line in pathlib.Path('/proc/self/mountinfo').read_text().splitlines():
    left, right = line.split(' - ', 1)
    fields = left.split()
    if right.split()[0] == 'cgroup2' and group is not None:
        root, mount = fields[3:5]
        if group == root or group.startswith(root.rstrip('/') + '/'):
            path = pathlib.Path(mount) / group[len(root):].lstrip('/')
            for key in ('peak', 'current'):
                try: result[key] = int((path / ('memory.' + key)).read_text())
                except (OSError, ValueError): pass
print(json.dumps(result))
"""

RUNTIME_SCRIPT = """
import importlib.metadata as m, json, os, pathlib, platform
packages = {}
for name in ('fastapi', 'uvicorn', 'transformers', 'torch', 'onnxruntime', 'tokenizers', 'numpy'):
    try: packages[name] = m.version(name)
    except m.PackageNotFoundError: pass
servers = []
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
    try:
        args = p.read_bytes().split(b'\\0')
        if b'uvicorn' in args and b'src.app:app' in args: servers.append([a.decode() for a in args if a])
    except (OSError, UnicodeError): pass
print(json.dumps({'uid': os.getuid(), 'gid': os.getgid(), 'model_threads': os.environ['MODEL_THREADS'],
                  'packages': {'python': platform.python_version(), **packages}, 'server_processes': servers}))
"""


def create_args(args, plan, name):
    limits = plan["common_run_rules"]["container"]
    return ["create", "--pull=never", "--name", name, f"--cpus={limits['cpus']}",
            f"--memory={limits['memory_limit']}", "--user=10001:10001", "-e", f"MODEL_THREADS={args.model_threads}",
            "-p", "127.0.0.1::8000", "--entrypoint", "python", args.image,
            "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000",
            "--workers", str(limits["server_workers"])]


async def metadata(container, args, plan):
    inspection = json.loads(await docker("inspect", container))[0]
    image = json.loads(await docker("image", "inspect", inspection["Image"]))[0]
    runtime = json.loads(await docker("exec", container, "python", "-c", RUNTIME_SCRIPT))
    limits = plan["common_run_rules"]["container"]
    config = inspection["HostConfig"]
    if config["NanoCpus"] != limits["cpus"] * 1_000_000_000 or config["Memory"] != limits["memory_limit_bytes"]:
        raise ValueError("Actual Docker resource limits differ from registration")
    if runtime["uid"] == 0 or int(runtime["model_threads"]) != args.model_threads:
        raise ValueError("Runtime identity or MODEL_THREADS mismatch")
    servers = runtime["server_processes"]
    if len(servers) != 1 or servers[0][-2:] != ["--workers", str(limits["server_workers"])]:
        raise ValueError("Cannot verify exactly one Uvicorn server worker")
    required = {"python", "fastapi", "uvicorn", "transformers", "torch"} if args.version == "v0" else {"python", "fastapi", "uvicorn", "onnxruntime", "tokenizers"}
    if not required <= runtime["packages"].keys() or not isinstance(image["Size"], int) or image["Size"] <= 0:
        raise ValueError("Required software versions or image size unavailable")
    return {"container_id": inspection["Id"], "image_identity": inspection["Image"],
            "image_repo_digests": image.get("RepoDigests", []), "docker_image_size_bytes": image["Size"],
            "cpu_limit": config["NanoCpus"] / 1_000_000_000, "memory_limit": config["Memory"],
            "runtime_identity": {"uid": runtime["uid"], "gid": runtime["gid"]},
            "actual_model_threads": int(runtime["model_threads"]), "server_workers": len(servers),
            "relevant_software_versions": runtime["packages"], "server_processes": servers,
            "model_threads_measurement_method": "runtime MODEL_THREADS environment used by backend"}


async def memory_read(container):
    values = json.loads(await docker("exec", container, "python", "-c", MEMORY_SCRIPT))
    if "peak" in values:
        return values["peak"], "cgroup memory.peak"
    if "current" in values:
        return values["current"], "sampled peak"
    usage = json.loads(await docker("stats", "--no-stream", "--format", "{{json .}}", container))["MemUsage"].split(" / ")[0]
    units = {"B": 1, "kB": 1000, "MB": 1000**2, "GB": 1000**3, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}
    for unit in sorted(units, key=len, reverse=True):
        if usage.endswith(unit):
            return math.ceil(float(usage[:-len(unit)]) * units[unit]), "sampled peak"
    raise ValueError("Cannot parse actual Docker memory statistics")


async def monitor_memory(container, stop, record):
    last, methods = time.perf_counter(), set()
    while True:
        value, method = await memory_read(container)
        now = time.perf_counter()
        if not finite_number(value):
            raise ValueError("Invalid memory counter")
        if now - last > 1 and (method == "sampled peak" or "sampled peak" in methods):
            raise ValueError("Memory fallback sampling interval exceeded one second")
        methods.add(method)
        if len(methods) != 1:
            raise ValueError("Memory measurement method changed during attempt")
        record.setdefault("memory_samples", []).append({"monotonic_seconds": now, "bytes": value, "method": method})
        record["peak_container_memory"] = max(record.get("peak_container_memory") or 0, value)
        record["peak_memory_measurement_method"] = "sampled peak" if "sampled peak" in methods else method
        last = now
        if stop.is_set():
            return
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.5)
        except TimeoutError:
            pass


async def healthy(client, version):
    deadline = time.perf_counter() + 120
    errors = []
    while time.perf_counter() < deadline:
        try:
            response = await client.get("/health", timeout=2)
            if response.status_code == 200 and response.json() == {"status": "ok", "version": version}:
                return
            errors.append({"status": response.status_code, "body": response.text})
        except (httpx.HTTPError, ValueError) as error:
            errors.append({"type": type(error).__name__, "error": str(error)})
        await asyncio.sleep(0.1)
    raise TimeoutError(f"Startup health deadline exceeded: {errors!r}")


def validity(record, rows, plan):
    errors = record["failure_details"]
    warmups, measured = record["warmup_records"], record["request_records"]
    rules = plan["common_run_rules"]
    if len(warmups) != rules["warmup_requests"] or not all(r and r.get("success") for r in warmups):
        errors.append({"error": "Incomplete or failed warm-up phase"})
    expected = Counter(r["id"] for r in rows)
    if len(measured) != rules["measured_requests"] or Counter(r["sentence_id"] for r in measured if r) != expected:
        errors.append({"error": "Missing or duplicate measured IDs"})
    required = plan["mandatory_measurements"]["per_request"]
    if not all(r and r.get("success") and all(key in r and r[key] is not None for key in required)
               and finite_number(r["client_observed_request_latency_ms"]) for r in measured):
        errors.append({"error": "Incomplete or failed measured request records"})
    missing = [key for key in plan["mandatory_measurements"]["per_run"] if record.get(key) is None]
    if missing:
        errors.append({"error": "Missing mandatory metadata", "fields": missing})
    record["validity_status"] = "INVALID" if errors else "VALID"
    record["error_count"] = len(errors)


async def attempt(args, plan, rows, origin, slot, number):
    record = {key: None for key in plan["mandatory_measurements"]["per_run"]}
    record.update(origin, version=args.version, concurrency=args.concurrency, mode=args.mode,
                  run_slot=slot, attempt_number=number, validity_status="INVALID", error_count=0,
                  failure_details=[], warmup_failures=[], measured_request_failures=[], warmup_records=[], request_records=[],
                  warmup_request_count=0, measured_request_count=0, seed=args.seed,
                  dataset_sha256=plan["identity"]["dataset"]["sha256"], model_revision=plan["identity"]["model_revision"],
                  measurement_units={"latency": "milliseconds", "memory": "bytes", "image_size": "bytes", "startup": "seconds"},
                  measurement_methods=plan["measurement_definitions"], run_timing={"start_utc": datetime.now(timezone.utc).isoformat()})
    name = f"measured-load-{uuid.uuid4().hex}"
    created, monitor, stop = False, None, asyncio.Event()
    try:
        # The unique name also permits cleanup after an interrupted create response.
        created = True
        container = (await docker(*create_args(args, plan, name))).strip()
        record["container_id"] = container
        start = time.perf_counter_ns()
        await docker("start", container)
        monitor = asyncio.create_task(monitor_memory(container, stop, record))
        inspection = json.loads(await docker("inspect", container))[0]
        bindings = inspection["NetworkSettings"]["Ports"]["8000/tcp"]
        if len(bindings) != 1 or bindings[0]["HostIp"] != "127.0.0.1":
            raise ValueError("Service must be published only on localhost")
        url = f"http://127.0.0.1:{int(bindings[0]['HostPort'])}"
        async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=httpx.Timeout(60),
                                     limits=httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)) as client:
            await healthy(client, args.version)
            record["startup_time_seconds"] = (time.perf_counter_ns() - start) / 1_000_000_000
            record.update(await metadata(container, args, plan))
            ordered = seeded_order(rows, args.seed)
            warmups, failures, _ = await phase(client, ordered[:args.warmup], args.concurrency,
                                               observations=record["warmup_records"])
            record.update(warmup_records=warmups, warmup_failures=failures, warmup_request_count=len(warmups))
            measured, failures, elapsed = await phase(client, ordered, args.concurrency,
                                                     observations=record["request_records"])
            record.update(request_records=measured, measured_request_failures=failures, measured_request_count=len(measured))
            record["run_timing"]["measured_elapsed_seconds"] = elapsed
            record.update(metrics(measured, elapsed))
    except asyncio.CancelledError as error:
        record["interrupted"] = True
        record["failure_details"].append({"error_type": type(error).__name__, "error": "Execution interrupted"})
        if error.__cause__ is not None:
            record["failure_details"].append({"error_type": type(error.__cause__).__name__, "error": str(error.__cause__)})
    except Exception as error:
        record["failure_details"].append({"error_type": type(error).__name__, "error": str(error)})
    finally:
        async def cleanup():
            stop.set()
            if monitor is not None:
                try:
                    await monitor
                except (Exception, asyncio.CancelledError) as error:
                    record["failure_details"].append({"error_type": type(error).__name__, "error": str(error), "phase": "memory"})
            if created:
                try:
                    await docker("rm", "-f", name)
                except Exception as error:
                    record["failure_details"].append({"error_type": type(error).__name__, "error": str(error), "phase": "cleanup"})

        # Also defer repeated cancellation while draining telemetry or removing
        # the container, so neither step can be skipped by a second interrupt.
        try:
            await await_completion(asyncio.create_task(cleanup()))
        except asyncio.CancelledError as error:
            record["interrupted"] = True
            record["failure_details"].append({"error_type": type(error).__name__, "error": "Execution interrupted during cleanup"})
            if error.__cause__ is not None:
                record["failure_details"].append({"error_type": type(error.__cause__).__name__, "error": str(error.__cause__), "phase": "cleanup"})
        record["run_timing"]["end_utc"] = datetime.now(timezone.utc).isoformat()
        for phase_name, records_key, failures_key, count_key in (
                ("warmup", "warmup_records", "warmup_failures", "warmup_request_count"),
                ("measured", "request_records", "measured_request_failures", "measured_request_count")):
            observations = [r for r in record[records_key] if r]
            record[count_key] = len(observations)
            record[failures_key] = [r for r in observations if not r.get("success")]
            record["failure_details"].extend({"phase": phase_name, **r} for r in record[failures_key])
        validity(record, rows, plan)
    return record


def save_record(directory, record, kind="attempt"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{kind}-{uuid.uuid4().hex}.json"
    with path.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return path


def summary(valid, plan):
    required = plan["official_benchmarks"]["valid_runs_per_version_and_concurrency"]
    if len(valid) != required or any(r["validity_status"] != "VALID" for r in valid):
        raise ValueError("Summary requires exactly three VALID repetitions")
    keys = ("p50", "p95", "p99", "throughput_requests_per_second", "startup_time_seconds",
            "peak_container_memory", "docker_image_size_bytes")
    result = {key: {"median": statistics.median(r[key] for r in valid),
                    "range": [min(r[key] for r in valid), max(r[key] for r in valid)]} for key in keys}
    result["server_stage_timings"] = {stage: {"median_ms": statistics.median(r["server_stage_timings"][stage]["median_ms"] for r in valid),
                                              "range_ms": [min(r["server_stage_timings"][stage]["median_ms"] for r in valid), max(r["server_stage_timings"][stage]["median_ms"] for r in valid)]}
                                      for stage in STAGES}
    result["peak_memory_measurement_methods"] = [r["peak_memory_measurement_method"] for r in valid]
    return result


async def run_slots(args, plan, rows, origin, attempt_fn=attempt):
    valid, paths = [], []
    for slot in range(1, args.runs + 1):
        for number in range(1, plan["invalid_runs"]["max_attempts_per_run_slot"] + 1):
            record = await attempt_fn(args, plan, rows, origin, slot, number)
            path = save_record(args.output_dir, record)
            print(f"{record['validity_status']}: {path}")
            if record.get("interrupted"):
                raise RuntimeError("Execution interrupted; invalid attempt preserved")
            if record["validity_status"] == "VALID":
                valid.append(record)
                paths.append(str(path))
                break
            # Socket denial is fatal; it must not trigger a different configuration or retries.
            if any("Docker socket access denied" in e.get("error", "") for e in record["failure_details"]):
                raise RuntimeError(json.dumps(record["failure_details"]))
        else:
            raise RuntimeError(f"Run slot {slot} exhausted three attempts: {json.dumps(record['failure_details'])}")
    result = {"mode": args.mode, "version": args.version, "concurrency": args.concurrency,
              "model_threads": args.model_threads, "valid_attempt_files": paths, "statistics": summary(valid, plan)}
    print(f"Summary: {save_record(args.output_dir, result, 'summary')}")
    return result


def main(argv=None):
    plan = load_plan()
    cli = parser(plan)
    args = cli.parse_args(argv)
    try:
        validate_config(args, plan)
        rows = load_dataset(plan)
        origin = provenance(args.mode)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if not os.access(args.output_dir, os.W_OK):
            raise ValueError("Results directory is not writable")
        asyncio.run(run_slots(args, plan, rows, origin))
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
