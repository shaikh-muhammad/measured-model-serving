"""Offline runner tests. All timings, Docker data and HTTP responses are fake."""

import asyncio
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import threading
from types import SimpleNamespace

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_load", ROOT / "bench/run_load.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Real process or network operation forbidden in offline tests")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    originals = {p: p.read_bytes() for p in (ROOT / "bench/thresholds.yaml", ROOT / "data/sentences.jsonl")}
    yield
    assert all(p.read_bytes() == data for p, data in originals.items())


@pytest.fixture
def plan():
    return bench.load_plan()


@pytest.fixture
def rows(plan):
    return bench.load_dataset(plan)


@pytest.fixture
def args(plan, tmp_path):
    # run_slots is tested with a temporary directory; CLI paths are validated separately.
    value = bench.parser(plan).parse_args(["--version", "v0", "--mode", "diagnostic", "--concurrency", "4",
                                          "--model-threads", "2", "--image", "fake:v0"])
    value.output_dir = tmp_path
    return value


def payload():
    return {"label": "POSITIVE", "score": 0.9, "truncated": False,
            "timings_ms": {"tokenize": 1.0, "model": 2.0, "postprocess": 0.5, "server_total": 4.0}}


class Clock:
    def __init__(self):
        self.value = 0

    def __call__(self):
        self.value += 1_000_000
        return self.value


class FakeClient:
    def __init__(self, failure=None, **kwargs):
        self.calls, self.active, self.maximum = [], 0, 0
        self.failure = failure

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, path, **kwargs):
        assert path == "/health"
        return httpx.Response(200, json={"status": "ok", "version": "v0"})

    async def post(self, path, *, json):
        assert path == "/classify"
        index = len(self.calls)
        self.calls.append(json)
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        try:
            await asyncio.sleep(0)
            if self.failure and index == 0:
                if isinstance(self.failure, Exception):
                    raise self.failure
                return self.failure
            return httpx.Response(200, json=payload())
        finally:
            self.active -= 1


def origin():
    return {"load_runner_git_commit": "f7b3cf4316d102ab36f8016479eebcef595920d4",
            "load_runner_file_sha256": bench.sha256(bench.SOURCE), "cpu_architecture": "fake-arch",
            "hardware": {"cpuinfo": "fake CPU", "platform": "fake OS"}}


@pytest.fixture
def fake_boundaries(monkeypatch):
    calls = []
    inspection = {"Id": "fake-container", "Image": "sha256:fake-image", "HostConfig": {"NanoCpus": 4_000_000_000, "Memory": 2147483648},
                  "NetworkSettings": {"Ports": {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "12345"}]}}}
    runtime = {"uid": 10001, "gid": 10001, "model_threads": "2",
               "packages": {k: "fake-version" for k in ("python", "fastapi", "uvicorn", "torch", "transformers")},
               "server_processes": [["python", "-m", "uvicorn", "src.app:app", "--workers", "1"]]}

    async def docker(*argv):
        calls.append(argv)
        if argv[0] == "create":
            return "fake-container"
        if argv[0] == "inspect":
            return json.dumps([inspection])
        if argv[:2] == ("image", "inspect"):
            return json.dumps([{"Size": 123456, "RepoDigests": ["fake@sha256:identity"]}])
        if argv[0] == "exec":
            return json.dumps({"peak": 1024, "current": 512} if argv[-1] == bench.MEMORY_SCRIPT else runtime)
        return ""

    client = FakeClient()
    monkeypatch.setattr(bench, "docker", docker)
    monkeypatch.setattr(bench.httpx, "AsyncClient", lambda **kwargs: client)
    monkeypatch.setattr(bench.time, "perf_counter_ns", Clock())
    monkeypatch.setattr(bench.time, "perf_counter", lambda: 0.0)
    return SimpleNamespace(calls=calls, client=client, inspection=inspection, runtime=runtime)


def test_committed_plan_dataset_and_reproducible_order(plan, rows):
    assert len(rows) == len({r["id"] for r in rows}) == 1000
    assert Counter(r["length_bucket"] for r in rows) == {"short": 400, "medium": 400, "long": 200}
    assert bench.sha256(ROOT / plan["identity"]["dataset"]["path"]) == plan["identity"]["dataset"]["sha256"]
    assert bench.seeded_order(rows, 20261010) == bench.seeded_order(rows, 20261010)
    assert bench.seeded_order(rows, 20261010) != rows
    assert {r["id"] for r in bench.seeded_order(rows, 20261010)} == {r["id"] for r in rows}


def test_dataset_rejects_hash_fields_duplicates_and_buckets(plan, rows, tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "ROOT", tmp_path)
    for mutation in ("hash", "fields", "duplicate", "buckets"):
        changed = copy.deepcopy(rows)
        if mutation == "fields":
            del changed[0]["text"]
        elif mutation == "duplicate":
            changed[0]["id"] = changed[1]["id"]
        elif mutation == "buckets":
            changed[0]["length_bucket"] = "long"
        path = tmp_path / "sentences.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in changed))
        registered = copy.deepcopy(plan)
        registered["identity"]["dataset"].update(path="sentences.jsonl", sha256="wrong" if mutation == "hash" else bench.sha256(path))
        with pytest.raises(ValueError):
            bench.load_dataset(registered)


@pytest.mark.parametrize("field,value", [("warmup", 99), ("requests", 999), ("seed", 1), ("runs", 2),
                                         ("version", "v1"), ("concurrency", 1), ("model_threads", 3), ("image", "-bad")])
def test_rejects_unregistered_cli_before_docker(plan, args, field, value):
    args.output_dir = ROOT / "results"
    setattr(args, field, value)
    with pytest.raises(ValueError):
        bench.validate_config(args, plan)


def test_official_cli_and_freeze(plan, args, monkeypatch):
    args.output_dir = ROOT / "results"
    for version in ("v0", "v1", "v2"):
        args.mode, args.version, args.concurrency, args.model_threads = "official", version, 1, 2
        bench.validate_config(args, plan)
    monkeypatch.setattr(bench, "command", lambda argv: "fake-head" if argv[1] == "rev-parse" else "old-source")
    with pytest.raises(ValueError, match="frozen and committed"):
        bench.provenance("official")
    monkeypatch.setattr(bench, "command", lambda argv: "fake-head" if argv[1] == "rev-parse" else bench.SOURCE.read_text())
    captured = bench.provenance("official")
    assert captured["load_runner_git_commit"] == "fake-head"
    assert captured["load_runner_file_sha256"] == bench.sha256(bench.SOURCE)
    assert captured["hardware"]["cpuinfo"] and captured["hardware"]["os_release"]


@pytest.mark.parametrize("concurrency", [1, 4])
def test_exact_workload_bounded_and_warmups_excluded(rows, concurrency):
    client, clock = FakeClient(), Clock()

    async def exercise():
        ordered = bench.seeded_order(rows, 20261010)
        warm, failures, _ = await bench.phase(client, ordered[:100], concurrency, clock)
        assert not failures and len(warm) == 100
        measured, failures, elapsed = await bench.phase(client, ordered, concurrency, clock)
        assert not failures and len(measured) == 1000
        assert [r["sentence_id"] for r in measured] == [r["id"] for r in ordered]
        assert client.maximum == concurrency and len(client.calls) == 1100
        assert all(r["model_ms"] == 2 and r["client_observed_request_latency_ms"] > 0 for r in measured)
        result = bench.metrics(measured, elapsed)
        assert result["throughput_requests_per_second"] == 1000 / elapsed
        for r in warm:
            r["client_observed_request_latency_ms"] = 999999
        assert bench.metrics(measured, elapsed) == result
    asyncio.run(exercise())


@pytest.mark.parametrize("change", [lambda p: p.pop("score"), lambda p: p.update(label="BAD"),
                                   lambda p: p.update(score=True), lambda p: p.update(score=1.1),
                                   lambda p: p.update(truncated="false"), lambda p: p.update(truncated=True),
                                   lambda p: p["timings_ms"].update(model=-1),
                                   lambda p: p["timings_ms"].update(model=float("nan")),
                                   lambda p: p["timings_ms"].update(model=float("inf")),
                                   lambda p: p["timings_ms"].pop("server_total")])
def test_exact_response_contract(change):
    body = payload()
    assert bench.response_record(body)["server_total_ms"] == 4
    change(body)
    with pytest.raises(ValueError):
        bench.response_record(body)


def test_nearest_rank():
    values = list(range(1000, 0, -1))
    assert [bench.nearest_rank(values, p) for p in (50, 95, 99)] == [500, 950, 990]


def test_complete_attempt_telemetry_and_cleanup(plan, rows, args, fake_boundaries):
    record = asyncio.run(bench.attempt(args, plan, rows, origin(), 1, 1))
    assert record["validity_status"] == "VALID", record["failure_details"]
    assert all(record[k] is not None for k in plan["mandatory_measurements"]["per_run"])
    assert record["warmup_request_count"] == 100 and record["measured_request_count"] == 1000
    assert record["peak_memory_measurement_method"] == "cgroup memory.peak"
    assert record["container_id"] == "fake-container" and record["docker_image_size_bytes"] == 123456
    assert record["runtime_identity"]["uid"] != 0 and record["server_workers"] == 1
    assert record["startup_time_seconds"] > 0
    command = fake_boundaries.calls[0]
    assert "--cpus=4" in command and "--memory=2g" in command and "MODEL_THREADS=2" in command
    assert "--pull=never" in command and "--user=10001:10001" in command
    assert "127.0.0.1::8000" in command and command[-2:] == ("--workers", "1")
    assert fake_boundaries.calls[-1][:2] == ("rm", "-f")
    assert fake_boundaries.calls[-1][2].startswith("measured-load-")
    for mutation in ("missing", "duplicate", "metadata"):
        bad = copy.deepcopy(record)
        if mutation == "missing":
            bad["request_records"].pop()
        elif mutation == "duplicate":
            bad["request_records"][0]["sentence_id"] = bad["request_records"][1]["sentence_id"]
        else:
            bad["hardware"] = None
        bench.validity(bad, rows, plan)
        assert bad["validity_status"] == "INVALID"


@pytest.mark.parametrize("failure", [httpx.Response(503, text="unavailable"), httpx.Response(200, text="not JSON"),
                                     httpx.Response(200, json={"label": "POSITIVE"}),
                                     httpx.ReadTimeout("fake timeout"), httpx.ConnectError("fake connection")])
def test_failed_warmup_invalid_preserved(plan, rows, args, fake_boundaries, failure):
    fake_boundaries.client.failure = failure
    record = asyncio.run(bench.attempt(args, plan, rows, origin(), 1, 1))
    assert record["validity_status"] == "INVALID" and len(record["warmup_failures"]) == 1
    assert record["warmup_failures"][0]["error_type"] and record["warmup_failures"][0]["sentence_id"]
    assert len(record["request_records"]) == 1000
    first = bench.save_record(args.output_dir, record)
    second = bench.save_record(args.output_dir, record)
    assert first != second and json.loads(first.read_text())["validity_status"] == "INVALID"
    assert json.loads(first.read_text())["warmup_failures"] == record["warmup_failures"]


def test_measured_failure_and_cleanup_after_start_failure(plan, rows, args, fake_boundaries, monkeypatch):
    original = fake_boundaries.client.post

    async def fail_measured(path, **kwargs):
        if len(fake_boundaries.client.calls) == 100:
            fake_boundaries.client.calls.append(kwargs["json"])
            return httpx.Response(500, text="measured failure")
        return await original(path, **kwargs)
    monkeypatch.setattr(fake_boundaries.client, "post", fail_measured)
    record = asyncio.run(bench.attempt(args, plan, rows, origin(), 1, 1))
    assert record["validity_status"] == "INVALID"
    assert record["measured_request_failures"][0]["response_body"] == "measured failure"
    fake_docker = bench.docker

    async def fail_start(*argv):
        if argv[0] == "start":
            raise RuntimeError("fake start failure: raw stderr")
        return await fake_docker(*argv)
    monkeypatch.setattr(bench, "docker", fail_start)
    failed = asyncio.run(bench.attempt(args, plan, rows, origin(), 1, 2))
    assert failed["validity_status"] == "INVALID"
    assert "raw stderr" in failed["failure_details"][0]["error"]
    assert fake_boundaries.calls[-1][:2] == ("rm", "-f")


@pytest.mark.parametrize("mutation", ["cpu", "memory", "root", "threads", "workers", "versions", "port"])
def test_required_constraints_invalid(plan, rows, args, fake_boundaries, mutation):
    if mutation in ("cpu", "memory"):
        fake_boundaries.inspection["HostConfig"]["NanoCpus" if mutation == "cpu" else "Memory"] = 1
    elif mutation == "port":
        fake_boundaries.inspection["NetworkSettings"]["Ports"]["8000/tcp"][0]["HostIp"] = "0.0.0.0"
    else:
        key, value = {"root": ("uid", 0), "threads": ("model_threads", "4"), "workers": ("server_processes", []),
                      "versions": ("packages", {})}[mutation]
        fake_boundaries.runtime[key] = value
    result = asyncio.run(bench.attempt(args, plan, rows, origin(), 1, 1))
    assert result["validity_status"] == "INVALID"
    assert fake_boundaries.calls[-1][:2] == ("rm", "-f")


def test_memory_methods_and_sampling_failure(monkeypatch):
    async def exercise():
        for values, expected in (({"peak": 100, "current": 40}, "cgroup memory.peak"), ({"current": 40}, "sampled peak"), ({}, "sampled peak")):
            async def docker(*argv):
                return json.dumps(values if argv[0] == "exec" else {"MemUsage": "1.5MiB / 2GiB"})
            monkeypatch.setattr(bench, "docker", docker)
            value, method = await bench.memory_read("fake")
            assert method == expected and value == (values.get("peak", values.get("current", 1572864)))
            stop = asyncio.Event()
            stop.set()
            record = {}
            monkeypatch.setattr(bench.time, "perf_counter", lambda: 0)
            await bench.monitor_memory("fake", stop, record)
            assert record["peak_memory_measurement_method"] == expected
        stop, record = asyncio.Event(), {}
        async def delayed_read(container):
            return 10, "sampled peak"
        times = iter([0, 0, 2])
        monkeypatch.setattr(bench, "memory_read", delayed_read)
        monkeypatch.setattr(bench.time, "perf_counter", lambda: next(times))
        with pytest.raises(ValueError, match="one second"):
            await bench.monitor_memory("fake", stop, record)
        assert len(record["memory_samples"]) == 1
    asyncio.run(exercise())


def test_retry_limit_stops_and_saves_every_attempt(plan, rows, args):
    calls = []
    async def invalid(*values):
        calls.append(values[-2:])
        return {"validity_status": "INVALID", "failure_details": [{"error": "fake failure"}], "run_slot": values[-2], "attempt_number": values[-1]}
    with pytest.raises(RuntimeError, match="exhausted three attempts"):
        asyncio.run(bench.run_slots(args, plan, rows, origin(), invalid))
    assert calls == [(1, 1), (1, 2), (1, 3)]
    saved = [json.loads(p.read_text()) for p in args.output_dir.glob("attempt-*.json")]
    assert len(saved) == 3 and all(r["validity_status"] == "INVALID" for r in saved)


def test_three_valid_slots_same_slot_retry_and_summary(plan, rows, args, fake_boundaries):
    template = asyncio.run(bench.attempt(args, plan, rows, origin(), 1, 1))
    calls = []
    async def mixed(*values):
        slot, number = values[-2:]
        calls.append((slot, number))
        record = copy.deepcopy(template)
        record.update(run_slot=slot, attempt_number=number, p99=slot * 10)
        if (slot, number) == (1, 1):
            record.update(validity_status="INVALID", failure_details=[{"error": "fake retry"}])
        return record
    result = asyncio.run(bench.run_slots(args, plan, rows, origin(), mixed))
    assert calls == [(1, 1), (1, 2), (2, 1), (3, 1)]
    assert result["statistics"]["p99"] == {"median": 20, "range": [10, 30]}
    assert len(result["valid_attempt_files"]) == 3
    assert len(list(args.output_dir.glob("attempt-*.json"))) == 4
    with pytest.raises(ValueError, match="three VALID"):
        bench.summary([template, template], plan)
    with pytest.raises(ValueError, match="three VALID"):
        bench.summary([template, template, {"validity_status": "INVALID"}], plan)


def test_socket_denial_actionable_fatal_and_raw_error(plan, rows, args, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="permission denied on Docker socket\n"))
    with pytest.raises(RuntimeError, match="administrator") as error:
        bench.command(["docker", "create", "fake"])
    assert "stdout=''" in str(error.value) and "permission denied" in str(error.value)
    calls = []
    async def denied(*values):
        calls.append(values[-2:])
        return {"validity_status": "INVALID", "failure_details": [{"error": str(error.value)}]}
    with pytest.raises(RuntimeError, match="Docker socket access denied"):
        asyncio.run(bench.run_slots(args, plan, rows, origin(), denied))
    assert calls == [(1, 1)] and len(list(args.output_dir.glob("attempt-*.json"))) == 1


def test_health_wrong_version_times_out_offline(monkeypatch):
    class Wrong(FakeClient):
        async def get(self, *args, **kwargs):
            return httpx.Response(200, json={"status": "ok", "version": "v2"})
    times = iter([0, 0, 121])
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(times))
    with pytest.raises(TimeoutError, match="v2"):
        asyncio.run(bench.healthy(Wrong(), "v0"))


def test_interruption_preserves_partial_records_and_cleans_up(plan, rows, args, fake_boundaries, monkeypatch):
    async def interrupted(client, workload, concurrency, **kwargs):
        kwargs["observations"].append({"sentence_id": workload[0]["id"], "success": False,
                                       "error_type": "ReadTimeout", "error": "fake partial failure"})
        raise asyncio.CancelledError()
    monkeypatch.setattr(bench, "phase", interrupted)
    record = asyncio.run(bench.attempt(args, plan, rows, origin(), 1, 1))
    assert record["validity_status"] == "INVALID" and record["interrupted"]
    assert record["warmup_request_count"] == 1 and len(record["warmup_failures"]) == 1
    assert fake_boundaries.calls[-1][:2] == ("rm", "-f")
    async def stopped(*values):
        return record
    with pytest.raises(RuntimeError, match="interrupted"):
        asyncio.run(bench.run_slots(args, plan, rows, origin(), stopped))
    assert len(list(args.output_dir.glob("attempt-*.json"))) == 1


@pytest.fixture
def pending_subprocess(monkeypatch):
    """Exercise the real to_thread boundary using an event-controlled fake CLI."""
    original_shield = asyncio.shield

    def configure(pending, outcome="success", cleanup_failure=False):
        loop = asyncio.get_running_loop()
        state = SimpleNamespace(entered=asyncio.Event(), cancelled=asyncio.Event(),
                                release=threading.Event(), calls=[], events=[],
                                containers={"unrelated-container"}, name=None)

        def observed_shield(task):
            future = original_shield(task)
            def completed(future):
                if future.cancelled():
                    state.cancelled.set()
            future.add_done_callback(completed)
            return future

        def run(argv, **options):
            assert argv[0] == "docker"
            assert options == {"cwd": bench.ROOT, "capture_output": True, "text": True, "timeout": 30}
            operation = argv[1]
            state.calls.append(tuple(argv))
            if operation == "create":
                state.name = argv[argv.index("--name") + 1]
            blocked = operation == pending and (operation != "exec" or argv[-1] == bench.MEMORY_SCRIPT)
            if blocked:
                loop.call_soon_threadsafe(state.entered.set)
                assert state.release.wait(timeout=5), "Test watchdog: pending fake command was not released"
                state.events.append(f"{operation}_finished")
                if outcome == "timeout":
                    raise subprocess.TimeoutExpired(argv, 30, output="fake timeout stdout", stderr="fake timeout stderr")
                if outcome == "failure":
                    return SimpleNamespace(returncode=1, stdout="fake create stdout", stderr="fake delayed create failure")
            output, error, status = "", "", 0
            if operation == "create":
                state.containers.add(state.name)
                output = "fake-container\n"
            elif operation == "rm":
                state.events.append("remove")
                assert argv[2:] == ["-f", state.name]
                if cleanup_failure:
                    error, status = "fake removal failure", 1
                elif state.name not in state.containers:
                    error, status = "No such container", 1
                else:
                    state.containers.remove(state.name)
            elif operation == "inspect":
                output = json.dumps([{"Id": "fake-container", "Image": "sha256:fake-image",
                                      "HostConfig": {"NanoCpus": 4_000_000_000, "Memory": 2147483648},
                                      "NetworkSettings": {"Ports": {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "12345"}]}}}])
            elif operation == "image":
                output = json.dumps([{"Size": 123456}])
            elif operation == "exec":
                runtime = {"uid": 10001, "gid": 10001, "model_threads": "2",
                           "packages": {k: "fake-version" for k in ("python", "fastapi", "uvicorn", "torch", "transformers")},
                           "server_processes": [["python", "-m", "uvicorn", "src.app:app", "--workers", "1"]]}
                output = json.dumps({"peak": 1024} if argv[-1] == bench.MEMORY_SCRIPT else runtime)
            return SimpleNamespace(returncode=status, stdout=output, stderr=error)

        monkeypatch.setattr(subprocess, "run", run)
        monkeypatch.setattr(bench.asyncio, "shield", observed_shield)
        monkeypatch.setattr(bench.httpx, "AsyncClient", lambda **kwargs: FakeClient())
        monkeypatch.setattr(bench.time, "perf_counter_ns", Clock())
        monkeypatch.setattr(bench.time, "perf_counter", lambda: 0.0)
        return state
    return configure


@pytest.mark.parametrize("outcome", ["success", "failure", "timeout"])
def test_cancelled_create_finishes_before_removal(plan, rows, args, pending_subprocess, outcome):
    async def exercise():
        state = pending_subprocess("create", outcome)
        task = asyncio.create_task(bench.attempt(args, plan, rows, origin(), 1, 1))
        try:
            await asyncio.wait_for(state.entered.wait(), timeout=5)
            for _ in range(2):
                state.cancelled.clear()
                task.cancel()
                await asyncio.wait_for(state.cancelled.wait(), timeout=5)
                assert not task.done()
                assert not any(call[1] == "rm" for call in state.calls)
            state.release.set()
            record = await asyncio.wait_for(task, timeout=5)
            assert state.events == ["create_finished", "remove"]
            assert state.containers == {"unrelated-container"}
            assert record["validity_status"] == "INVALID" and record["interrupted"]
            assert any(e["error_type"] == "CancelledError" for e in record["failure_details"] if "error_type" in e)
            if outcome != "success":
                details = json.dumps(record["failure_details"])
                expected = "fake delayed create failure" if outcome == "failure" else "fake timeout stderr"
                assert expected in details and "stdout" in details
            saved = json.loads(bench.save_record(args.output_dir, record).read_text())
            assert saved["failure_details"] == record["failure_details"]
            assert all(call[-1] == state.name for call in state.calls if call[1] == "rm")
        finally:
            state.release.set()
            if not task.done():
                task.cancel()
                await asyncio.wait_for(task, timeout=5)
    asyncio.run(exercise())


def test_cancelled_start_finishes_before_removal(plan, rows, args, pending_subprocess):
    async def exercise():
        state = pending_subprocess("start")
        task = asyncio.create_task(bench.attempt(args, plan, rows, origin(), 1, 1))
        try:
            await asyncio.wait_for(state.entered.wait(), timeout=5)
            task.cancel()
            await asyncio.wait_for(state.cancelled.wait(), timeout=5)
            assert not task.done() and not any(call[1] == "rm" for call in state.calls)
            state.release.set()
            record = await asyncio.wait_for(task, timeout=5)
            assert state.events == ["start_finished", "remove"]
            assert state.containers == {"unrelated-container"}
            assert record["interrupted"] and record["container_id"] == "fake-container"
        finally:
            state.release.set()
            if not task.done():
                task.cancel()
                await asyncio.wait_for(task, timeout=5)
    asyncio.run(exercise())


@pytest.mark.parametrize("pending", ["exec", "rm"])
@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_cleanup_survives_cancellation_and_exposes_failures(plan, rows, args, pending_subprocess, monkeypatch,
                                                          pending, cleanup_failure):
    async def exercise():
        state = pending_subprocess(pending, cleanup_failure=cleanup_failure)
        phase_entered = asyncio.Event()
        async def interrupted(client, workload, concurrency, **kwargs):
            kwargs["observations"].append({"sentence_id": workload[0]["id"], "success": False,
                                           "error_type": "ReadTimeout", "error": "fake partial failure"})
            phase_entered.set()
            raise asyncio.CancelledError()
        monkeypatch.setattr(bench, "phase", interrupted)
        task = asyncio.create_task(bench.attempt(args, plan, rows, origin(), 1, 1))
        try:
            await asyncio.wait_for(state.entered.wait(), timeout=5)
            await asyncio.wait_for(phase_entered.wait(), timeout=5)
            for _ in range(2):
                state.cancelled.clear()
                task.cancel()
                await asyncio.wait_for(state.cancelled.wait(), timeout=5)
                assert not task.done()
            state.release.set()
            record = await asyncio.wait_for(task, timeout=5)
            assert state.events == [f"{pending}_finished", "remove"]
            assert record["validity_status"] == "INVALID" and record["interrupted"]
            assert any(e["error"] == "Execution interrupted during cleanup" for e in record["failure_details"])
            assert all(call[-1] == state.name for call in state.calls if call[1] == "rm")
            assert "unrelated-container" in state.containers
            if cleanup_failure:
                assert state.name in state.containers
                assert any(e.get("phase") == "cleanup" and "fake removal failure" in e["error"] for e in record["failure_details"])
            else:
                assert state.containers == {"unrelated-container"}
            assert record["warmup_request_count"] == 1
            assert record["warmup_failures"][0]["error"] == "fake partial failure"
            saved = json.loads(bench.save_record(args.output_dir, record).read_text())
            assert saved["failure_details"] == record["failure_details"]
        finally:
            state.release.set()
            if not task.done():
                task.cancel()
                await asyncio.wait_for(task, timeout=5)
    asyncio.run(exercise())


def test_deferred_cancellation_is_propagated_and_internal_cancellation_is_bounded():
    async def exercise():
        owned = asyncio.create_task(asyncio.Event().wait())
        owned.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(bench.await_completion(owned), timeout=5)
        entered, release = asyncio.Event(), asyncio.Event()
        async def work():
            entered.set()
            await release.wait()
            raise RuntimeError("fake delayed error")
        owned = asyncio.create_task(work())
        caller = asyncio.create_task(bench.await_completion(owned))
        await entered.wait()
        caller.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError) as error:
            await asyncio.wait_for(caller, timeout=5)
        assert isinstance(error.value.__cause__, RuntimeError)
        assert str(error.value.__cause__) == "fake delayed error"
        assert owned.done() and not owned.cancelled()
    asyncio.run(exercise())
