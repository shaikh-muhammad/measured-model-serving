"""Reproduction safety tests; every Docker/runner invocation is mocked."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import yaml

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reproduction", PROJECT / "scripts/reproduce_official.py")
repro = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repro)
IMAGES = {v: "sha256:" + str(i) * 64 for i, v in enumerate(repro.VERSIONS, 1)}


@pytest.fixture(autouse=True)
def no_live_processes(monkeypatch):
    original = subprocess.Popen

    def guarded(argv, *args, **kwargs):
        assert argv[:2] == ["git", "show"], "Live Docker/runner processes forbidden"
        return original(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", guarded)


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(repro, "ROOT", tmp_path)
    return tmp_path


def test_original_artifact_metadata_is_model_free():
    for v in repro.VERSIONS:
        manifest = repro.artifact_manifest(v)
        assert len(manifest) == (4 if v == "v0" else 2)
        assert all(n.startswith("model-cache/") and len(sha) == 64 for n, sha in manifest.items())
        assert repro.expected_image(v).startswith("sha256:")
    assert repro.artifact_manifest("v2")["model-cache/v2/model.onnx"] != repro.artifact_manifest("v1")["model-cache/v1/model.onnx"]


def test_frozen_matrix_and_exact_cli(root):
    matrix = repro.commands(root / "tmp/reproductions/new", IMAGES)
    assert [(v, c) for v, c, _ in matrix] == [(v, c) for v in ("v0", "v1", "v2") for c in (1, 4)]
    for v, c, argv in matrix:
        options = dict(zip(argv[2::2], argv[3::2]))
        assert options == {"--mode": "official", "--version": v, "--concurrency": str(c),
                           "--model-threads": "4", "--image": IMAGES[v], "--warmup": "100",
                           "--requests": "1000", "--seed": "20261010", "--runs": "3",
                           "--output-dir": str(root / "tmp/reproductions/new" / v / f"c{c}")}
        assert argv[1] == str(root / "bench/run_load.py")


@pytest.mark.parametrize("name", ["results/v0-baseline", "results/v1-official/new", "results/v2-official",
                                  "results/new", "tmp/reproductions", "README.md", "../outside"])
def test_original_and_outside_destinations_refused(root, name):
    with pytest.raises(ValueError, match="NEW directory"):
        repro.destination(name)


@pytest.mark.parametrize("with_result", [False, True])
def test_existing_destination_refused_even_when_empty(root, with_result):
    path = root / "tmp/reproductions/existing"
    path.mkdir(parents=True)
    if with_result:
        (path / "attempt-original.json").write_text('{"validity_status":"INVALID"}')
    with pytest.raises(ValueError, match="existing output"):
        repro.destination(path)


def test_symlinked_parent_refused(root):
    base = root / "tmp/reproductions"
    base.mkdir(parents=True)
    (base / "real").mkdir()
    (base / "link").symlink_to(base / "real", target_is_directory=True)
    with pytest.raises(ValueError, match="Symlinked"):
        repro.destination(base / "link/new")


def test_missing_and_changed_artifacts_fail_without_loading(root, monkeypatch):
    path = root / "model-cache/v1/model.onnx"
    monkeypatch.setattr(repro, "artifact_manifest", lambda version: {str(path.relative_to(root)): hashlib.sha256(b"verified").hexdigest()})
    with pytest.raises(ValueError, match="Missing regular local artifact"):
        repro.verify_artifacts("v1")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"substitute")
    with pytest.raises(ValueError, match="refusing another model"):
        repro.verify_artifacts("v1")
    path.write_bytes(b"verified")
    repro.verify_artifacts("v1")


def image_record(v):
    return {"Id": IMAGES[v], "Architecture": "arm64", "Os": "linux", "Config": {
        "User": "10001:10001", "Env": ["MODEL_THREADS=4"],
        "Cmd": ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]}}


@pytest.mark.parametrize("version", ["v0", "v1", "v2"])
def test_exact_image_selection_without_live_docker(root, monkeypatch, version):
    calls = []
    monkeypatch.setattr(repro, "verify_artifacts", lambda v: None)
    monkeypatch.setattr(repro, "expected_image", IMAGES.__getitem__)

    def inspect(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=json.dumps([image_record(version)]))

    monkeypatch.setattr(subprocess, "run", inspect)
    assert repro.verify_image(version) == IMAGES[version]
    assert calls == [["docker", "image", "inspect", f"measured-model-serving:{version}"]]


@pytest.mark.parametrize("problem", ["missing", "substitute", "architecture", "user", "threads", "workers"])
def test_image_missing_or_conflicting_is_refused(root, monkeypatch, problem):
    monkeypatch.setattr(repro, "verify_artifacts", lambda v: None)
    monkeypatch.setattr(repro, "expected_image", IMAGES.__getitem__)
    image = image_record("v1")
    if problem == "substitute": image["Id"] = IMAGES["v2"]
    elif problem == "architecture": image["Architecture"] = "amd64"
    elif problem == "user": image["Config"]["User"] = "root"
    elif problem == "threads": image["Config"]["Env"] = ["MODEL_THREADS=2"]
    elif problem == "workers": image["Config"]["Cmd"][-1] = "2"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=1 if problem == "missing" else 0, stdout=json.dumps([image])))
    with pytest.raises(ValueError):
        repro.verify_image("v1")


@pytest.fixture
def prepared(root, monkeypatch):
    (root / "bench").mkdir()
    (root / "bench/run_load.py").write_bytes(b"frozen runner")
    monkeypatch.setattr(repro, "digest", lambda p: repro.FROZEN[p.relative_to(root).as_posix()])
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: b"frozen runner")
    monkeypatch.setattr(repro.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(repro.platform, "system", lambda: "Linux")
    monkeypatch.setattr(repro, "verify_image", IMAGES.__getitem__)
    return root


def save_fake_setting(argv):
    options = dict(zip(argv[2::2], argv[3::2]))
    v, c = options["--version"], int(options["--concurrency"])
    directory = Path(options["--output-dir"])
    directory.mkdir(parents=True)
    paths = []
    for slot in (1, 2, 3):
        a = dict(mode="official", version=v, concurrency=c, actual_model_threads=4, run_slot=slot,
                 validity_status="VALID", error_count=0, failure_details=[], warmup_request_count=100,
                 measured_request_count=1000, seed=20261010, image_identity=IMAGES[v],
                 load_runner_file_sha256=repro.FROZEN["bench/run_load.py"], dataset_sha256=repro.FROZEN["data/sentences.jsonl"],
                 cpu_limit=4, memory_limit=2147483648, server_workers=1)
        path = directory / f"attempt-{slot}.json"
        path.write_text(json.dumps(a))
        paths.append(str(path))
    (directory / "attempt-invalid.json").write_text('{"validity_status":"INVALID","failure_details":["fake failure"]}')
    (directory / "summary-fake.json").write_text(json.dumps(dict(mode="official", version=v,
        concurrency=c, model_threads=4, valid_attempt_files=paths)))


def test_sequential_complete_protocol_preserves_invalid_evidence(prepared, monkeypatch, capsys):
    calls = []

    def fake_run(argv, **kwargs):
        assert kwargs["check"] is True
        assert argv[1].endswith("bench/run_load.py")
        calls.append(argv)
        save_fake_setting(argv)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert repro.main(["--output", "tmp/reproductions/new"]) == 0
    assert calls == [argv for _, _, argv in repro.commands(prepared / "tmp/reproductions/new", IMAGES)]
    assert "COMPLETE protocol rerun" in capsys.readouterr().out
    assert len(list((prepared / "tmp/reproductions/new").rglob("attempt-invalid.json"))) == 6


@pytest.mark.parametrize("failure", ["exit", "no_evidence", "partial"])
def test_partial_or_empty_run_cannot_claim_completion(prepared, monkeypatch, capsys, failure):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if failure == "exit": raise subprocess.CalledProcessError(1, argv)
        if failure == "partial":
            save_fake_setting(argv)
            path = Path(argv[-1]) / "attempt-3.json"
            path.unlink()

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert repro.main(["--output", "tmp/reproductions/new"]) == 1
    streams = capsys.readouterr()
    assert "COMPLETE protocol rerun" not in streams.out
    assert "INCOMPLETE: 0/6" in streams.err and len(calls) == 1
    assert (prepared / "tmp/reproductions/new").is_dir()


def test_prerequisite_failure_creates_no_output(prepared, monkeypatch):
    def missing(version):
        raise ValueError("Required local image unavailable")
    monkeypatch.setattr(repro, "verify_image", missing)
    assert repro.main(["--output", "tmp/reproductions/new"]) == 1
    assert not (prepared / "tmp/reproductions/new").exists()


def test_ci_and_make_use_the_same_model_free_tests():
    names = ("tests/test_data.py", "tests/test_report.py", "tests/test_bench.py", "tests/test_reproduction.py")
    make = (PROJECT / "Makefile").read_text()
    workflow = (PROJECT / ".github/workflows/ci.yml").read_text()
    config = yaml.safe_load(workflow)
    assert "pull_request" in config["on"] and config["permissions"] == {"contents": "read"}
    assert config["jobs"]["offline"]["env"]["HF_HUB_OFFLINE"] == "1"
    assert "OFFLINE_TESTS := " + " ".join(names) in make
    assert "python -m pytest " + " ".join(names) in workflow
    assert "fetch-depth: 0" in workflow and "persist-credentials: false" in workflow
    assert "contents: read" in workflow and "pull_request:" in workflow
