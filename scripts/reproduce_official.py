"""Repeat the frozen protocol from verified local artifacts/images, never rebuild them.

Results must use a nonexistent directory beneath tmp/reproductions/. Partial and
INVALID evidence is retained. The frozen runner owns all measurements and retries.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ("v0", "v1", "v2")
DIRECTORIES = {"v0": "v0-baseline", "v1": "v1-official", "v2": "v2-official"}
FROZEN = {
    "bench/run_load.py": "f61e09b2f0af90d6e64bcda44844dc87024197f354e01a5e0f8ee21f93d6ac33",
    "bench/thresholds.yaml": "3f31248914dcd93754995f23002a15bcc8e7329259b6787ff638a4d7bc795439",
    "data/sentences.jsonl": "bfe687d4cff16e7d06e8580d09b5947760095b2829fceb0ac9b5fc7f70a99be0",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evidence(name):
    """Read original committed correctness/identity metadata, without model imports."""
    path = ROOT / name
    require(path.read_bytes() == subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT),
            f"Original evidence changed: {name}")
    return json.loads(path.read_text())


def artifact_manifest(version):
    require(version in VERSIONS, "Version must be v0, v1 or v2")
    export = evidence("results/parity/v1-export.json")
    tokenizer = evidence("results/parity/v1-tokenizer.json")
    quantized = evidence("results/parity/v2-quantization.json")
    require(export["status"] == tokenizer["outcome"] == quantized["correctness"] == "PASS",
            "Original artifact verification did not pass")
    manifests = {
        "v0": {f"model-cache/v0/{name}": sha for name, sha in export["source_sha256"].items()},
        "v1": {"model-cache/v1/model.onnx": export["onnx_sha256"],
               "model-cache/v1/tokenizer.json": tokenizer["tokenizer_sha256"]},
        "v2": {"model-cache/v2/model.onnx": quantized["v2_sha256"],
               "model-cache/v1/tokenizer.json": tokenizer["tokenizer_sha256"]},
    }
    return manifests[version]


def verify_artifacts(version):
    for name, expected in artifact_manifest(version).items():
        path = ROOT / name
        require(path.is_file() and not path.is_symlink(),
                f"Missing regular local artifact: {name}. Prepare the verified local cache; "
                "make setup prepares only v0. Frozen ONNX preparation needs a separately approved workflow.")
        require(digest(path) == expected, f"Artifact identity differs: {name}; refusing another model")


def expected_image(version):
    paths = sorted((ROOT / "results" / DIRECTORIES[version]).glob("attempt-*.json"))
    require(len(paths) == 6, f"Expected six original attempts for {version}")
    images = set()
    for path in paths:
        a = evidence(path.relative_to(ROOT).as_posix())
        require(a["version"] == version and a["validity_status"] == "VALID", "Original image provenance invalid")
        images.add(a["image_identity"])
    require(len(images) == 1, f"Conflicting original {version} images")
    return images.pop()


def verify_image(version):
    verify_artifacts(version)
    expected = expected_image(version)
    tag = f"measured-model-serving:{version}"
    result = subprocess.run(["docker", "image", "inspect", tag], cwd=ROOT, capture_output=True, text=True)
    require(result.returncode == 0, f"Required local image unavailable: {tag}. Build/restore the verified image first; no pull attempted.")
    images = json.loads(result.stdout)
    require(len(images) == 1, f"Ambiguous local image: {tag}")
    image = images[0]
    require(image["Id"] == expected, f"Image identity differs for {tag}; refusing a substitute or rebuild with a different ID")
    require(image["Architecture"] == "arm64" and image["Os"] == "linux", "Expected linux/arm64 image")
    config = image["Config"]
    require(config["User"] == "10001:10001" and "MODEL_THREADS=4" in config["Env"], "Image user or threads differ")
    require(config["Cmd"] == ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"],
            "Image server command differs")
    return expected


def destination(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    resolved = path.resolve()
    base = ROOT / "tmp/reproductions"
    require(resolved.is_relative_to(base) and resolved != base,
            "Output must be a NEW directory beneath tmp/reproductions/, outside all original results")
    require(not path.exists() and not path.is_symlink(), f"Refusing existing output destination: {path}")
    require(all(not p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(ROOT)),
            "Symlinked output path is forbidden")
    return resolved


def commands(output, images):
    return [(version, concurrency, [
        sys.executable, str(ROOT / "bench/run_load.py"), "--mode", "official", "--version", version,
        "--concurrency", str(concurrency), "--model-threads", "4", "--image", images[version],
        "--warmup", "100", "--requests", "1000", "--seed", "20261010", "--runs", "3",
        "--output-dir", str(output / version / f"c{concurrency}"),
    ]) for version in VERSIONS for concurrency in (1, 4)]


def verify_setting(directory, version, concurrency, image):
    """Confirm the runner actually saved a complete group; preserve INVALID attempts."""
    summaries = list(directory.glob("summary-*.json"))
    require(len(summaries) == 1, f"Incomplete {version}/c{concurrency}: expected one saved summary")
    summary = json.loads(summaries[0].read_text())
    require((summary["mode"], summary["version"], summary["concurrency"], summary["model_threads"]) ==
            ("official", version, concurrency, 4), "Rerun summary configuration differs")
    paths = [Path(name).resolve() for name in summary["valid_attempt_files"]]
    require(len(paths) == len(set(paths)) == 3 and all(p.parent == directory for p in paths),
            "Rerun summary requires three distinct local attempts")
    slots = set()
    for path in paths:
        a = json.loads(path.read_text())
        expected = dict(mode="official", version=version, concurrency=concurrency, actual_model_threads=4,
                        validity_status="VALID", error_count=0, failure_details=[], warmup_request_count=100,
                        measured_request_count=1000, seed=20261010, image_identity=image,
                        load_runner_file_sha256=FROZEN["bench/run_load.py"], dataset_sha256=FROZEN["data/sentences.jsonl"],
                        cpu_limit=4, memory_limit=2147483648, server_workers=1)
        require(all(a.get(k) == v for k, v in expected.items()), f"Incomplete/conflicting rerun attempt: {path.name}")
        slots.add(a["run_slot"])
    require(slots == {1, 2, 3}, "Incomplete rerun slots")


def reproduce(value):
    output = destination(value)
    for name, expected in FROZEN.items():
        require(digest(ROOT / name) == expected, f"Frozen input changed: {name}")
    require(subprocess.check_output(["git", "show", "HEAD:bench/run_load.py"], cwd=ROOT) ==
            (ROOT / "bench/run_load.py").read_bytes(), "Runner must be frozen and committed")
    require(platform.machine() == "aarch64" and platform.system() == "Linux", "Benchmark rerun requires the ARM64 Linux environment")
    images = {version: verify_image(version) for version in VERSIONS}
    # Reserve exclusively only after every local prerequisite passes. Never resume
    # or delete an earlier directory, including one containing partial evidence.
    output.mkdir(parents=True, exist_ok=False)
    completed = 0
    try:
        for version, concurrency, argv in commands(output, images):
            print(f"Starting {version}/c{concurrency}; output {argv[-1]}", flush=True)
            subprocess.run(argv, cwd=ROOT, check=True)  # inherit all runner stdout/stderr
            verify_setting(Path(argv[-1]), version, concurrency, images[version])
            completed += 1
    except BaseException:
        print(f"INCOMPLETE: {completed}/6 settings verified. All partial/INVALID evidence retained at {output}", file=sys.stderr)
        raise
    print(f"COMPLETE protocol rerun: six settings, 18 VALID runs verified; evidence at {output}. "
          "No new acceptance decision or comparison report has been calculated.")


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    mode = cli.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", help="NEW directory beneath tmp/reproductions/; runs the full protocol")
    mode.add_argument("--check-artifacts", choices=VERSIONS, help="read-only local artifact check, no Docker")
    mode.add_argument("--check-image", choices=VERSIONS, help="read-only artifact/image check; print exact verified image ID")
    args = cli.parse_args(argv)
    try:
        if args.check_artifacts:
            verify_artifacts(args.check_artifacts)
        elif args.check_image:
            print(verify_image(args.check_image))
        else:
            reproduce(args.output)
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"Reproduction refused or incomplete: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
