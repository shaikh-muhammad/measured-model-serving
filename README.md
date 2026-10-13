# Measured Model Serving

Measuring and optimizing CPU-only sentiment classification without changing predicted labels unnecessarily.

Model: `distilbert-base-uncased-finetuned-sst-2-english`. Verified model revision, licence, and Docker base-image digest are recorded below.

Planned versions:

- **v0:** PyTorch + Transformers.
- **v1:** ONNX Runtime.
- **v2:** ONNX Runtime + signed 8-bit dynamic quantization.

The target container has 4 CPUs and 2 GB RAM, runs as a non-root user, and performs offline model inference. The planned API is `POST /classify` and `GET /health`.

Inputs are limited to 2,000 characters and 256 model tokens, including special tokens. `truncated` is true only when the original input exceeds 256 tokens, special tokens included.

The planned benchmark uses 1,000 fixed synthetic sentences per measured run, separate warm-ups, concurrency 1 and 4, and three valid runs per setting. The primary acceptance gate is a concurrency-4 median p99 latency improvement of at least 20%, greater than the baseline run-to-run p99 spread, with at least 99% label agreement with v0. Regressions and inconclusive results must be disclosed.

Before freezing the chosen thread configuration for all versions, a v0-only diagnostic will test 4, 2, and 1 threads at concurrency 4. The thresholds file will be created, reviewed, and explicitly authorized for commit during Day 1, before any performance test, including this diagnostic.

Python 3.14.4 is preferred; Python 3.12 is a fallback only for unavailable prebuilt wheels. ONNX `tokenizer.json` will be generated using Transformers and checked for exact tokenization equivalence before lightweight runtime use.

`requirements-dev.lock` captures the current development environment. Separate pinned runtime dependency sets for the final images will be prepared and verified during their Docker build steps. The v0 setup, data, build, run, and test commands are available; bench and report remain failing placeholders.

Current status: **SETUP COMPLETE; V0 BUILD AND SMOKE CHECKS COMPLETE; NO BENCHMARK RESULTS YET.**

## Pinned local v0 baseline

The public source is [distilbert/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english). The immutable revision is `714eb0fa89d2f80546fda750413ed43d93601a13`. Its [official revision metadata](https://huggingface.co/api/models/distilbert/distilbert-base-uncased-finetuned-sst-2-english/revision/714eb0fa89d2f80546fda750413ed43d93601a13) declares **Apache-2.0**. The application model ID remains `distilbert-base-uncased-finetuned-sst-2-english`.

The image uses the [official Python image](https://hub.docker.com/_/python), tag `python:3.14.4-slim-bookworm`, pinned to index digest `sha256:fc74d22ffd0d5ac395a4b7bdda75a4539758862c49ebf3005647084631e63789`. Its verified ARM64 manifest is `sha256:8cc758368ddbbc2ac75f93c734e34e3b1bbb3751f0b56f5a1e1e0acc0c55d46f`. No Python 3.12 fallback was needed.

From the project root, with the development `.venv` already prepared, run:

```bash
make setup       # Anonymous preparation of the four pinned original model files.
make data        # Local-only generation; checks identical existing dataset bytes.
make build       # Build linux/arm64 with binary-only CPU dependencies.
make run VERSION=v0  # One server process, 4 CPUs, 2 GB RAM, non-root.
make test        # All local correctness tests; requires the prepared model.
```

For a fresh clone, prepare Python 3.14.4 and `.venv` using the pinned development inventory, installing CPU PyTorch first from `https://download.pytorch.org/whl/cpu` with `--only-binary=:all:`. Source `scripts/env.sh` before Python tools. `make setup` needs anonymous network access; the application only loads `MODEL_DIR` locally. It never downloads models at runtime. If this shell has not inherited existing Docker group membership, use `sg docker -c 'make build'` or `sg docker -c 'make run VERSION=v0'`.

`model-cache/` is ignored by Git. The ready directory contains only `config.json`, `model.safetensors`, `tokenizer_config.json`, and `vocab.txt`, as regular files. No `tokenizer.json` was generated. Preparation verifies pinned original-file identities and SHA-256 equality after copying:

| Prepared file | SHA-256 |
| --- | --- |
| config.json | `582122c8f414793d131e10022ce9ba04e3811a9da6389137ee2f18665b4f4d15` |
| model.safetensors | `7c3919835e442510166d267fe7cbe847e0c51cd26d9ba07b89a57b952b49b8aa` |
| tokenizer_config.json | `5ab9097b4149371c5fd52b2d6e26cb6f9c07c0d19fcfdda895b1adad6b57c3e0` |
| vocab.txt | `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3` |

The validated dataset uses seed `20261010`, with SHA-256 `bfe687d4cff16e7d06e8580d09b5947760095b2829fceb0ac9b5fc7f70a99be0`. The real original tokenizer confirms 400 short (observed 13–18 tokens), 400 medium (40–80), and 200 long (123–197) inputs, including special tokens. All satisfy the agreed bucket ranges, character limit, and token limit; none requires truncation. Regeneration matches the saved file byte-for-byte.

`MODEL_THREADS=4` is provisional until the registered Day 2 diagnostic. `make run` uses `--cpus=4 --memory=2g` and one Uvicorn worker. `server_total` measures handler entry through inference, before response serialization. Response timings are correctness smoke output, not benchmark results. `bench` and `report` remain failing placeholders.

Model-free tests: `tests/test_data.py` runs in pull-request CI with only `requirements-ci.lock`; it needs no model or PyTorch and runs offline after dependency installation. Local tests: `tests/test_service.py` uses fake components but imports PyTorch; `tests/test_real_service.py` requires the cached real model. Real 256/257-token service boundaries are verified; lightweight-tokenizer equivalence remains scheduled for Day 3.

The ARM64 image build and online/offline runtime smoke checks passed. Image packages match the pinned runtime inventory, CPU-only PyTorch is confirmed, and `pip check` passes. Local checks passed 9 data tests, 18 fake-component service tests, 11 real-model tests, and all 38 tests together. Test containers were removed; the built image remains available. No performance testing has occurred.

## Pre-registered measurement rules

[bench/thresholds.yaml](bench/thresholds.yaml) records the rules before testing so measured results cannot change the acceptance thresholds. **The threshold file must be committed with explicit user permission before ANY performance run, including the v0 thread diagnostic.** No diagnostic or benchmark has occurred, so no performance results are present yet.

p99 describes the slowest 1% tail of requests, where delays can affect users even when typical requests are fast. We use nearest-rank percentiles: sort N latencies and take the 1-based position `ceil((percentile/100) * N)`. For 1,000 requests, p99 is the 990th latency. Each version runs at concurrency 1 and 4 with three valid runs per setting; report the median of each run-level statistic and its minimum–maximum range. Disclose same-version variation over 10%.

Concurrency 4 is the primary gate because it tests the agreed four simultaneous requests within the fixed four-CPU container. Let B and C be the medians of the three valid v0 and candidate p99 values. Improvement is `I=(B-C)/B`; baseline noise spread is `S=(max(v0 p99)-min(v0 p99))/B`. Performance passes only if `I >= 0.20` AND `I > S`: at least 20% lower p99, exceeding the baseline's observed variation. If improvement reaches 20% but does not strictly exceed S, report INCONCLUSIVE. Below 20% is FAIL.

Quality requires at least 990 of the fixed 1,000 sentence IDs to match v0 labels: 99% label agreement, not ground-truth accuracy. KEEP requires passing both gates and all mandatory validity, correctness, and parity checks. A performance failure or agreement below 99% means REJECT, even alongside an inconclusive condition. Inconclusive performance with passing quality remains INCONCLUSIVE. Unresolved mandatory failures prevent KEEP and must be disclosed. Compare v1 and v2 separately against v0; v2 versus v1 is informational only. Report concurrency 1 fully, including regressions, although it does not decide acceptance.

Every official and diagnostic measured run sends each fixed dataset ID exactly once, in the same seeded order, using the unchanged dataset hash and model revision. Each run has 100 separate warm-ups, excluded from statistics and label comparison, then all 1,000 measured requests; no measured input is intentionally truncated. Require exactly one successful saved record per ID. Save ID, client latency, label, and all four server timings. Save the required run metadata, including software, hardware, resource limits, actual threads, image identity, load-runner Git commit and file hash. Use the same VM, comparable conditions, one non-root server worker, 4 CPUs, and `--memory=2g`. Freeze the runner before official v0 benchmarking and preserve it across versions; never tune ONNX separately.

Any request failure, including a warm-up failure, or missing/inconsistent required data makes the attempt INVALID. Preserve its errors and partial observations; exclude it from medians. Repeat that same run slot at most twice after the original attempt (three total attempts). If it still fails, stop all further benchmarking and show the raw failure output.

On Day 2, after the runner is implemented and tested, v0 alone tests 4, 2, and 1 model threads at concurrency 4. Each candidate has three valid repetitions, each with 100 discarded warm-ups and the complete 1,000-sentence measured set in identical order. Score each candidate by the median of its three run-level median request latencies. With M4 as the four-thread score and Mbest as the lowest score, use `G=(M4-Mbest)/M4`; four-thread noise is `N=(max(four-thread run medians)-min(four-thread run medians))/M4`. Choose the winner only if `G >= 0.05` AND `G > N`; otherwise retain four threads. Exact ties between eligible non-four-thread candidates favor the higher thread count. These are diagnostics, not official baseline p99 benchmarks. Freeze the chosen count for v0, v1, and v2, record its v0-only origin, and disclose later ONNX threading problems without retuning.

Record memory, startup, image size, run timing, and throughput during each run. Prefer the container's own cgroup `memory.peak` counter when available and permitted; otherwise sample container memory statistics at least once per second and label the value **sampled peak** in results and README. Startup runs from issuing the Docker start command until `/health` first reports that version healthy. Image size is Docker's actual built-image size field. Client latency covers issuing the request through receipt of the complete response; throughput uses successful measured requests divided by measured-phase elapsed time. Compute agreement only from saved sentence IDs and labels. The report generator must only read saved results and never perform fresh inference or take new measurements. Disclose rejected, inconclusive, and regressed results.

## Thread selection

The v0-only thread diagnostic completed with nine VALID repetitions, zero INVALID attempts, and zero retries. These are diagnostic results, not the official baseline benchmark. Medians below were calculated from each repetition's 1,000 saved client request latencies; warm-ups were excluded.

| Model threads | Run 1 median (ms) | Run 2 median (ms) | Run 3 median (ms) | Median of run medians (ms) |
| --- | --- | --- | --- | --- |
| 4 | 327.2677815 | 318.186574 | 317.9213455 | 318.186574 |
| 2 | 402.641602 | 408.4925755 | 368.8119615 | 402.641602 |
| 1 | 367.099298 | 367.92113 | 365.1154285 | 367.099298 |

M4 = 318.186574 ms, M2 = 402.641602 ms, and M1 = 367.099298 ms. Mbest = M4. The registered calculations give G = 0 (0%) and N = 0.02937407409276792 (2.937407409276792%). The gain condition G >= 0.05 failed, and the noise guard G > N failed. The registered rule therefore selects **MODEL_THREADS=4**, frozen identically for **v0, v1, and v2**. This selection used v0 only; the later runtimes must use the same count without retuning.

Every repetition passed verification of 100 successful separate warm-ups, all 1,000 distinct measured IDs in the identical seeded order, concurrency 4, actual 4-CPU and 2-GiB Docker limits, non-root UID 10001, one worker, complete response timings, and the pinned model/dataset/image identities. All nine memory measurements used the container's own **cgroup memory.peak**, and all experiment containers were removed.

Runner source SHA-256: `f61e09b2f0af90d6e64bcda44844dc87024197f354e01a5e0f8ee21f93d6ac33`. Recorded runner HEAD: `f7b3cf4316d102ab36f8016479eebcef595920d4`; the runner was uncommitted during this diagnostic.

Preserved attempt evidence, in run-slot order:

- 4 threads, run 1: `results/thread-diagnostic/attempt-baf51cfa5f814e25958fdaac116a3f80.json`
- 4 threads, run 2: `results/thread-diagnostic/attempt-ae1810110c1042f089fecf26a44a2e27.json`
- 4 threads, run 3: `results/thread-diagnostic/attempt-e71de5075c7440b0b66af23c64515415.json`
- 2 threads, run 1: `results/thread-diagnostic/attempt-2f85a170dca847ed8c8f7b3fc1aafef6.json`
- 2 threads, run 2: `results/thread-diagnostic/attempt-9b768af911444d198161132d1ce4d899.json`
- 2 threads, run 3: `results/thread-diagnostic/attempt-b28cca66a4d2413dabd083567c8c6ce3.json`
- 1 thread, run 1: `results/thread-diagnostic/attempt-9bad629c8cd24c82b57661ab0238ab90.json`
- 1 thread, run 2: `results/thread-diagnostic/attempt-c88fdd08f2ff4d53afbce7a59a72b727.json`
- 1 thread, run 3: `results/thread-diagnostic/attempt-5b37713fe4d84121ba917b234e899c76.json`

Preserved setting summaries:

- `results/thread-diagnostic/summary-57cf556e176c456bb9ed645fd7d57545.json`
- `results/thread-diagnostic/summary-e9ddf435b28d4c9a9b2f81014a72b558.json`
- `results/thread-diagnostic/summary-f3ec26f64be24816bb545130ffcb7985.json`
