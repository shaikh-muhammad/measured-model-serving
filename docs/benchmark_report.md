# Official benchmark results

Saved-data comparison of v0 PyTorch, v1 ONNX Runtime, and corrected v2 MatMul-only signed INT8 dynamic quantization.
No new measurements or inference are performed. Diagnostic and feasibility runs are excluded.

Regenerate from the project root: `source scripts/env.sh && PYTHONDONTWRITEBYTECODE=1 python bench/report.py`
or `make report`. Verify without writing: `python bench/report.py --check` (after sourcing the environment).

## Environment and frozen configuration

ARM64 Ubuntu VM: Ubuntu 26.04.1 LTS; 7.0.0-38-generic; 4 CPUs, 2147483648 bytes (2 GiB) container limit, 1 worker, 4 model threads, selected by the v0-only diagnostic and frozen across versions.
Dataset: `data/sentences.jsonl`, seed 20261010, SHA-256 `bfe687d4cff16e7d06e8580d09b5947760095b2829fceb0ac9b5fc7f70a99be0`.
Model revision: `714eb0fa89d2f80546fda750413ed43d93601a13`. Frozen runner SHA-256: `f61e09b2f0af90d6e64bcda44844dc87024197f354e01a5e0f8ee21f93d6ac33`.
Thresholds SHA-256: `3f31248914dcd93754995f23002a15bcc8e7329259b6787ff638a4d7bc795439`.

- v0 PyTorch: image `sha256:d310c95bd14a88704c2b5721f78165910e3e907907b21be10b635b0830a3ab4c`; runner Git commit `cca39eb2924ed5bed8b80aeba78309fa44a2ea34`; fastapi 0.143.0, numpy 2.5.4, python 3.14.4, tokenizers 0.23.3, torch 2.14.1+cpu, transformers 5.19.0, uvicorn 0.54.0.
- v1 ONNX: image `sha256:bc7f3c9727910661fd29a69a7c72ae68c0cfb724af1c5c7d824c17d99c66c914`; runner Git commit `0a40c4eb7e7577ee7c8b06ff520864cdd9d8f9fe`; fastapi 0.143.0, numpy 2.5.4, onnxruntime 1.31.0, python 3.14.4, tokenizers 0.23.3, uvicorn 0.54.0.
- v2 signed INT8: image `sha256:2356df36161fd393d39608d742d28290fa2a644335603f16aad268067e76e2d7`; runner Git commit `0a40c4eb7e7577ee7c8b06ff520864cdd9d8f9fe`; fastapi 0.143.0, numpy 2.5.4, onnxruntime 1.31.0, python 3.14.4, tokenizers 0.23.3, uvicorn 0.54.0.

## Client latency and throughput

Every cell is the three-run median (minimum–maximum) of run-level statistics. Latency is milliseconds; throughput is requests/s.

### Concurrency 4 — primary comparison

| Version | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (requests/s) |
| --- | --- | --- | --- | --- |
| v0 PyTorch | 367.973152 (363.729137–369.104750) | 859.620583 (836.103477–865.230095) | 987.331188 (983.507651–1028.586478) | 9.467402 (9.462252–9.566476) |
| v1 ONNX | 402.820539 (395.367355–416.879244) | 985.267498 (967.385086–1000.916849) | 1233.355484 (1213.410183–1271.798754) | 8.510558 (8.396170–8.701532) |
| v2 signed INT8 | 278.551001 (261.315565–282.404815) | 522.090772 (515.484890–546.227027) | 625.352030 (612.040066–668.270409) | 13.519013 (13.394471–14.155556) |

### Concurrency 1

| Version | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (requests/s) |
| --- | --- | --- | --- | --- |
| v0 PyTorch | 263.127529 (238.543110–264.980276) | 501.834485 (467.567909–521.658491) | 568.889771 (553.931729–606.837001) | 3.568621 (3.481548–3.860035) |
| v1 ONNX | 161.981393 (152.645517–166.256699) | 449.277257 (440.213071–478.532524) | 569.923653 (556.269799–613.368213) | 5.022235 (4.780366–5.255713) |
| v2 signed INT8 | 93.116056 (89.032169–97.797401) | 233.837834 (227.630708–240.739466) | 310.740351 (301.079745–316.840813) | 9.011440 (8.841494–9.428666) |

[Official p99 chart](p99_by_version.svg)

## Resources

Peak container memory uses **cgroup memory.peak**; MiB = bytes / 1,048,576. Docker image size is its actual built-image size in bytes. Startup is seconds from issuing Docker start until the first correct healthy response.

| Version | Concurrency | Peak memory (MiB) | Startup (s) | Image size (bytes) |
| --- | --- | --- | --- | --- |
| v0 PyTorch | 1 | 389.585938 (387.546875–393.062500) | 8.129933 (7.967704–8.145302) | 2106058478 (2106058478–2106058478) |
| v0 PyTorch | 4 | 503.183594 (458.476562–584.679688) | 10.981760 (8.624873–11.309167) | 2106058478 (2106058478–2106058478) |
| v1 ONNX | 1 | 472.011719 (471.382812–796.574219) | 3.858929 (3.679682–5.385727) | 983250492 (983250492–983250492) |
| v1 ONNX | 4 | 523.621094 (522.675781–528.937500) | 3.991549 (3.937453–4.435775) | 983250492 (983250492–983250492) |
| v2 signed INT8 | 1 | 286.566406 (279.343750–288.328125) | 2.686598 (2.490055–2.748775) | 720020107 (720020107–720020107) |
| v2 signed INT8 | 4 | 308.078125 (306.226562–309.082031) | 3.246321 (2.643636–3.353620) | 720020107 (720020107–720020107) |

## Server-stage timings

Median across three run-level medians (minimum–maximum of those medians), in milliseconds. server_total spans handler entry through inference, before response serialization; stages need not sum to it.

| Version | Concurrency | tokenize | model | postprocess | server_total |
| --- | --- | --- | --- | --- | --- |
| v0 PyTorch | 1 | 0.555839 (0.527144–0.559789) | 248.297047 (228.259600–249.665183) | 0.070822 (0.065375–0.071522) | 248.984893 (229.089209–250.551122) |
| v0 PyTorch | 4 | 0.613414 (0.606011–0.620613) | 345.727573 (341.333963–345.766278) | 0.065226 (0.065000–0.066377) | 347.867837 (341.999702–348.527117) |
| v1 ONNX | 1 | 39.320499 (37.223011–42.715678) | 110.572792 (106.337961–115.942341) | 0.071838 (0.068859–0.072177) | 152.280659 (143.349858–157.627493) |
| v1 ONNX | 4 | 89.083338 (85.473747–94.452124) | 235.274095 (226.572306–239.389784) | 0.071071 (0.069694–0.071296) | 343.196992 (334.968316–352.470804) |
| v2 signed INT8 | 1 | 37.830668 (37.612896–39.316501) | 44.921972 (43.039369–46.190486) | 0.069153 (0.069152–0.069423) | 83.060486 (80.940677–88.633931) |
| v2 signed INT8 | 4 | 82.341829 (79.228876–83.188111) | 103.048187 (102.120992–105.484013) | 0.068588 (0.067302–0.068634) | 201.814199 (191.316963–203.088664) |

Both ONNX implementations reconstruct the tokenizer for every request in `StandaloneEncoder.encode()` ([scripts/verify_tokenizer.py:124](../scripts/verify_tokenizer.py#L124)), called by [src/onnx_backend.py:65](../src/onnx_backend.py#L65); v2 uses the same encoder and inherited inference path ([src/onnx_backend_v2.py:13](../src/onnx_backend_v2.py#L13), [line 31](../src/onnx_backend_v2.py#L31)). v0 initializes its tokenizer once ([src/backend.py:34](../src/backend.py#L34)). At concurrency 4, v1 had a faster model stage than v0 (235.3 vs 345.7 ms), while the ONNX tokenize-stage medians (89.1 ms for v1 and 82.3 ms for v2, versus 0.61 ms for v0) are consistent with this per-request overhead. Its contribution to the end-to-end p99 regression was not isolated by a separate measurement.

## Label agreement

All six original v0 HTTP runs agree for every dataset ID. Each version is compared against these original labels across both concurrency settings.

| Version | Unique IDs | Matching HTTP labels | Agreement | Mismatching IDs | Mismatch records by bucket |
| --- | --- | --- | --- | --- | --- |
| v0 | 1000 | 6000/6000 | 100.000000% | none | long: 0, medium: 0, short: 0 |
| v1 | 1000 | 6000/6000 | 100.000000% | none | long: 0, medium: 0, short: 0 |
| v2 | 1000 | 6000/6000 | 100.000000% | none | long: 0, medium: 0, short: 0 |

## Registered decisions

Using concurrency 4, B = baseline median p99, C = candidate median p99, I = (B − C) / B, S = (baseline maximum − baseline minimum) / B. Performance requires I ≥ 20% and I > S; quality requires ≥ 99% label agreement. Performance failure or quality failure takes REJECT precedence. Meeting the minimum improvement without exceeding S is INCONCLUSIVE. KEEP also requires mandatory validity, correctness and parity checks.

Baseline spread S = (1028.586478 − 983.507651) / 987.331188 = 4.565725%.

| Candidate | Improvement calculation (ms) | I | Minimum improvement | I > S | Quality | Performance | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | (987.331188 − 1233.355484) / 987.331188 | -24.918112% | FAIL | FAIL | PASS | FAIL | **REJECT** |
| v2 | (987.331188 − 625.352030) / 987.331188 | 36.662385% | PASS | PASS | PASS | PASS | **KEEP** |

Mandatory saved HTTP validity, summary consistency, tokenizer/export parity, corrected signed quantization, and container integration evidence passed. The archived failed quantization experiment is not the official v2 input.

What did not help: v1 became slower under the primary concurrency-4 workload despite preserving predictions. Its smaller image does not establish faster inference. Concurrency 1 is disclosed fully but does not decide acceptance.

## Variation disclosures

Relative spread = (maximum − minimum) / median. The registered disclosure threshold is strictly above 10%. Valid runs are retained. The following run metrics and stage medians exceed it:

- v0, concurrency 1, p50: 10.047282%.
- v0, concurrency 1, p95: 10.778570%.
- v0, concurrency 1, throughput_requests_per_second: 10.606000%.
- v0, concurrency 4, peak_container_memory: 25.080930%.
- v0, concurrency 4, startup_time_seconds: 24.443199%.
- v1, concurrency 1, p99: 10.018608%.
- v1, concurrency 1, peak_container_memory: 68.894774%.
- v1, concurrency 1, startup_time_seconds: 44.210329%.
- v1, concurrency 1, tokenize stage median: 13.968967%.
- v1, concurrency 4, startup_time_seconds: 12.484424%.
- v1, concurrency 4, tokenize stage median: 10.078627%.
- v2, concurrency 4, startup_time_seconds: 21.870405%.

## Methodology and limitations

One ARM64 VM, a fixed generated text dataset, and three repetitions per setting. The baseline was measured earlier; virtual machine scheduling noise and changing host conditions limit comparisons. Results may differ on x86 or another system. Label agreement is not human-labeled accuracy.

Client latency covers request issuance through receipt of the complete response. The registered nearest-rank method uses sorted observations 500, 950 and 990 for p50, p95 and p99 of each 1,000-request run. The 100 separate warm-ups are excluded. Throughput is successful measured requests / measured elapsed seconds. Server stages use saved per-request timings. Memory, startup, image size and client p99 describe different measurements.

## Original evidence

All 18 VALID attempts and six summaries are checked against committed bytes; summaries are checked against original observations. Each table row aggregates the three matching version/concurrency run slots from these records:

- [results/v0-baseline/attempt-1040d6c3275b4f74a94278599fd78d24.json](../results/v0-baseline/attempt-1040d6c3275b4f74a94278599fd78d24.json)
- [results/v0-baseline/attempt-692ebe3ba524439e9027fddf80d949ff.json](../results/v0-baseline/attempt-692ebe3ba524439e9027fddf80d949ff.json)
- [results/v0-baseline/attempt-cf0ea92acea54b6b99cf1176d3225e6e.json](../results/v0-baseline/attempt-cf0ea92acea54b6b99cf1176d3225e6e.json)
- [results/v0-baseline/attempt-d46685a9b179494cb78f02aed2219675.json](../results/v0-baseline/attempt-d46685a9b179494cb78f02aed2219675.json)
- [results/v0-baseline/attempt-f6612b3b9db344ad8dd45222aa69606c.json](../results/v0-baseline/attempt-f6612b3b9db344ad8dd45222aa69606c.json)
- [results/v0-baseline/attempt-f78e100ceefc45f8ace980ce57808610.json](../results/v0-baseline/attempt-f78e100ceefc45f8ace980ce57808610.json)
- [results/v0-baseline/summary-6f538529b57545fc90efe02db0ad37a2.json](../results/v0-baseline/summary-6f538529b57545fc90efe02db0ad37a2.json)
- [results/v0-baseline/summary-a05f97da5df344699b971f7ca96088f9.json](../results/v0-baseline/summary-a05f97da5df344699b971f7ca96088f9.json)
- [results/v1-official/attempt-57f9c386c6724d6fafa7f92dcac29082.json](../results/v1-official/attempt-57f9c386c6724d6fafa7f92dcac29082.json)
- [results/v1-official/attempt-5e9e84aaf4664cd581795c09d4731206.json](../results/v1-official/attempt-5e9e84aaf4664cd581795c09d4731206.json)
- [results/v1-official/attempt-b02462327cab4e869a10d664fb598177.json](../results/v1-official/attempt-b02462327cab4e869a10d664fb598177.json)
- [results/v1-official/attempt-c04413f245f047f5a3a8808ae4abea5f.json](../results/v1-official/attempt-c04413f245f047f5a3a8808ae4abea5f.json)
- [results/v1-official/attempt-e87b51c5d5304dd0a6cfd12e33f76671.json](../results/v1-official/attempt-e87b51c5d5304dd0a6cfd12e33f76671.json)
- [results/v1-official/attempt-ec55d0b8a88c4da9b49b779b40bf6da3.json](../results/v1-official/attempt-ec55d0b8a88c4da9b49b779b40bf6da3.json)
- [results/v1-official/summary-2c94ca783707487b85afc345c16dbbe9.json](../results/v1-official/summary-2c94ca783707487b85afc345c16dbbe9.json)
- [results/v1-official/summary-ba51d0c2eac04d39a0e6cd238e45110d.json](../results/v1-official/summary-ba51d0c2eac04d39a0e6cd238e45110d.json)
- [results/v2-official/attempt-1c27a767a2c94168be97a447e949d3cd.json](../results/v2-official/attempt-1c27a767a2c94168be97a447e949d3cd.json)
- [results/v2-official/attempt-96b154b38fe1423d9eb2d1c0d76c7bfe.json](../results/v2-official/attempt-96b154b38fe1423d9eb2d1c0d76c7bfe.json)
- [results/v2-official/attempt-a76714bbf76f4965aaea9c03241c8737.json](../results/v2-official/attempt-a76714bbf76f4965aaea9c03241c8737.json)
- [results/v2-official/attempt-a84612ed4fda43c29a4ebde58c2a5551.json](../results/v2-official/attempt-a84612ed4fda43c29a4ebde58c2a5551.json)
- [results/v2-official/attempt-c05c694963984cd4b0793e16ae84d0e4.json](../results/v2-official/attempt-c05c694963984cd4b0793e16ae84d0e4.json)
- [results/v2-official/attempt-ff98a85ca1a941dd822df0e5ef83f1a6.json](../results/v2-official/attempt-ff98a85ca1a941dd822df0e5ef83f1a6.json)
- [results/v2-official/summary-18cd5dd068124ae38962211a4bce2b0f.json](../results/v2-official/summary-18cd5dd068124ae38962211a4bce2b0f.json)
- [results/v2-official/summary-2c7cdd05a3584f1ea926c28f7f6faaba.json](../results/v2-official/summary-2c7cdd05a3584f1ea926c28f7f6faaba.json)

Existing mandatory correctness evidence:

- [results/parity/v1-tokenizer.json](../results/parity/v1-tokenizer.json)
- [results/parity/v1-export.json](../results/parity/v1-export.json)
- [results/parity/v2-quantization.json](../results/parity/v2-quantization.json)
- [results/checks/v1-container.json](../results/checks/v1-container.json)
- [results/checks/v2-container.json](../results/checks/v2-container.json)
