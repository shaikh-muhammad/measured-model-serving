# Project 2: Measured Model Serving

- Work only in `/home/sm/Projects/measured-model-serving`. Never touch another project or repository. The external pytest temporary directory specified below is the sole filesystem exception.
- Never commit, push, merge, create branches, or perform GitHub operations without explicit user authorization. Read-only Git commands are allowed.
- Never create, request, store, or commit API keys, access tokens, passwords, or other secrets. Do not use paid services. Ordinary existing developer-tool authentication is allowed; never expose credentials.
- Make no network calls except installing approved packages and downloading the pinned public model once, when explicitly requested. Never use a Hugging Face token.
- This project is CPU-only. Install PyTorch from `https://download.pytorch.org/whl/cpu` first. Never install NVIDIA, CUDA, or Triton packages.
- Before running any Python tools, load `scripts/env.sh` so `TMPDIR` and `PIP_CACHE_DIR` use the project's main-disk directories, never `/tmp`.
- Run pytest with `--basetemp=/home/sm/Projects/.pytest-tmp-measured-model-serving`. This directory is outside the Git repository, on the main disk, and is the only permitted filesystem exception.
- Keep changes small and restricted to the exact files named in each task prompt. If anything is unclear, stop and ask.
- Report every exact command executed and its complete raw stdout and stderr, including empty streams; do not provide only summaries. Include `git status --short --branch` in reports.
- Never invent benchmark numbers or test results. Preserve experimental fairness and reproducibility.
- The Project 2 brief defines the scope: v0 PyTorch, v1 ONNX Runtime, and v2 ONNX Runtime with 8-bit dynamic quantization. Do not add features or experiments without approval.
