SHELL := /bin/bash
VERSION ?= v0
IMAGE := measured-model-serving:$(VERSION)
MODEL_REVISION := 714eb0fa89d2f80546fda750413ed43d93601a13
BENCH_OUTPUT ?= tmp/reproductions/official
OFFLINE_TESTS := tests/test_data.py tests/test_report.py tests/test_bench.py tests/test_reproduction.py
TEST_ENV := PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider' HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
PYTEST_TEMP := /home/sm/Projects/.pytest-tmp-measured-model-serving

.PHONY: setup data build run bench report test test-local

setup:
	source scripts/env.sh && PYTHONDONTWRITEBYTECODE=1 python scripts/prepare_model.py

data:
	source scripts/env.sh && PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 python data/make_sentences.py --revision $(MODEL_REVISION)

build:
	source scripts/env.sh && PYTHONDONTWRITEBYTECODE=1 python scripts/reproduce_official.py --check-artifacts "$(VERSION)"
ifeq ($(VERSION),v0)
	docker build --platform linux/arm64 --tag $(IMAGE) .
else ifeq ($(VERSION),v1)
	set -o pipefail; tar -cf - Dockerfile.v1 requirements-v1.txt src/onnx_backend.py src/app_v1.py scripts/verify_tokenizer.py | docker buildx build --platform linux/arm64 --pull=false --build-context model=./model-cache/v1 --load --progress plain -f Dockerfile.v1 -t $(IMAGE) -
else ifeq ($(VERSION),v2)
	set -o pipefail; tar -cf - Dockerfile.v2 requirements-v1.txt src/app_v2.py src/app_v1.py src/onnx_backend.py src/onnx_backend_v2.py scripts/verify_tokenizer.py | docker buildx build --platform linux/arm64 --pull=false --build-context v2model=./model-cache/v2 --build-context tokenizer=./model-cache/v1 --load --progress plain -f Dockerfile.v2 -t $(IMAGE) -
endif

run:
	source scripts/env.sh && image_id=$$(PYTHONDONTWRITEBYTECODE=1 python scripts/reproduce_official.py --check-image "$(VERSION)") && docker run --rm --pull=never --cpus=4 --memory=2g --user=10001:10001 -e MODEL_THREADS=4 -p 127.0.0.1:8000:8000 "$$image_id"

test:
	source scripts/env.sh && $(TEST_ENV) python -m pytest $(OFFLINE_TESTS) -v --basetemp=$(PYTEST_TEMP)

test-local: test
	source scripts/env.sh && for version in v0 v1 v2; do PYTHONDONTWRITEBYTECODE=1 python scripts/reproduce_official.py --check-artifacts "$$version" || exit; done
	source scripts/env.sh && $(TEST_ENV) python -m pytest tests/test_service.py tests/test_real_service.py tests/test_onnx_parity.py tests/test_tokenizer_parity.py tests/test_v1_service.py tests/test_v2_quantization.py tests/test_v2_service.py -v --basetemp=$(PYTEST_TEMP)

bench:
	source scripts/env.sh && PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TRANSFORMERS_OFFLINE=1 python scripts/reproduce_official.py --output "$(BENCH_OUTPUT)"

report:
	source scripts/env.sh && PYTHONDONTWRITEBYTECODE=1 python bench/report.py
