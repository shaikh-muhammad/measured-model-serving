SHELL := /bin/bash
VERSION ?= v0
IMAGE := measured-model-serving:v0
MODEL_REVISION := 714eb0fa89d2f80546fda750413ed43d93601a13

.PHONY: setup data build run bench report test

setup:
	source scripts/env.sh && PYTHONDONTWRITEBYTECODE=1 python scripts/prepare_model.py

data:
	source scripts/env.sh && PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 python data/make_sentences.py --revision $(MODEL_REVISION)

build:
	docker build --platform linux/arm64 --tag $(IMAGE) .

run:
	@test "$(VERSION)" = "v0" || { echo "run VERSION=$(VERSION): not implemented" >&2; exit 1; }
	docker run --rm --cpus=4 --memory=2g -p 127.0.0.1:8000:8000 $(IMAGE)

test:
	source scripts/env.sh && PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider' HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TRANSFORMERS_OFFLINE=1 python -m pytest tests/test_data.py tests/test_service.py tests/test_real_service.py -v --basetemp=/home/sm/Projects/.pytest-tmp-measured-model-serving

bench report:
	@echo "$@: not implemented" >&2
	@exit 1
