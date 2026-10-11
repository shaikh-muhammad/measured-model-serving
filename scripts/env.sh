#!/usr/bin/env bash

# Find the project directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Keep temporary files and pip cache on the main disk
export TMPDIR="$PROJECT_ROOT/tmp"
export PIP_CACHE_DIR="$PROJECT_ROOT/.pip-cache"

mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

# Activate the project's Python virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"
