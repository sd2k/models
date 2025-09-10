#!/bin/bash

# Script to run Rust tests with Python integration using uv-managed environment

echo "Setting up Python environment for tests..."

# Create and activate virtual environment with uv
if [ ! -d ".venv" ]; then
    uv venv
fi

# Install dependencies with uv
echo "Installing Python dependencies..."
uv pip install numpy torch transformers --quiet

# Export the Python path so PyO3 can find it
export PYO3_PYTHON="$(pwd)/.venv/bin/python"
export PYTHONPATH="$(pwd)/.venv/lib/python3.13/site-packages:$PYTHONPATH"

# Verify numpy is available
echo "Verifying Python environment..."
.venv/bin/python -c "import numpy; import torch; print(f'NumPy version: {numpy.__version__}'); print(f'PyTorch version: {torch.__version__}')"

echo ""
echo "Running Rust tests with Python integration..."
cargo test --lib -- --nocapture

echo ""
echo "Test run complete!"