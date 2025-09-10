#!/bin/bash

# Script to download T5 weights and test the implementation

echo "Installing Python dependencies..."
pip install -q transformers safetensors torch

echo ""
echo "Downloading and converting T5-small model..."
python convert_weights.py \
    --model t5-small \
    --output ./weights/t5-small \
    --conditional-generation

echo ""
echo "Model conversion complete!"
echo "Files created in ./weights/t5-small/"
ls -lh ./weights/t5-small/

echo ""
echo "Running Rust tests..."
cargo test --lib --release

echo ""
echo "Running inference example..."
cargo run --example inference --release

echo ""
echo "Setup complete! You can now use the T5 model in Rust."