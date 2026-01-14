#!/bin/bash
# Simple demo runner - no prompts, no monitoring

cd "$(dirname "${BASH_SOURCE[0]}")"
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python demo.py
