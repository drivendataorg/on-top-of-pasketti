#!/bin/bash
set -euo pipefail

# ── Inference ──────────────────────────────────────────────────────────
PYTORCH_ALLOC_CONF=expandable_segments:True python -u src/run_inference.py
