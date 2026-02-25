#!/bin/bash
# Lambda Labs GPU Training Setup
#
# Usage: SSH into your Lambda instance, then:
#   git clone https://github.com/jonathan-politzki/shared-matryoshka.git
#   cd shared-matryoshka
#   bash scripts/lambda_setup.sh
#
# Runs all 7 experimental conditions end-to-end.

set -e

echo "=========================================="
echo "  Shared Matryoshka — Lambda GPU Training"
echo "=========================================="

# 1. Install dependencies
echo ""
echo "[1/4] Installing dependencies..."
pip install -e ".[dev]"

# 2. Check GPU
echo ""
echo "[2/4] Checking GPU..."
python -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'  GPU: {gpu} ({mem:.0f} GB)')
else:
    print('  WARNING: No GPU detected!')
    exit(1)
"

# 3. Run tests to verify setup
echo ""
echo "[3/4] Running tests..."
python -m pytest tests/ -v --tb=short

# 4. Run all experiments
echo ""
echo "[4/4] Running all 7 conditions (data → train → eval → table)..."
echo ""
python scripts/run_all.py

echo ""
echo "=========================================="
echo "  DONE!"
echo "=========================================="
echo ""
echo "Results are in results/"
echo "  - all_results.json       (combined metrics)"
echo "  - results_table.tex      (LaTeX table for paper)"
echo "  - *_metrics.json         (per-condition metrics)"
echo "  - *_history.json         (training loss curves)"
echo ""
echo "Checkpoints are in checkpoints/"
echo ""
echo "To copy results back to your Mac:"
echo "  scp -r ubuntu@<LAMBDA_IP>:~/shared-matryoshka/results ./results"
echo "  scp -r ubuntu@<LAMBDA_IP>:~/shared-matryoshka/checkpoints ./checkpoints"
