#!/bin/bash
# LABS Solver Benchmark Runner
# Run this script on the Brev GPU instance

set -e

echo "=========================================="
echo "LABS Solver - Complete Benchmark Suite"
echo "=========================================="
echo ""

# Navigate to source directory
cd "$(dirname "$0")/src"

# Check environment
echo "Checking environment..."
python3 -c "import numpy; print(f'  NumPy: {numpy.__version__}')" 2>/dev/null || echo "  NumPy: Not installed"
python3 -c "import cupy; print(f'  CuPy: {cupy.__version__}')" 2>/dev/null || echo "  CuPy: Not installed"
python3 -c "import cudaq; print(f'  CUDA-Q: Available')" 2>/dev/null || echo "  CUDA-Q: Not installed"
echo ""

# Parse arguments
QUICK=""
MAX_N="25"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --quick) QUICK="--quick"; shift ;;
        --max-n) MAX_N="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# Run benchmarks
echo "Running complete benchmark suite..."
echo "  Quick mode: ${QUICK:-No}"
echo "  Max N: $MAX_N"
echo ""

python3 run_complete_benchmark.py $QUICK --max-n $MAX_N

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $(dirname $0)/results/"
echo ""
echo "Generated files:"
ls -la ../results/

echo ""
echo "To view the report:"
echo "  cat ../results/BENCHMARK_REPORT.md"
