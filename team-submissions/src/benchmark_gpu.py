#!/usr/bin/env python3
"""
Phase 2 GPU Benchmark Script

This script benchmarks the GPU-accelerated LABS solver against CPU baseline.
Run this on Brev with GPU access.

Steps:
1. Validates environment (CUDA-Q with nvidia backend, CuPy)
2. Runs quantum sampling benchmarks
3. Runs MTS CPU vs GPU benchmarks
4. Generates performance plots
5. Tests scaling to larger N

Usage:
    python benchmark_gpu.py [--quick] [--max-n N]

Options:
    --quick     Run faster benchmarks with fewer iterations
    --max-n N   Maximum sequence length to test (default: 25)
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

# Check for GPU libraries
print("=" * 70)
print("Environment Check")
print("=" * 70)

try:
    import cupy as cp
    print(f"CuPy: Available (version {cp.__version__})")
    print(f"  GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"  Memory: {cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] / 1e9:.1f} GB")
    CUPY_AVAILABLE = True
except ImportError as e:
    print(f"CuPy: NOT AVAILABLE ({e})")
    CUPY_AVAILABLE = False

try:
    import cudaq
    print(f"CUDA-Q: Available")
    # Try to set nvidia target
    try:
        cudaq.set_target("nvidia")
        print(f"  Target: nvidia (GPU)")
        CUDAQ_GPU = True
    except Exception as e:
        print(f"  Target: CPU only ({e})")
        CUDAQ_GPU = False
    CUDAQ_AVAILABLE = True
except ImportError as e:
    print(f"CUDA-Q: NOT AVAILABLE ({e})")
    CUDAQ_AVAILABLE = False
    CUDAQ_GPU = False

print()

# Import our modules
from labs_solver.energy import calculate_energy, merit_factor
from labs_solver.mts import memetic_tabu_search as cpu_mts, random_sequence

if CUPY_AVAILABLE:
    from labs_solver.energy_gpu import (
        calculate_energy_gpu,
        calculate_energy_batch_gpu,
        calculate_energy_batch_cpu,
        GPUEnergyCache,
    )
    from labs_solver.mts_gpu import (
        memetic_tabu_search_gpu,
        benchmark_cpu_vs_gpu,
    )

if CUDAQ_AVAILABLE:
    from labs_solver.quantum import (
        sample_quantum_population,
        quantum_enhanced_mts,
    )


# Known optimal energies for validation
KNOWN_OPTIMA = {
    3: 1, 4: 2, 5: 2, 6: 7, 7: 3, 8: 8, 9: 12, 10: 13,
    11: 12, 13: 13, 15: 28, 17: 32, 19: 52, 21: 42, 23: 58, 25: 76,
}


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(label: str, value, unit: str = ""):
    """Print formatted result."""
    if isinstance(value, float):
        print(f"  {label}: {value:.3f} {unit}")
    else:
        print(f"  {label}: {value} {unit}")


def benchmark_energy_calculation(n_values: list, batch_size: int = 100):
    """Benchmark energy calculation CPU vs GPU."""
    print_header("Benchmark 1: Energy Calculation")

    if not CUPY_AVAILABLE:
        print("  SKIPPED: CuPy not available")
        return {}

    results = []

    for n in n_values:
        print(f"\n  N={n}, batch={batch_size}")

        # Generate random sequences
        sequences = np.array([
            [1 if np.random.random() > 0.5 else -1 for _ in range(n)]
            for _ in range(batch_size)
        ], dtype=np.float32)

        # CPU batch
        cpu_start = time.time()
        cpu_energies = calculate_energy_batch_cpu(sequences)
        cpu_time = time.time() - cpu_start

        # GPU batch
        gpu_start = time.time()
        gpu_energies = calculate_energy_batch_gpu(sequences)
        gpu_time = time.time() - gpu_start

        # Verify correctness
        max_diff = np.max(np.abs(cpu_energies - gpu_energies))
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0

        print(f"    CPU: {cpu_time*1000:.2f}ms, GPU: {gpu_time*1000:.2f}ms")
        print(f"    Speedup: {speedup:.2f}x, Max diff: {max_diff:.6f}")

        results.append({
            'n': n,
            'batch_size': batch_size,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': speedup,
            'max_diff': max_diff,
        })

    return results


def benchmark_mts_scaling(n_values: list, quick: bool = False):
    """Benchmark MTS scaling with problem size."""
    print_header("Benchmark 2: MTS Scaling")

    if not CUPY_AVAILABLE:
        print("  SKIPPED: CuPy not available")
        return {}

    pop_size = 15 if quick else 20
    generations = 20 if quick else 40
    results = []

    for n in n_values:
        print(f"\n  N={n}:")
        result = benchmark_cpu_vs_gpu(n, population_size=pop_size, generations=generations)
        results.append(result)

        # Check solution quality
        optimal = KNOWN_OPTIMA.get(n, None)
        if optimal:
            ratio = optimal / min(result['cpu_energy'], result['gpu_cache_energy'])
            print(f"  Approx ratio: {ratio:.3f} (optimal={optimal})")

    return results


def benchmark_quantum_sampling(n_values: list, quick: bool = False):
    """Benchmark quantum circuit sampling."""
    print_header("Benchmark 3: Quantum Sampling")

    if not CUDAQ_AVAILABLE:
        print("  SKIPPED: CUDA-Q not available")
        return {}

    results = []
    shots = 100 if quick else 500
    pop_size = 10 if quick else 20

    for n in n_values:
        if n > 15 and quick:
            print(f"\n  N={n}: SKIPPED (quick mode)")
            continue

        print(f"\n  N={n}:")

        start = time.time()
        population = sample_quantum_population(
            n, population_size=pop_size, n_steps=1, T=1.0, shots=shots
        )
        sample_time = time.time() - start

        energies = [calculate_energy(seq) for seq in population]

        print(f"    Time: {sample_time:.3f}s")
        print(f"    Energies: min={min(energies):.0f}, mean={np.mean(energies):.1f}, max={max(energies):.0f}")

        optimal = KNOWN_OPTIMA.get(n, min(energies))
        approx_ratio = optimal / min(energies) if min(energies) > 0 else 1.0

        results.append({
            'n': n,
            'sample_time': sample_time,
            'min_energy': min(energies),
            'mean_energy': np.mean(energies),
            'approx_ratio': approx_ratio,
            'target': 'nvidia' if CUDAQ_GPU else 'cpu',
        })

    return results


def benchmark_quantum_enhanced_mts(n_values: list, quick: bool = False):
    """Benchmark quantum-enhanced MTS vs classical."""
    print_header("Benchmark 4: Quantum-Enhanced vs Classical MTS")

    if not CUDAQ_AVAILABLE:
        print("  SKIPPED: CUDA-Q not available")
        return {}

    results = []
    pop_size = 10 if quick else 15
    generations = 15 if quick else 30

    for n in n_values:
        if n > 12 and quick:
            print(f"\n  N={n}: SKIPPED (quick mode)")
            continue

        print(f"\n  N={n}:")
        optimal = KNOWN_OPTIMA.get(n, None)

        # Classical MTS (multiple trials)
        classical_results = []
        classical_start = time.time()
        for _ in range(3):
            _, energy, _, _ = cpu_mts(n, population_size=pop_size, max_generations=generations)
            classical_results.append(energy)
        classical_time = time.time() - classical_start

        # Quantum-enhanced MTS (multiple trials)
        quantum_results = []
        quantum_start = time.time()
        for _ in range(3):
            _, energy, _, _ = quantum_enhanced_mts(
                n, population_size=pop_size, max_generations=generations, n_trotter_steps=1
            )
            quantum_results.append(energy)
        quantum_time = time.time() - quantum_start

        print(f"    Classical: {classical_time:.2f}s, min={min(classical_results):.0f}, mean={np.mean(classical_results):.1f}")
        print(f"    Quantum:   {quantum_time:.2f}s, min={min(quantum_results):.0f}, mean={np.mean(quantum_results):.1f}")
        if optimal:
            print(f"    Optimal: {optimal}")

        results.append({
            'n': n,
            'classical_time': classical_time,
            'classical_min': min(classical_results),
            'classical_mean': np.mean(classical_results),
            'quantum_time': quantum_time,
            'quantum_min': min(quantum_results),
            'quantum_mean': np.mean(quantum_results),
            'optimal': optimal,
        })

    return results


def benchmark_large_n(max_n: int = 25, quick: bool = False):
    """Benchmark larger problem sizes."""
    print_header(f"Benchmark 5: Large N Scaling (up to N={max_n})")

    if not CUPY_AVAILABLE:
        print("  SKIPPED: CuPy not available")
        return {}

    results = []
    pop_size = 10 if quick else 15
    generations = 15 if quick else 25

    n_values = list(range(15, max_n + 1, 2 if quick else 1))

    for n in n_values:
        print(f"\n  N={n}:")
        start = time.time()

        _, energy, _, _, metrics = memetic_tabu_search_gpu(
            n,
            population_size=pop_size,
            max_generations=generations,
            use_shared_cache=True,
            verbose=False,
        )
        total_time = time.time() - start

        mf = (n * n) / (2 * energy) if energy > 0 else float('inf')
        optimal = KNOWN_OPTIMA.get(n, energy)
        approx_ratio = optimal / energy if energy > 0 else 1.0

        print(f"    Time: {total_time:.2f}s, Energy: {energy:.0f}, MF: {mf:.2f}")
        print(f"    Cache hit rate: {metrics['final_cache_hit_rate']:.1%}")
        if n in KNOWN_OPTIMA:
            print(f"    Approx ratio: {approx_ratio:.3f} (optimal={optimal})")

        results.append({
            'n': n,
            'time': total_time,
            'energy': energy,
            'merit_factor': mf,
            'approx_ratio': approx_ratio,
            'cache_hit_rate': metrics['final_cache_hit_rate'],
        })

    return results


def generate_plots(all_results: dict, output_dir: Path):
    """Generate performance visualization plots."""
    print_header("Generating Plots")

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        MATPLOTLIB_AVAILABLE = True
    except ImportError:
        print("  Matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: CPU vs GPU Speedup
    if 'mts_scaling' in all_results and all_results['mts_scaling']:
        fig, ax = plt.subplots(figsize=(10, 6))
        data = all_results['mts_scaling']
        ns = [d['n'] for d in data]
        speedups_cache = [d['speedup_with_cache'] for d in data]
        speedups_no_cache = [d['speedup_no_cache'] for d in data]

        ax.plot(ns, speedups_cache, 'b-o', label='GPU with Cache', linewidth=2)
        ax.plot(ns, speedups_no_cache, 'r--s', label='GPU without Cache', linewidth=2)
        ax.axhline(y=1.0, color='gray', linestyle=':', label='Baseline (CPU)')
        ax.set_xlabel('Sequence Length (N)', fontsize=12)
        ax.set_ylabel('Speedup vs CPU', fontsize=12)
        ax.set_title('MTS Performance: CPU vs GPU', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'speedup_vs_n.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_dir / 'speedup_vs_n.png'}")

    # Plot 2: Cache Hit Rate
    if 'mts_scaling' in all_results and all_results['mts_scaling']:
        fig, ax = plt.subplots(figsize=(10, 6))
        data = all_results['mts_scaling']
        ns = [d['n'] for d in data]
        hit_rates = [d['gpu_cache_hit_rate'] * 100 for d in data]

        ax.bar(ns, hit_rates, color='green', alpha=0.7)
        ax.set_xlabel('Sequence Length (N)', fontsize=12)
        ax.set_ylabel('Cache Hit Rate (%)', fontsize=12)
        ax.set_title('Shared Cache Effectiveness', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        plt.savefig(output_dir / 'cache_hit_rate.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_dir / 'cache_hit_rate.png'}")

    # Plot 3: Scaling Analysis
    if 'large_n' in all_results and all_results['large_n']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        data = all_results['large_n']
        ns = [d['n'] for d in data]
        times = [d['time'] for d in data]
        energies = [d['energy'] for d in data]
        mfs = [d['merit_factor'] for d in data]

        # Time scaling
        ax1.semilogy(ns, times, 'b-o', linewidth=2)
        ax1.set_xlabel('Sequence Length (N)', fontsize=12)
        ax1.set_ylabel('Time (seconds, log scale)', fontsize=12)
        ax1.set_title('Time to Solution vs Problem Size', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Merit factor
        ax2.plot(ns, mfs, 'g-o', linewidth=2)
        ax2.axhline(y=12.3, color='red', linestyle='--', label='Theoretical limit ~12.3')
        ax2.set_xlabel('Sequence Length (N)', fontsize=12)
        ax2.set_ylabel('Merit Factor', fontsize=12)
        ax2.set_title('Solution Quality vs Problem Size', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'scaling_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_dir / 'scaling_analysis.png'}")

    # Plot 4: Quantum vs Classical
    if 'quantum_mts' in all_results and all_results['quantum_mts']:
        fig, ax = plt.subplots(figsize=(10, 6))
        data = all_results['quantum_mts']
        ns = [d['n'] for d in data]
        classical = [d['classical_min'] for d in data]
        quantum = [d['quantum_min'] for d in data]
        optimal = [d['optimal'] if d['optimal'] else d['classical_min'] for d in data]

        x = np.arange(len(ns))
        width = 0.25

        ax.bar(x - width, classical, width, label='Classical MTS', color='blue', alpha=0.7)
        ax.bar(x, quantum, width, label='Quantum-Enhanced MTS', color='green', alpha=0.7)
        ax.bar(x + width, optimal, width, label='Known Optimal', color='red', alpha=0.7)

        ax.set_xlabel('Sequence Length (N)', fontsize=12)
        ax.set_ylabel('Best Energy Found', fontsize=12)
        ax.set_title('Quantum Enhancement Effect on Solution Quality', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(ns)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.savefig(output_dir / 'quantum_vs_classical.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_dir / 'quantum_vs_classical.png'}")


def save_results(all_results: dict, output_dir: Path):
    """Save benchmark results to file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'benchmark_results.txt'

    with open(output_file, 'w') as f:
        f.write("LABS Solver - Phase 2 GPU Benchmark Results\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n\n")

        for benchmark_name, results in all_results.items():
            f.write(f"\n{benchmark_name.upper()}\n")
            f.write("-" * 40 + "\n")
            if isinstance(results, list):
                for r in results:
                    f.write(str(r) + "\n")
            else:
                f.write(str(results) + "\n")

    print(f"  Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='LABS GPU Benchmark')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmarks')
    parser.add_argument('--max-n', type=int, default=25, help='Maximum N to test')
    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  LABS Solver - Phase 2 GPU Benchmark".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    output_dir = Path(__file__).parent.parent / 'results'
    all_results = {}

    # Run benchmarks
    small_n = [5, 7, 9, 11, 13]
    medium_n = [7, 10, 13, 15, 17]

    all_results['energy'] = benchmark_energy_calculation(medium_n, batch_size=50 if args.quick else 100)
    all_results['mts_scaling'] = benchmark_mts_scaling(medium_n if args.quick else [7, 10, 13, 15, 17, 19], quick=args.quick)
    all_results['quantum_sampling'] = benchmark_quantum_sampling(small_n, quick=args.quick)
    all_results['quantum_mts'] = benchmark_quantum_enhanced_mts(small_n[:3] if args.quick else small_n, quick=args.quick)
    all_results['large_n'] = benchmark_large_n(max_n=args.max_n, quick=args.quick)

    # Generate outputs
    save_results(all_results, output_dir)
    generate_plots(all_results, output_dir)

    print_header("Benchmark Complete")
    print(f"\n  Results saved to: {output_dir}")
    print(f"  GPU available: {CUPY_AVAILABLE}")
    print(f"  CUDA-Q GPU: {CUDAQ_GPU}")


if __name__ == "__main__":
    main()
