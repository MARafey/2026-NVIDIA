#!/usr/bin/env python3
"""
Complete Benchmark Script for LABS Solver
==========================================

This script runs comprehensive benchmarks comparing:
1. Basic HPC (CPU) execution
2. GPU-accelerated execution (CuPy)
3. Quantum-enhanced execution (CUDA-Q)
4. Full hybrid pipeline

Generates detailed statistics, timing comparisons, and publication-ready plots.

Usage:
    python run_complete_benchmark.py [--quick] [--max-n N]

Run on Brev GPU:
    brev shell nvidia-iquhack-2026-challenge-519c62
    cd /path/to/team-submissions/src
    python run_complete_benchmark.py
"""

import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

# =============================================================================
# Environment Detection
# =============================================================================

print("=" * 80)
print("LABS Solver - Complete Benchmark Suite")
print("=" * 80)
print(f"Timestamp: {datetime.now().isoformat()}")
print()

# Check CuPy
try:
    import cupy as cp
    GPU_NAME = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    GPU_MEM = cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] / 1e9
    print(f"[+] CuPy: Available")
    print(f"    GPU: {GPU_NAME}")
    print(f"    Memory: {GPU_MEM:.1f} GB")
    CUPY_AVAILABLE = True
except ImportError:
    print("[-] CuPy: Not available (CPU fallback)")
    CUPY_AVAILABLE = False
    GPU_NAME = "N/A"
    GPU_MEM = 0
except Exception as e:
    print(f"[-] CuPy: Error - {e}")
    CUPY_AVAILABLE = False
    GPU_NAME = "N/A"
    GPU_MEM = 0

# Check CUDA-Q
try:
    import cudaq
    print(f"[+] CUDA-Q: Available")
    try:
        cudaq.set_target("nvidia")
        CUDAQ_TARGET = "nvidia"
        print(f"    Target: nvidia (GPU accelerated)")
    except:
        try:
            cudaq.set_target("nvidia-fp64")
            CUDAQ_TARGET = "nvidia-fp64"
            print(f"    Target: nvidia-fp64 (GPU accelerated)")
        except:
            CUDAQ_TARGET = "default"
            print(f"    Target: default (CPU simulation)")
    CUDAQ_AVAILABLE = True
except ImportError:
    print("[-] CUDA-Q: Not available")
    CUDAQ_AVAILABLE = False
    CUDAQ_TARGET = None

print()

# Import our modules
from labs_solver.energy import calculate_energy, merit_factor
from labs_solver.mts import memetic_tabu_search as cpu_mts, random_sequence
from labs_solver.utils import get_interactions, pack_sequence

if CUPY_AVAILABLE:
    from labs_solver.energy_gpu import (
        calculate_energy_batch_gpu,
        calculate_energy_batch_cpu,
        GPUEnergyCache,
    )
    from labs_solver.mts_gpu import memetic_tabu_search_gpu, benchmark_cpu_vs_gpu

if CUDAQ_AVAILABLE:
    from labs_solver.quantum import sample_quantum_population, quantum_enhanced_mts

# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class QuantumCircuitStats:
    """Statistics about quantum circuit complexity."""
    n_qubits: int
    n_trotter_steps: int
    n_2qubit_pairs: int
    n_4qubit_groups: int
    total_2qubit_gates: int  # R_YZ + R_ZY per pair
    total_4qubit_gates: int  # 4 gates per group (R_YZZZ, R_ZYZZ, R_ZZYZ, R_ZZZY)
    circuit_depth_estimate: int
    total_gate_count: int

@dataclass
class BenchmarkResult:
    """Single benchmark run result."""
    n: int
    method: str
    energy: float
    time_seconds: float
    merit_factor: float
    optimal_energy: Optional[int]
    approx_ratio: Optional[float]
    extra: Dict

# =============================================================================
# Known Optimal Energies
# =============================================================================

KNOWN_OPTIMA = {
    3: 1, 4: 2, 5: 2, 6: 7, 7: 3, 8: 8, 9: 12, 10: 13,
    11: 12, 13: 13, 15: 28, 17: 32, 19: 52, 21: 42, 23: 58, 25: 76,
}

def get_optimal(n: int) -> Optional[int]:
    return KNOWN_OPTIMA.get(n)

# =============================================================================
# Quantum Circuit Analysis
# =============================================================================

def analyze_quantum_circuit(n: int, n_steps: int = 1) -> QuantumCircuitStats:
    """
    Analyze the quantum circuit complexity for a given problem size.

    The counteradiabatic circuit uses:
    - 2-body terms: R_YZ and R_ZY gates for each pair (i,j) with |i-j| < n/2
    - 4-body terms: R_YZZZ, R_ZYZZ, R_ZZYZ, R_ZZZY for each 4-tuple
    """
    G2, G4 = get_interactions(n)

    n_2qubit_pairs = len(G2)
    n_4qubit_groups = len(G4)

    # Per Trotter step:
    # - 2 gates per 2-body term (R_YZ + R_ZY)
    # - 4 gates per 4-body term (R_YZZZ, R_ZYZZ, R_ZZYZ, R_ZZZY)
    gates_2q_per_step = n_2qubit_pairs * 2
    gates_4q_per_step = n_4qubit_groups * 4

    total_2q = gates_2q_per_step * n_steps
    total_4q = gates_4q_per_step * n_steps

    # Each 2-qubit gate decomposes to: RX + 2 CNOT + RZ + 2 RX ~= 5 base gates
    # Each 4-qubit gate decomposes to: RX + 6 CNOT + RZ ~= 8 base gates
    depth_2q = 5 * gates_2q_per_step
    depth_4q = 8 * gates_4q_per_step

    # Plus initial H gates
    depth_estimate = n + (depth_2q + depth_4q) * n_steps

    return QuantumCircuitStats(
        n_qubits=n,
        n_trotter_steps=n_steps,
        n_2qubit_pairs=n_2qubit_pairs,
        n_4qubit_groups=n_4qubit_groups,
        total_2qubit_gates=total_2q,
        total_4qubit_gates=total_4q,
        circuit_depth_estimate=depth_estimate,
        total_gate_count=n + total_2q + total_4q,  # H gates + parametric gates
    )

# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_cpu_mts(n: int, pop_size: int, generations: int, trials: int = 3) -> BenchmarkResult:
    """Benchmark CPU-based MTS."""
    times = []
    energies = []

    for _ in range(trials):
        start = time.time()
        seq, energy, _, _ = cpu_mts(
            n, population_size=pop_size, max_generations=generations
        )
        elapsed = time.time() - start
        times.append(elapsed)
        energies.append(energy)

    best_energy = min(energies)
    avg_time = np.mean(times)
    mf = (n * n) / (2 * best_energy) if best_energy > 0 else float('inf')
    opt = get_optimal(n)

    return BenchmarkResult(
        n=n,
        method="CPU MTS",
        energy=best_energy,
        time_seconds=avg_time,
        merit_factor=mf,
        optimal_energy=opt,
        approx_ratio=opt / best_energy if opt and best_energy > 0 else None,
        extra={
            'trials': trials,
            'all_energies': energies,
            'all_times': times,
            'population_size': pop_size,
            'generations': generations,
        }
    )

def benchmark_gpu_mts(n: int, pop_size: int, generations: int, trials: int = 3) -> Optional[BenchmarkResult]:
    """Benchmark GPU-accelerated MTS with shared cache."""
    if not CUPY_AVAILABLE:
        return None

    times = []
    energies = []
    cache_hit_rates = []

    for _ in range(trials):
        start = time.time()
        seq, energy, _, _, metrics = memetic_tabu_search_gpu(
            n, population_size=pop_size, max_generations=generations,
            use_shared_cache=True
        )
        elapsed = time.time() - start
        times.append(elapsed)
        energies.append(energy)
        cache_hit_rates.append(metrics.get('final_cache_hit_rate', 0))

    best_energy = min(energies)
    avg_time = np.mean(times)
    mf = (n * n) / (2 * best_energy) if best_energy > 0 else float('inf')
    opt = get_optimal(n)

    return BenchmarkResult(
        n=n,
        method="GPU MTS (CuPy + Cache)",
        energy=best_energy,
        time_seconds=avg_time,
        merit_factor=mf,
        optimal_energy=opt,
        approx_ratio=opt / best_energy if opt and best_energy > 0 else None,
        extra={
            'trials': trials,
            'all_energies': energies,
            'all_times': times,
            'cache_hit_rates': cache_hit_rates,
            'avg_cache_hit_rate': np.mean(cache_hit_rates),
            'population_size': pop_size,
            'generations': generations,
            'gpu_name': GPU_NAME,
        }
    )

def benchmark_quantum_sampling(n: int, pop_size: int, shots: int, trials: int = 3) -> Optional[Tuple[BenchmarkResult, QuantumCircuitStats]]:
    """Benchmark quantum circuit sampling."""
    if not CUDAQ_AVAILABLE:
        return None

    times = []
    min_energies = []
    mean_energies = []

    for _ in range(trials):
        start = time.time()
        population = sample_quantum_population(n, population_size=pop_size, n_steps=1, T=1.0, shots=shots)
        elapsed = time.time() - start
        times.append(elapsed)

        energies = [calculate_energy(seq) for seq in population]
        min_energies.append(min(energies))
        mean_energies.append(np.mean(energies))

    best_energy = min(min_energies)
    avg_time = np.mean(times)
    mf = (n * n) / (2 * best_energy) if best_energy > 0 else float('inf')
    opt = get_optimal(n)

    circuit_stats = analyze_quantum_circuit(n, n_steps=1)

    result = BenchmarkResult(
        n=n,
        method="Quantum Sampling",
        energy=best_energy,
        time_seconds=avg_time,
        merit_factor=mf,
        optimal_energy=opt,
        approx_ratio=opt / best_energy if opt and best_energy > 0 else None,
        extra={
            'trials': trials,
            'all_min_energies': min_energies,
            'all_mean_energies': mean_energies,
            'all_times': times,
            'shots': shots,
            'population_size': pop_size,
            'cudaq_target': CUDAQ_TARGET,
        }
    )

    return result, circuit_stats

def benchmark_quantum_enhanced_mts(n: int, pop_size: int, generations: int, trials: int = 3) -> Optional[BenchmarkResult]:
    """Benchmark quantum-enhanced MTS (quantum seeding + classical MTS)."""
    if not CUDAQ_AVAILABLE:
        return None

    times = []
    energies = []
    quantum_times = []
    mts_times = []

    for _ in range(trials):
        # Time quantum sampling
        q_start = time.time()
        quantum_pop = sample_quantum_population(n, population_size=pop_size, n_steps=1, T=1.0)
        q_time = time.time() - q_start
        quantum_times.append(q_time)

        # Time MTS refinement
        mts_start = time.time()
        seq, energy, _, _ = cpu_mts(
            n, population_size=pop_size, max_generations=generations,
            initial_population=quantum_pop
        )
        mts_time = time.time() - mts_start
        mts_times.append(mts_time)

        times.append(q_time + mts_time)
        energies.append(energy)

    best_energy = min(energies)
    avg_time = np.mean(times)
    mf = (n * n) / (2 * best_energy) if best_energy > 0 else float('inf')
    opt = get_optimal(n)

    return BenchmarkResult(
        n=n,
        method="Quantum-Enhanced MTS",
        energy=best_energy,
        time_seconds=avg_time,
        merit_factor=mf,
        optimal_energy=opt,
        approx_ratio=opt / best_energy if opt and best_energy > 0 else None,
        extra={
            'trials': trials,
            'all_energies': energies,
            'all_times': times,
            'avg_quantum_time': np.mean(quantum_times),
            'avg_mts_time': np.mean(mts_times),
            'quantum_time_fraction': np.mean(quantum_times) / avg_time,
            'population_size': pop_size,
            'generations': generations,
        }
    )

def benchmark_full_hybrid_pipeline(n: int, pop_size: int, generations: int) -> Optional[BenchmarkResult]:
    """Benchmark the full hybrid pipeline: Quantum → GPU MTS."""
    if not CUDAQ_AVAILABLE or not CUPY_AVAILABLE:
        return None

    total_start = time.time()

    # Stage 1: Quantum sampling
    q_start = time.time()
    quantum_pop = sample_quantum_population(n, population_size=pop_size, n_steps=2, T=1.5)
    q_time = time.time() - q_start

    quantum_energies = [calculate_energy(seq) for seq in quantum_pop]

    # Stage 2: GPU MTS with quantum seeds
    gpu_start = time.time()
    seq, energy, _, _, metrics = memetic_tabu_search_gpu(
        n, population_size=pop_size, max_generations=generations,
        initial_population=quantum_pop, use_shared_cache=True
    )
    gpu_time = time.time() - gpu_start

    total_time = time.time() - total_start
    mf = (n * n) / (2 * energy) if energy > 0 else float('inf')
    opt = get_optimal(n)

    return BenchmarkResult(
        n=n,
        method="Full Hybrid (Quantum + GPU MTS)",
        energy=energy,
        time_seconds=total_time,
        merit_factor=mf,
        optimal_energy=opt,
        approx_ratio=opt / energy if opt and energy > 0 else None,
        extra={
            'quantum_time': q_time,
            'gpu_mts_time': gpu_time,
            'quantum_min_energy': min(quantum_energies),
            'quantum_mean_energy': np.mean(quantum_energies),
            'improvement_from_quantum': (min(quantum_energies) - energy) / min(quantum_energies) * 100 if min(quantum_energies) > 0 else 0,
            'cache_hit_rate': metrics.get('final_cache_hit_rate', 0),
            'population_size': pop_size,
            'generations': generations,
        }
    )

# =============================================================================
# Main Benchmark Runner
# =============================================================================

def run_complete_benchmark(quick: bool = False, max_n: int = 25) -> Dict:
    """Run all benchmarks and collect results."""

    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'gpu_name': GPU_NAME,
            'gpu_memory_gb': GPU_MEM,
            'cupy_available': CUPY_AVAILABLE,
            'cudaq_available': CUDAQ_AVAILABLE,
            'cudaq_target': CUDAQ_TARGET,
            'quick_mode': quick,
            'max_n': max_n,
        },
        'quantum_circuit_stats': [],
        'cpu_mts': [],
        'gpu_mts': [],
        'quantum_sampling': [],
        'quantum_enhanced_mts': [],
        'hybrid_pipeline': [],
        'time_comparisons': [],
    }

    # Test configurations
    if quick:
        small_n = [5, 7, 9]
        medium_n = [11, 13, 15]
        large_n = [17, 19, 21]
        pop_size = 12
        generations = 25
        trials = 2
    else:
        small_n = [5, 7, 9, 11]
        medium_n = [13, 15, 17]
        large_n = list(range(19, min(max_n + 1, 26), 2))
        pop_size = 20
        generations = 50
        trials = 3

    all_n = small_n + medium_n + large_n

    # ==========================================================================
    # Benchmark 1: Quantum Circuit Analysis
    # ==========================================================================
    print("\n" + "=" * 80)
    print("BENCHMARK 1: Quantum Circuit Analysis")
    print("=" * 80)

    print(f"\n{'N':>4} | {'Qubits':>6} | {'2Q Pairs':>8} | {'4Q Groups':>9} | {'Total Gates':>11} | {'Depth Est':>10}")
    print("-" * 70)

    for n in all_n:
        stats = analyze_quantum_circuit(n, n_steps=1)
        results['quantum_circuit_stats'].append(asdict(stats))
        print(f"{n:>4} | {stats.n_qubits:>6} | {stats.n_2qubit_pairs:>8} | {stats.n_4qubit_groups:>9} | {stats.total_gate_count:>11} | {stats.circuit_depth_estimate:>10}")

    # ==========================================================================
    # Benchmark 2: CPU MTS Baseline
    # ==========================================================================
    print("\n" + "=" * 80)
    print("BENCHMARK 2: CPU MTS (Baseline)")
    print("=" * 80)

    print(f"\n{'N':>4} | {'Energy':>8} | {'Time (s)':>10} | {'MF':>8} | {'Approx':>8}")
    print("-" * 50)

    for n in all_n:
        result = benchmark_cpu_mts(n, pop_size, generations, trials)
        results['cpu_mts'].append(asdict(result))
        ratio_str = f"{result.approx_ratio:.4f}" if result.approx_ratio else "N/A"
        print(f"{n:>4} | {result.energy:>8.0f} | {result.time_seconds:>10.3f} | {result.merit_factor:>8.3f} | {ratio_str:>8}")

    # ==========================================================================
    # Benchmark 3: GPU MTS
    # ==========================================================================
    if CUPY_AVAILABLE:
        print("\n" + "=" * 80)
        print("BENCHMARK 3: GPU MTS (CuPy + Shared Cache)")
        print("=" * 80)

        print(f"\n{'N':>4} | {'Energy':>8} | {'Time (s)':>10} | {'Cache Hit':>10} | {'Speedup':>8}")
        print("-" * 55)

        for i, n in enumerate(all_n):
            result = benchmark_gpu_mts(n, pop_size, generations, trials)
            if result:
                results['gpu_mts'].append(asdict(result))
                cpu_time = results['cpu_mts'][i]['time_seconds']
                speedup = cpu_time / result.time_seconds if result.time_seconds > 0 else 0
                hit_rate = result.extra['avg_cache_hit_rate'] * 100
                print(f"{n:>4} | {result.energy:>8.0f} | {result.time_seconds:>10.3f} | {hit_rate:>9.1f}% | {speedup:>7.2f}x")

    # ==========================================================================
    # Benchmark 4: Quantum Sampling
    # ==========================================================================
    if CUDAQ_AVAILABLE:
        print("\n" + "=" * 80)
        print("BENCHMARK 4: Quantum Circuit Sampling")
        print("=" * 80)

        quantum_test_n = small_n + medium_n[:2]  # Quantum is expensive for large N

        print(f"\n{'N':>4} | {'Min E':>8} | {'Time (s)':>10} | {'Gates':>8} | {'Depth':>8}")
        print("-" * 50)

        for n in quantum_test_n:
            result_tuple = benchmark_quantum_sampling(n, pop_size, shots=500, trials=trials)
            if result_tuple:
                result, circuit_stats = result_tuple
                results['quantum_sampling'].append(asdict(result))
                print(f"{n:>4} | {result.energy:>8.0f} | {result.time_seconds:>10.3f} | {circuit_stats.total_gate_count:>8} | {circuit_stats.circuit_depth_estimate:>8}")

    # ==========================================================================
    # Benchmark 5: Quantum-Enhanced MTS
    # ==========================================================================
    if CUDAQ_AVAILABLE:
        print("\n" + "=" * 80)
        print("BENCHMARK 5: Quantum-Enhanced MTS (Quantum Seeding + Classical MTS)")
        print("=" * 80)

        quantum_mts_n = small_n + medium_n[:2]

        print(f"\n{'N':>4} | {'Energy':>8} | {'Total (s)':>10} | {'Q Time':>8} | {'MTS Time':>8} | {'Improvement':>11}")
        print("-" * 65)

        for i, n in enumerate(quantum_mts_n):
            result = benchmark_quantum_enhanced_mts(n, pop_size, generations, trials)
            if result:
                results['quantum_enhanced_mts'].append(asdict(result))

                # Find corresponding CPU result for comparison
                cpu_result = next((r for r in results['cpu_mts'] if r['n'] == n), None)
                improvement = ""
                if cpu_result:
                    cpu_energy = cpu_result['energy']
                    if cpu_energy > 0:
                        imp = (cpu_energy - result.energy) / cpu_energy * 100
                        improvement = f"{imp:+.1f}%"

                print(f"{n:>4} | {result.energy:>8.0f} | {result.time_seconds:>10.3f} | {result.extra['avg_quantum_time']:>8.3f} | {result.extra['avg_mts_time']:>8.3f} | {improvement:>11}")

    # ==========================================================================
    # Benchmark 6: Full Hybrid Pipeline
    # ==========================================================================
    if CUDAQ_AVAILABLE and CUPY_AVAILABLE:
        print("\n" + "=" * 80)
        print("BENCHMARK 6: Full Hybrid Pipeline (Quantum + GPU MTS)")
        print("=" * 80)

        hybrid_n = small_n + medium_n

        print(f"\n{'N':>4} | {'Energy':>8} | {'Total (s)':>10} | {'Q→MTS Imp':>10} | {'Cache Hit':>10}")
        print("-" * 55)

        for n in hybrid_n:
            result = benchmark_full_hybrid_pipeline(n, pop_size, generations)
            if result:
                results['hybrid_pipeline'].append(asdict(result))
                imp = result.extra['improvement_from_quantum']
                hit = result.extra['cache_hit_rate'] * 100
                print(f"{n:>4} | {result.energy:>8.0f} | {result.time_seconds:>10.3f} | {imp:>9.1f}% | {hit:>9.1f}%")

    # ==========================================================================
    # Time Comparison Summary
    # ==========================================================================
    print("\n" + "=" * 80)
    print("TIME COMPARISON SUMMARY: Basic HPC vs Quantum Execution")
    print("=" * 80)

    print(f"\n{'N':>4} | {'CPU MTS':>10} | {'GPU MTS':>10} | {'Quantum+MTS':>12} | {'Hybrid':>10} | {'Best Method':>15}")
    print("-" * 75)

    for n in all_n:
        cpu_r = next((r for r in results['cpu_mts'] if r['n'] == n), None)
        gpu_r = next((r for r in results['gpu_mts'] if r['n'] == n), None)
        qmts_r = next((r for r in results['quantum_enhanced_mts'] if r['n'] == n), None)
        hybrid_r = next((r for r in results['hybrid_pipeline'] if r['n'] == n), None)

        cpu_t = f"{cpu_r['time_seconds']:.3f}s" if cpu_r else "N/A"
        gpu_t = f"{gpu_r['time_seconds']:.3f}s" if gpu_r else "N/A"
        qmts_t = f"{qmts_r['time_seconds']:.3f}s" if qmts_r else "N/A"
        hybrid_t = f"{hybrid_r['time_seconds']:.3f}s" if hybrid_r else "N/A"

        # Find best energy
        methods = []
        if cpu_r: methods.append(('CPU MTS', cpu_r['energy']))
        if gpu_r: methods.append(('GPU MTS', gpu_r['energy']))
        if qmts_r: methods.append(('Q+MTS', qmts_r['energy']))
        if hybrid_r: methods.append(('Hybrid', hybrid_r['energy']))

        best = min(methods, key=lambda x: x[1]) if methods else ('N/A', 0)

        results['time_comparisons'].append({
            'n': n,
            'cpu_time': cpu_r['time_seconds'] if cpu_r else None,
            'gpu_time': gpu_r['time_seconds'] if gpu_r else None,
            'quantum_mts_time': qmts_r['time_seconds'] if qmts_r else None,
            'hybrid_time': hybrid_r['time_seconds'] if hybrid_r else None,
            'best_method': best[0],
            'best_energy': best[1],
        })

        print(f"{n:>4} | {cpu_t:>10} | {gpu_t:>10} | {qmts_t:>12} | {hybrid_t:>10} | {best[0]:>15}")

    return results

# =============================================================================
# Plot Generation
# =============================================================================

def generate_plots(results: Dict, output_dir: Path):
    """Generate publication-ready plots."""

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-whitegrid')
    except ImportError:
        print("\n[!] Matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # Plot 1: Time Comparison (CPU vs GPU vs Quantum)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))

    comparisons = results['time_comparisons']
    ns = [c['n'] for c in comparisons]

    cpu_times = [c['cpu_time'] or 0 for c in comparisons]
    gpu_times = [c['gpu_time'] or 0 for c in comparisons]
    quantum_times = [c['quantum_mts_time'] or 0 for c in comparisons]
    hybrid_times = [c['hybrid_time'] or 0 for c in comparisons]

    x = np.arange(len(ns))
    width = 0.2

    if any(cpu_times):
        ax.bar(x - 1.5*width, cpu_times, width, label='CPU MTS', color='#3498db', alpha=0.8)
    if any(gpu_times):
        ax.bar(x - 0.5*width, gpu_times, width, label='GPU MTS', color='#2ecc71', alpha=0.8)
    if any(quantum_times):
        ax.bar(x + 0.5*width, quantum_times, width, label='Quantum+MTS', color='#9b59b6', alpha=0.8)
    if any(hybrid_times):
        ax.bar(x + 1.5*width, hybrid_times, width, label='Full Hybrid', color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Sequence Length (N)', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Execution Time Comparison: Basic HPC vs Quantum', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ns)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'time_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'time_comparison.png'}")

    # --------------------------------------------------------------------------
    # Plot 2: GPU Speedup
    # --------------------------------------------------------------------------
    if results['gpu_mts']:
        fig, ax = plt.subplots(figsize=(10, 6))

        ns = [r['n'] for r in results['gpu_mts']]
        speedups = []
        for gpu_r in results['gpu_mts']:
            cpu_r = next((c for c in results['cpu_mts'] if c['n'] == gpu_r['n']), None)
            if cpu_r:
                speedups.append(cpu_r['time_seconds'] / gpu_r['time_seconds'])
            else:
                speedups.append(1.0)

        ax.bar(ns, speedups, color='#2ecc71', alpha=0.8, edgecolor='black')
        ax.axhline(y=1.0, color='red', linestyle='--', label='CPU Baseline')
        ax.set_xlabel('Sequence Length (N)', fontsize=12)
        ax.set_ylabel('Speedup (vs CPU)', fontsize=12)
        ax.set_title('GPU Acceleration: Speedup over CPU MTS', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'gpu_speedup.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_dir / 'gpu_speedup.png'}")

    # --------------------------------------------------------------------------
    # Plot 3: Cache Hit Rate
    # --------------------------------------------------------------------------
    if results['gpu_mts']:
        fig, ax = plt.subplots(figsize=(10, 6))

        ns = [r['n'] for r in results['gpu_mts']]
        hit_rates = [r['extra']['avg_cache_hit_rate'] * 100 for r in results['gpu_mts']]

        ax.bar(ns, hit_rates, color='#f39c12', alpha=0.8, edgecolor='black')
        ax.axhline(y=20, color='red', linestyle='--', label='Target (20%)')
        ax.axhline(y=40, color='green', linestyle='--', label='Stretch Goal (40%)')
        ax.set_xlabel('Sequence Length (N)', fontsize=12)
        ax.set_ylabel('Cache Hit Rate (%)', fontsize=12)
        ax.set_title('Shared Exploration Cache Effectiveness', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'cache_hit_rate.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_dir / 'cache_hit_rate.png'}")

    # --------------------------------------------------------------------------
    # Plot 4: Quantum Circuit Complexity
    # --------------------------------------------------------------------------
    if results['quantum_circuit_stats']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        stats = results['quantum_circuit_stats']
        ns = [s['n_qubits'] for s in stats]
        gates = [s['total_gate_count'] for s in stats]
        depths = [s['circuit_depth_estimate'] for s in stats]

        ax1.semilogy(ns, gates, 'b-o', linewidth=2, markersize=8)
        ax1.set_xlabel('Sequence Length (N)', fontsize=12)
        ax1.set_ylabel('Total Gate Count (log scale)', fontsize=12)
        ax1.set_title('Quantum Circuit Gate Count', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        ax2.semilogy(ns, depths, 'r-s', linewidth=2, markersize=8)
        ax2.set_xlabel('Sequence Length (N)', fontsize=12)
        ax2.set_ylabel('Circuit Depth Estimate (log scale)', fontsize=12)
        ax2.set_title('Quantum Circuit Depth', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'quantum_circuit_complexity.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_dir / 'quantum_circuit_complexity.png'}")

    # --------------------------------------------------------------------------
    # Plot 5: Solution Quality (Energy vs Optimal)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    # Collect data from all methods
    methods_data = {
        'CPU MTS': results['cpu_mts'],
        'GPU MTS': results['gpu_mts'],
        'Quantum+MTS': results['quantum_enhanced_mts'],
        'Hybrid': results['hybrid_pipeline'],
    }

    colors = {'CPU MTS': '#3498db', 'GPU MTS': '#2ecc71', 'Quantum+MTS': '#9b59b6', 'Hybrid': '#e74c3c'}
    markers = {'CPU MTS': 'o', 'GPU MTS': 's', 'Quantum+MTS': '^', 'Hybrid': 'D'}

    for method, data in methods_data.items():
        if data:
            ns = [r['n'] for r in data if r.get('approx_ratio')]
            ratios = [r['approx_ratio'] for r in data if r.get('approx_ratio')]
            if ns and ratios:
                ax.plot(ns, ratios, f'{markers[method]}-', color=colors[method],
                       label=method, linewidth=2, markersize=8, alpha=0.8)

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Optimal')
    ax.axhline(y=0.95, color='gray', linestyle=':', label='Target (0.95)')
    ax.set_xlabel('Sequence Length (N)', fontsize=12)
    ax.set_ylabel('Approximation Ratio (optimal/found)', fontsize=12)
    ax.set_title('Solution Quality: Approximation Ratio vs Problem Size', fontsize=14, fontweight='bold')
    ax.set_ylim(0.5, 1.05)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'solution_quality.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'solution_quality.png'}")

    # --------------------------------------------------------------------------
    # Plot 6: Merit Factor Scaling
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, data in methods_data.items():
        if data:
            ns = [r['n'] for r in data]
            mfs = [r['merit_factor'] for r in data]
            if ns and mfs:
                ax.plot(ns, mfs, f'{markers[method]}-', color=colors[method],
                       label=method, linewidth=2, markersize=8, alpha=0.8)

    ax.axhline(y=12.3, color='red', linestyle='--', linewidth=2, label='Theoretical Limit (~12.3)')
    ax.set_xlabel('Sequence Length (N)', fontsize=12)
    ax.set_ylabel('Merit Factor (N²/2E)', fontsize=12)
    ax.set_title('Merit Factor vs Problem Size', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'merit_factor.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'merit_factor.png'}")

    # --------------------------------------------------------------------------
    # Plot 7: Quantum vs Classical Energy Distribution
    # --------------------------------------------------------------------------
    if results['quantum_sampling'] and results['cpu_mts']:
        fig, ax = plt.subplots(figsize=(10, 6))

        q_data = results['quantum_sampling']
        c_data = results['cpu_mts']

        common_ns = set(r['n'] for r in q_data) & set(r['n'] for r in c_data)
        common_ns = sorted(common_ns)

        x = np.arange(len(common_ns))
        width = 0.35

        q_energies = [next(r['energy'] for r in q_data if r['n'] == n) for n in common_ns]
        c_energies = [next(r['energy'] for r in c_data if r['n'] == n) for n in common_ns]
        optimals = [KNOWN_OPTIMA.get(n, None) for n in common_ns]

        ax.bar(x - width/2, q_energies, width, label='Quantum Sampling', color='#9b59b6', alpha=0.8)
        ax.bar(x + width/2, c_energies, width, label='CPU MTS', color='#3498db', alpha=0.8)

        # Add optimal markers
        for i, opt in enumerate(optimals):
            if opt:
                ax.scatter([i], [opt], color='red', s=100, marker='*', zorder=5,
                          label='Optimal' if i == 0 else '')

        ax.set_xlabel('Sequence Length (N)', fontsize=12)
        ax.set_ylabel('Best Energy Found', fontsize=12)
        ax.set_title('Solution Quality: Quantum Sampling vs Classical MTS', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(common_ns)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_dir / 'quantum_vs_classical_energy.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_dir / 'quantum_vs_classical_energy.png'}")

# =============================================================================
# Report Generation
# =============================================================================

def generate_report(results: Dict, output_dir: Path):
    """Generate a comprehensive markdown report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / 'BENCHMARK_REPORT.md'

    meta = results['metadata']

    with open(report_path, 'w') as f:
        f.write("# LABS Solver - Complete Benchmark Report\n\n")
        f.write(f"**Generated:** {meta['timestamp']}\n\n")

        f.write("## Executive Summary\n\n")
        f.write("This report presents comprehensive benchmarks comparing Basic HPC (CPU) execution ")
        f.write("with Quantum-enhanced and GPU-accelerated approaches for solving the ")
        f.write("Low Autocorrelation Binary Sequences (LABS) problem.\n\n")

        # Environment
        f.write("## Environment\n\n")
        f.write("| Component | Status |\n")
        f.write("|-----------|--------|\n")
        f.write(f"| GPU | {meta['gpu_name']} ({meta['gpu_memory_gb']:.1f} GB) |\n")
        f.write(f"| CuPy | {'Available' if meta['cupy_available'] else 'Not Available'} |\n")
        f.write(f"| CUDA-Q | {'Available' if meta['cudaq_available'] else 'Not Available'} |\n")
        f.write(f"| CUDA-Q Target | {meta['cudaq_target'] or 'N/A'} |\n\n")

        # Quantum Circuit Analysis
        if results['quantum_circuit_stats']:
            f.write("## Quantum Circuit Complexity\n\n")
            f.write("The counteradiabatic quantum circuit scales as follows:\n\n")
            f.write("| N | Qubits | 2Q Pairs | 4Q Groups | Total Gates | Depth Est |\n")
            f.write("|---|--------|----------|-----------|-------------|----------|\n")
            for s in results['quantum_circuit_stats']:
                f.write(f"| {s['n_qubits']} | {s['n_qubits']} | {s['n_2qubit_pairs']} | ")
                f.write(f"{s['n_4qubit_groups']} | {s['total_gate_count']} | {s['circuit_depth_estimate']} |\n")
            f.write("\n")

        # Time Comparison
        f.write("## Time Comparison: Basic HPC vs Quantum\n\n")
        f.write("| N | CPU MTS | GPU MTS | Quantum+MTS | Hybrid | Best Method |\n")
        f.write("|---|---------|---------|-------------|--------|-------------|\n")
        for c in results['time_comparisons']:
            cpu_t = f"{c['cpu_time']:.3f}s" if c['cpu_time'] else "N/A"
            gpu_t = f"{c['gpu_time']:.3f}s" if c['gpu_time'] else "N/A"
            q_t = f"{c['quantum_mts_time']:.3f}s" if c['quantum_mts_time'] else "N/A"
            h_t = f"{c['hybrid_time']:.3f}s" if c['hybrid_time'] else "N/A"
            f.write(f"| {c['n']} | {cpu_t} | {gpu_t} | {q_t} | {h_t} | {c['best_method']} |\n")
        f.write("\n")

        # GPU Performance
        if results['gpu_mts']:
            f.write("## GPU Acceleration Performance\n\n")
            f.write("| N | CPU Time | GPU Time | Speedup | Cache Hit Rate |\n")
            f.write("|---|----------|----------|---------|----------------|\n")
            for gpu_r in results['gpu_mts']:
                cpu_r = next((c for c in results['cpu_mts'] if c['n'] == gpu_r['n']), None)
                if cpu_r:
                    speedup = cpu_r['time_seconds'] / gpu_r['time_seconds']
                    hit_rate = gpu_r['extra']['avg_cache_hit_rate'] * 100
                    f.write(f"| {gpu_r['n']} | {cpu_r['time_seconds']:.3f}s | ")
                    f.write(f"{gpu_r['time_seconds']:.3f}s | {speedup:.2f}x | {hit_rate:.1f}% |\n")
            f.write("\n")

        # Solution Quality
        f.write("## Solution Quality\n\n")
        f.write("### Approximation Ratios (optimal/found)\n\n")
        f.write("| N | CPU MTS | GPU MTS | Quantum+MTS | Hybrid | Optimal |\n")
        f.write("|---|---------|---------|-------------|--------|--------|\n")

        all_ns = sorted(set(r['n'] for r in results['cpu_mts']))
        for n in all_ns:
            cpu_r = next((r for r in results['cpu_mts'] if r['n'] == n), None)
            gpu_r = next((r for r in results['gpu_mts'] if r['n'] == n), None)
            qmts_r = next((r for r in results['quantum_enhanced_mts'] if r['n'] == n), None)
            hybrid_r = next((r for r in results['hybrid_pipeline'] if r['n'] == n), None)
            opt = KNOWN_OPTIMA.get(n, "?")

            cpu_ratio = f"{cpu_r['approx_ratio']:.4f}" if cpu_r and cpu_r['approx_ratio'] else "N/A"
            gpu_ratio = f"{gpu_r['approx_ratio']:.4f}" if gpu_r and gpu_r['approx_ratio'] else "N/A"
            q_ratio = f"{qmts_r['approx_ratio']:.4f}" if qmts_r and qmts_r['approx_ratio'] else "N/A"
            h_ratio = f"{hybrid_r['approx_ratio']:.4f}" if hybrid_r and hybrid_r['approx_ratio'] else "N/A"

            f.write(f"| {n} | {cpu_ratio} | {gpu_ratio} | {q_ratio} | {h_ratio} | {opt} |\n")
        f.write("\n")

        # Plots
        f.write("## Visualizations\n\n")
        f.write("The following plots have been generated:\n\n")
        f.write("1. **time_comparison.png** - Execution time comparison across all methods\n")
        f.write("2. **gpu_speedup.png** - GPU acceleration speedup over CPU baseline\n")
        f.write("3. **cache_hit_rate.png** - Shared exploration cache effectiveness\n")
        f.write("4. **quantum_circuit_complexity.png** - Circuit gate count and depth scaling\n")
        f.write("5. **solution_quality.png** - Approximation ratio vs problem size\n")
        f.write("6. **merit_factor.png** - Merit factor scaling\n")
        f.write("7. **quantum_vs_classical_energy.png** - Energy comparison\n\n")

        # Conclusions
        f.write("## Conclusions\n\n")

        # Calculate key metrics
        if results['gpu_mts'] and results['cpu_mts']:
            avg_speedup = np.mean([
                c['time_seconds'] / g['time_seconds']
                for c, g in zip(results['cpu_mts'], results['gpu_mts'])
                if g['time_seconds'] > 0
            ])
            avg_cache_hit = np.mean([r['extra']['avg_cache_hit_rate'] for r in results['gpu_mts']])
            f.write(f"- **Average GPU Speedup:** {avg_speedup:.2f}x over CPU baseline\n")
            f.write(f"- **Average Cache Hit Rate:** {avg_cache_hit*100:.1f}%\n")

        if results['quantum_enhanced_mts'] and results['cpu_mts']:
            quantum_ns = [r['n'] for r in results['quantum_enhanced_mts']]
            improvements = []
            for qr in results['quantum_enhanced_mts']:
                cr = next((c for c in results['cpu_mts'] if c['n'] == qr['n']), None)
                if cr and cr['energy'] > 0:
                    imp = (cr['energy'] - qr['energy']) / cr['energy'] * 100
                    improvements.append(imp)
            if improvements:
                avg_imp = np.mean(improvements)
                f.write(f"- **Quantum Enhancement:** {avg_imp:+.1f}% average energy improvement\n")

        f.write("\n---\n")
        f.write("*Report generated by LABS Solver Benchmark Suite*\n")

    print(f"  Saved: {report_path}")

    # Also save raw results as JSON
    json_path = output_dir / 'benchmark_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {json_path}")

# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='LABS Complete Benchmark Suite')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmarks')
    parser.add_argument('--max-n', type=int, default=25, help='Maximum N to test')
    args = parser.parse_args()

    output_dir = Path(__file__).parent.parent / 'results'

    # Run benchmarks
    results = run_complete_benchmark(quick=args.quick, max_n=args.max_n)

    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    generate_plots(results, output_dir)

    # Generate report
    print("\n" + "=" * 80)
    print("GENERATING REPORT")
    print("=" * 80)
    generate_report(results, output_dir)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nFiles generated:")
    for f in sorted(output_dir.glob('*')):
        print(f"  - {f.name}")

if __name__ == "__main__":
    main()
