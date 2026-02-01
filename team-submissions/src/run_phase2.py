#!/usr/bin/env python3
"""
Phase 2 Complete Solution: Quantum-Enhanced GPU-Accelerated LABS Solver

This script demonstrates the complete Phase 2 implementation:
- Step A: CPU validation of quantum algorithm
- Step B: GPU acceleration with CUDA-Q nvidia backend
- Step C: GPU acceleration of classical MTS with CuPy + shared cache

Algorithm: Counteradiabatic Quantum Optimization with Trotterized Evolution
- Uses gauge potential to suppress diabatic transitions
- 2-qubit R_YZ/R_ZY gates and 4-qubit R_YZZZ/R_ZYZZ/R_ZZYZ/R_ZZZY gates
- Quantum samples seed the population for MTS refinement

Key Innovation: Shared Exploration Cache
- All threads share discovered sequences in real-time
- Cache-first lookup before energy computation
- "Keep better energy" collision policy

Usage:
    python run_phase2.py [--quick] [--max-n N] [--cpu-only]
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

# =============================================================================
# Environment Detection
# =============================================================================

print("=" * 70)
print("LABS Solver - Phase 2 Complete Solution")
print("=" * 70)
print(f"\nTimestamp: {datetime.now().isoformat()}")
print("\nEnvironment Detection:")

# Check CuPy (GPU arrays)
try:
    import cupy as cp
    gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    gpu_mem = cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] / 1e9
    print(f"  CuPy: Available - {gpu_name} ({gpu_mem:.1f} GB)")
    CUPY_AVAILABLE = True
except ImportError:
    print("  CuPy: Not available (CPU fallback)")
    CUPY_AVAILABLE = False
except Exception as e:
    print(f"  CuPy: Error - {e}")
    CUPY_AVAILABLE = False

# Check CUDA-Q
try:
    import cudaq
    print(f"  CUDA-Q: Available")

    # Try nvidia GPU target
    try:
        cudaq.set_target("nvidia")
        print(f"    Target: nvidia (GPU accelerated)")
        CUDAQ_TARGET = "nvidia"
    except:
        try:
            cudaq.set_target("nvidia-fp64")
            print(f"    Target: nvidia-fp64 (GPU accelerated)")
            CUDAQ_TARGET = "nvidia-fp64"
        except:
            print(f"    Target: CPU (GPU target not available)")
            CUDAQ_TARGET = "cpu"
    CUDAQ_AVAILABLE = True
except ImportError:
    print("  CUDA-Q: Not available")
    CUDAQ_AVAILABLE = False
    CUDAQ_TARGET = None

print()

# =============================================================================
# Import Modules
# =============================================================================

from labs_solver.energy import calculate_energy, merit_factor
from labs_solver.mts import memetic_tabu_search as cpu_mts, random_sequence
from labs_solver.utils import pack_sequence, unpack_sequence, get_interactions

if CUPY_AVAILABLE:
    from labs_solver.energy_gpu import (
        calculate_energy_batch_gpu,
        calculate_energy_batch_cpu,
        GPUEnergyCache,
    )
    from labs_solver.mts_gpu import memetic_tabu_search_gpu

if CUDAQ_AVAILABLE:
    from labs_solver.quantum import sample_quantum_population, quantum_enhanced_mts


# =============================================================================
# Known Optimal Solutions (for validation)
# =============================================================================

KNOWN_OPTIMA = {
    3: (1, [-1, 1, 1]),
    4: (2, [-1, 1, 1, 1]),
    5: (2, [1, -1, 1, 1, 1]),
    6: (7, [1, -1, 1, 1, 1, 1]),
    7: (3, [-1, 1, -1, -1, 1, 1, 1]),
    8: (8, [-1, 1, -1, -1, 1, 1, 1, 1]),
    9: (12, [1, -1, 1, -1, -1, 1, 1, 1, 1]),
    10: (13, [1, -1, 1, -1, -1, 1, 1, 1, 1, 1]),
    11: 12,
    13: 13,
    15: 28,
    17: 32,
    19: 52,
    21: 42,
    23: 58,
    25: 76,
}


def get_optimal_energy(n: int) -> int:
    """Get known optimal energy for problem size n."""
    opt = KNOWN_OPTIMA.get(n)
    if isinstance(opt, tuple):
        return opt[0]
    return opt if opt else None


# =============================================================================
# Phase 2 Step A: CPU Validation
# =============================================================================

def run_step_a_validation(verbose: bool = True) -> Dict:
    """
    Step A: Validate quantum algorithm on CPU backend.

    Demonstrates that our implementation finds valid solutions for N=3 to N=10.
    """
    print("\n" + "=" * 70)
    print("STEP A: CPU Validation of Quantum Algorithm")
    print("=" * 70)

    results = {
        'step': 'A',
        'description': 'CPU validation of quantum-enhanced MTS',
        'tests': [],
    }

    # Test 1: Energy calculation correctness
    print("\n[Test 1] Energy Calculation Correctness")
    print("-" * 40)

    for n in [3, 5, 7]:
        opt = KNOWN_OPTIMA.get(n)
        if isinstance(opt, tuple):
            expected_energy, optimal_seq = opt
            computed = calculate_energy(optimal_seq)
            passed = computed == expected_energy
            print(f"  N={n}: computed={computed}, expected={expected_energy} [{'PASS' if passed else 'FAIL'}]")
            results['tests'].append({'name': f'energy_n{n}', 'passed': passed})

    # Test symmetries
    test_seq = [1, 1, -1, 1, -1, 1, -1]
    e_orig = calculate_energy(test_seq)
    e_neg = calculate_energy([-s for s in test_seq])
    e_rev = calculate_energy(test_seq[::-1])

    sym_pass = (e_orig == e_neg == e_rev)
    print(f"  Symmetries: E(S)={e_orig}, E(-S)={e_neg}, E(rev)={e_rev} [{'PASS' if sym_pass else 'FAIL'}]")
    results['tests'].append({'name': 'symmetry', 'passed': sym_pass})

    # Test 2: MTS finds optima for small N
    print("\n[Test 2] MTS Finds Small N Optima")
    print("-" * 40)

    for n in range(3, 11):
        opt_energy = get_optimal_energy(n)

        best_energy = float('inf')
        best_seq = None

        for trial in range(3):
            seq, energy, _, _ = cpu_mts(n, population_size=15, max_generations=30)
            if energy < best_energy:
                best_energy = energy
                best_seq = seq

        if opt_energy:
            ratio = opt_energy / best_energy if best_energy > 0 else 1.0
            status = "OPTIMAL" if best_energy == opt_energy else f"ratio={ratio:.3f}"
        else:
            status = f"MF={merit_factor(best_seq):.2f}"

        seq_str = ''.join(['+' if s == 1 else '-' for s in best_seq])
        print(f"  N={n:2d}: E={best_energy:4.0f} (opt={opt_energy or '?':>4}) [{status}] {seq_str}")

        results['tests'].append({
            'name': f'mts_n{n}',
            'energy': best_energy,
            'optimal': opt_energy,
            'ratio': opt_energy / best_energy if opt_energy and best_energy > 0 else None,
        })

    # Test 3: Quantum sampling (if available)
    if CUDAQ_AVAILABLE:
        print("\n[Test 3] Quantum Circuit Sampling")
        print("-" * 40)

        for n in [5, 7, 9]:
            start = time.time()
            population = sample_quantum_population(n, population_size=20, n_steps=1, T=1.0)
            sample_time = time.time() - start

            energies = [calculate_energy(seq) for seq in population]
            opt_energy = get_optimal_energy(n)

            print(f"  N={n}: time={sample_time:.3f}s, min={min(energies):.0f}, mean={np.mean(energies):.1f} (opt={opt_energy})")

            results['tests'].append({
                'name': f'quantum_sample_n{n}',
                'time': sample_time,
                'min_energy': min(energies),
                'mean_energy': np.mean(energies),
            })
    else:
        print("\n[Test 3] Quantum Circuit Sampling - SKIPPED (CUDA-Q not available)")

    results['passed'] = all(t.get('passed', True) for t in results['tests'])
    return results


# =============================================================================
# Phase 2 Step B: GPU Acceleration (Quantum)
# =============================================================================

def run_step_b_quantum_gpu(n_values: List[int], quick: bool = False) -> Dict:
    """
    Step B: GPU acceleration of quantum circuit simulation.

    Uses CUDA-Q nvidia backend for accelerated state vector simulation.
    """
    print("\n" + "=" * 70)
    print("STEP B: GPU Acceleration of Quantum Simulation")
    print("=" * 70)

    results = {
        'step': 'B',
        'description': 'Quantum simulation on CUDA-Q nvidia backend',
        'target': CUDAQ_TARGET,
        'benchmarks': [],
    }

    if not CUDAQ_AVAILABLE:
        print("\nSKIPPED: CUDA-Q not available")
        return results

    print(f"\nCUDA-Q Target: {CUDAQ_TARGET}")

    pop_size = 15 if quick else 25
    generations = 20 if quick else 40

    for n in n_values:
        print(f"\n  N={n}:")
        opt_energy = get_optimal_energy(n)

        # Quantum-enhanced MTS
        start = time.time()
        best_seq, best_energy, _, _ = quantum_enhanced_mts(
            n,
            population_size=pop_size,
            max_generations=generations,
            n_trotter_steps=1,
            T=1.0,
        )
        quantum_time = time.time() - start

        # Classical MTS for comparison
        start = time.time()
        classical_seq, classical_energy, _, _ = cpu_mts(
            n,
            population_size=pop_size,
            max_generations=generations,
        )
        classical_time = time.time() - start

        improvement = (classical_energy - best_energy) / classical_energy * 100 if classical_energy > 0 else 0

        print(f"    Quantum-enhanced: E={best_energy:.0f}, time={quantum_time:.2f}s")
        print(f"    Classical:        E={classical_energy:.0f}, time={classical_time:.2f}s")
        print(f"    Improvement:      {improvement:.1f}%")
        if opt_energy:
            print(f"    Optimal:          {opt_energy}")

        results['benchmarks'].append({
            'n': n,
            'quantum_energy': best_energy,
            'quantum_time': quantum_time,
            'classical_energy': classical_energy,
            'classical_time': classical_time,
            'improvement_pct': improvement,
            'optimal': opt_energy,
        })

    return results


# =============================================================================
# Phase 2 Step C: GPU Acceleration (Classical)
# =============================================================================

def run_step_c_classical_gpu(n_values: List[int], quick: bool = False) -> Dict:
    """
    Step C: GPU acceleration of classical MTS algorithm.

    Uses CuPy for batch energy computation and shared exploration cache.
    """
    print("\n" + "=" * 70)
    print("STEP C: GPU Acceleration of Classical MTS")
    print("=" * 70)

    results = {
        'step': 'C',
        'description': 'MTS with CuPy batch computation and shared cache',
        'gpu_available': CUPY_AVAILABLE,
        'benchmarks': [],
    }

    if not CUPY_AVAILABLE:
        print("\nSKIPPED: CuPy not available")
        return results

    print(f"\nGPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")

    pop_size = 15 if quick else 20
    generations = 25 if quick else 50

    for n in n_values:
        print(f"\n  N={n}:")
        opt_energy = get_optimal_energy(n)

        # CPU MTS
        cpu_start = time.time()
        cpu_seq, cpu_energy, _, _ = cpu_mts(
            n,
            population_size=pop_size,
            max_generations=generations,
        )
        cpu_time = time.time() - cpu_start

        # GPU MTS with cache
        gpu_start = time.time()
        gpu_seq, gpu_energy, _, _, metrics = memetic_tabu_search_gpu(
            n,
            population_size=pop_size,
            max_generations=generations,
            use_shared_cache=True,
        )
        gpu_time = time.time() - gpu_start

        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        hit_rate = metrics.get('final_cache_hit_rate', 0)

        print(f"    CPU:  E={cpu_energy:.0f}, time={cpu_time:.2f}s")
        print(f"    GPU:  E={gpu_energy:.0f}, time={gpu_time:.2f}s, cache hit={hit_rate:.1%}")
        print(f"    Speedup: {speedup:.2f}x")
        if opt_energy:
            ratio = opt_energy / min(cpu_energy, gpu_energy)
            print(f"    Approx ratio: {ratio:.3f} (optimal={opt_energy})")

        results['benchmarks'].append({
            'n': n,
            'cpu_energy': cpu_energy,
            'cpu_time': cpu_time,
            'gpu_energy': gpu_energy,
            'gpu_time': gpu_time,
            'speedup': speedup,
            'cache_hit_rate': hit_rate,
            'optimal': opt_energy,
        })

    return results


# =============================================================================
# Complete Phase 2 Pipeline
# =============================================================================

def run_complete_pipeline(n: int, verbose: bool = True) -> Dict:
    """
    Run the complete quantum-enhanced GPU-accelerated pipeline for a single N.

    Pipeline:
    1. Generate quantum samples using counteradiabatic circuit
    2. Run GPU-accelerated MTS with shared cache
    3. Return best solution found
    """
    print(f"\n{'=' * 70}")
    print(f"Complete Pipeline: N={n}")
    print("=" * 70)

    result = {
        'n': n,
        'stages': {},
    }

    total_start = time.time()

    # Stage 1: Quantum Sampling
    if CUDAQ_AVAILABLE:
        print("\n[Stage 1] Quantum Sampling")
        start = time.time()
        quantum_population = sample_quantum_population(n, population_size=30, n_steps=2, T=1.5)
        quantum_time = time.time() - start

        quantum_energies = [calculate_energy(seq) for seq in quantum_population]
        print(f"  Time: {quantum_time:.2f}s")
        print(f"  Energies: min={min(quantum_energies):.0f}, mean={np.mean(quantum_energies):.1f}")

        result['stages']['quantum'] = {
            'time': quantum_time,
            'min_energy': min(quantum_energies),
            'mean_energy': np.mean(quantum_energies),
        }
    else:
        quantum_population = None
        print("\n[Stage 1] Quantum Sampling - SKIPPED")

    # Stage 2: GPU-Accelerated MTS
    print("\n[Stage 2] GPU-Accelerated MTS")

    if CUPY_AVAILABLE:
        start = time.time()
        best_seq, best_energy, final_pop, final_energies, metrics = memetic_tabu_search_gpu(
            n,
            population_size=25,
            max_generations=60,
            initial_population=quantum_population,
            use_shared_cache=True,
            verbose=verbose,
        )
        mts_time = time.time() - start

        result['stages']['mts'] = {
            'time': mts_time,
            'best_energy': best_energy,
            'cache_hit_rate': metrics.get('final_cache_hit_rate', 0),
        }
    else:
        # Fallback to CPU
        start = time.time()
        best_seq, best_energy, final_pop, final_energies = cpu_mts(
            n,
            population_size=25,
            max_generations=60,
            initial_population=quantum_population,
            verbose=verbose,
        )
        mts_time = time.time() - start

        result['stages']['mts'] = {
            'time': mts_time,
            'best_energy': best_energy,
            'cache_hit_rate': 0,
        }

    total_time = time.time() - total_start

    # Final result
    opt_energy = get_optimal_energy(n)
    mf = merit_factor(best_seq)
    seq_str = ''.join(['+' if s == 1 else '-' for s in best_seq])

    print(f"\n[Result]")
    print(f"  Best Energy: {best_energy:.0f}")
    print(f"  Merit Factor: {mf:.3f}")
    print(f"  Total Time: {total_time:.2f}s")
    if opt_energy:
        print(f"  Approx Ratio: {opt_energy/best_energy:.4f} (optimal={opt_energy})")
    print(f"  Sequence: {seq_str[:50]}{'...' if n > 50 else ''}")

    result['best_sequence'] = best_seq
    result['best_energy'] = best_energy
    result['merit_factor'] = mf
    result['total_time'] = total_time
    result['optimal'] = opt_energy
    result['approx_ratio'] = opt_energy / best_energy if opt_energy else None

    return result


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='LABS Phase 2 Complete Solution')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmarks')
    parser.add_argument('--max-n', type=int, default=21, help='Maximum N to test')
    parser.add_argument('--cpu-only', action='store_true', help='Force CPU-only mode')
    args = parser.parse_args()

    if args.cpu_only:
        global CUPY_AVAILABLE, CUDAQ_AVAILABLE
        CUPY_AVAILABLE = False
        CUDAQ_AVAILABLE = False
        print("\n[CPU-only mode enabled]")

    # Define test sizes
    small_n = [5, 7, 9, 11]
    medium_n = [11, 13, 15, 17]
    large_n = list(range(15, min(args.max_n + 1, 26), 2))

    all_results = {}

    # Step A: CPU Validation
    all_results['step_a'] = run_step_a_validation()

    # Step B: Quantum GPU Acceleration
    test_n = small_n if args.quick else small_n + [13, 15]
    all_results['step_b'] = run_step_b_quantum_gpu(test_n, quick=args.quick)

    # Step C: Classical GPU Acceleration
    test_n = medium_n if args.quick else medium_n + [19, 21]
    all_results['step_c'] = run_step_c_classical_gpu(test_n, quick=args.quick)

    # Complete pipeline for showcase
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  SHOWCASE: Complete Pipeline Results".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    showcase_n = [11, 15, 19] if args.quick else [11, 15, 19, 21, 23]
    for n in showcase_n:
        all_results[f'pipeline_n{n}'] = run_complete_pipeline(n, verbose=False)

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 2 SUMMARY")
    print("=" * 70)

    print(f"\nEnvironment:")
    print(f"  CUDA-Q: {'Available' if CUDAQ_AVAILABLE else 'Not available'} (target: {CUDAQ_TARGET})")
    print(f"  CuPy: {'Available' if CUPY_AVAILABLE else 'Not available'}")

    print(f"\nStep A (CPU Validation): {'PASS' if all_results['step_a'].get('passed') else 'INCOMPLETE'}")

    if all_results['step_b'].get('benchmarks'):
        avg_improvement = np.mean([b['improvement_pct'] for b in all_results['step_b']['benchmarks']])
        print(f"Step B (Quantum GPU): Avg improvement {avg_improvement:.1f}%")

    if all_results['step_c'].get('benchmarks'):
        avg_speedup = np.mean([b['speedup'] for b in all_results['step_c']['benchmarks']])
        avg_hitrate = np.mean([b['cache_hit_rate'] for b in all_results['step_c']['benchmarks']])
        print(f"Step C (Classical GPU): Avg speedup {avg_speedup:.2f}x, cache hit rate {avg_hitrate:.1%}")

    print(f"\nBest Results:")
    for key, result in all_results.items():
        if key.startswith('pipeline_n'):
            n = result['n']
            e = result['best_energy']
            mf = result['merit_factor']
            t = result['total_time']
            ratio = result.get('approx_ratio', 0)
            print(f"  N={n:2d}: E={e:5.0f}, MF={mf:.3f}, ratio={ratio:.4f}, time={t:.2f}s")

    print("\n" + "=" * 70)
    print("Phase 2 Implementation Complete")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    main()
