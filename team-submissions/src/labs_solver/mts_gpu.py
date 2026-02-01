"""
GPU-Accelerated Memetic Tabu Search (MTS) for LABS Problem

Implements GPU acceleration strategies from the PRD:
1. Batch neighbor evaluation on GPU
2. Shared exploration cache
3. Parallel population search

Reference: "Scaling advantage with quantum-enhanced memetic tabu search"
"""

import time
from typing import List, Tuple, Optional, Dict
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from .energy import calculate_energy
from .energy_gpu import (
    calculate_energy_batch_gpu,
    calculate_energy_batch_cpu,
    evaluate_neighbors_gpu,
    GPUEnergyCache,
    CUPY_AVAILABLE as GPU_AVAILABLE,
)


def random_sequence(n: int) -> List[int]:
    """Generate a random binary sequence."""
    return [1 if np.random.random() > 0.5 else -1 for _ in range(n)]


def tabu_search_gpu(
    sequence: np.ndarray,
    max_iterations: int = 100,
    tabu_tenure: int = 7,
    cache: Optional[GPUEnergyCache] = None,
) -> Tuple[np.ndarray, float, Dict]:
    """
    GPU-accelerated tabu search for a single sequence.

    Uses batch neighbor evaluation on GPU for speedup.

    Args:
        sequence: Initial sequence as numpy array
        max_iterations: Maximum iterations
        tabu_tenure: How long moves stay tabu
        cache: Optional shared cache for energy lookups

    Returns:
        Tuple of (best_sequence, best_energy, stats)
    """
    n = len(sequence)
    current = sequence.copy()

    # Compute initial energy
    if cache:
        found, energy = cache.lookup(current)
        if not found:
            energy = calculate_energy(current.tolist())
            cache.insert(current, energy)
        current_energy = energy
    else:
        current_energy = calculate_energy(current.tolist())

    best = current.copy()
    best_energy = current_energy

    # Tabu list: position -> iteration when it becomes non-tabu
    tabu_list = {}

    stats = {
        'iterations': 0,
        'cache_hits': 0,
        'cache_misses': 0,
        'improvements': 0,
    }

    for iteration in range(max_iterations):
        stats['iterations'] += 1

        # Generate all neighbors
        neighbors = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            neighbors[i] = current.copy()
            neighbors[i, i] *= -1

        # Batch evaluate energies on GPU
        if GPU_AVAILABLE:
            neighbor_energies = calculate_energy_batch_gpu(neighbors)
        else:
            neighbor_energies = calculate_energy_batch_cpu(neighbors)

        # Find best non-tabu move
        best_move = None
        best_move_energy = float('inf')

        for i in range(n):
            is_tabu = tabu_list.get(i, 0) > iteration
            aspiration = neighbor_energies[i] < best_energy

            if (not is_tabu or aspiration) and neighbor_energies[i] < best_move_energy:
                best_move = i
                best_move_energy = neighbor_energies[i]

        if best_move is None:
            break

        # Apply move
        current[best_move] *= -1
        current_energy = best_move_energy

        # Update cache
        if cache:
            cache.insert(current.copy(), current_energy)

        # Update tabu list
        tabu_list[best_move] = iteration + tabu_tenure

        # Update best
        if current_energy < best_energy:
            best = current.copy()
            best_energy = current_energy
            stats['improvements'] += 1

    if cache:
        stats['cache_hits'] = cache.hits
        stats['cache_misses'] = cache.misses

    return best, best_energy, stats


def parallel_tabu_search_gpu(
    population: np.ndarray,
    max_iterations: int = 100,
    tabu_tenure: int = 7,
    shared_cache: bool = True,
) -> Tuple[np.ndarray, float, List[Dict]]:
    """
    Parallel tabu search for entire population on GPU.

    All population members share an exploration cache, enabling
    collaborative search as described in the PRD.

    Args:
        population: 2D array of shape (pop_size, n)
        max_iterations: Max iterations per member
        tabu_tenure: Tabu tenure
        shared_cache: Whether to use shared exploration cache

    Returns:
        Tuple of (best_sequence, best_energy, stats_list)
    """
    pop_size, n = population.shape

    # Create shared cache
    cache = GPUEnergyCache(capacity=8192, sequence_length=n) if shared_cache else None

    best_overall = None
    best_energy_overall = float('inf')
    all_stats = []

    for i in range(pop_size):
        seq, energy, stats = tabu_search_gpu(
            population[i].copy(),
            max_iterations=max_iterations,
            tabu_tenure=tabu_tenure,
            cache=cache,
        )

        all_stats.append(stats)

        if energy < best_energy_overall:
            best_overall = seq
            best_energy_overall = energy

    # Add cache statistics to final stats
    if cache:
        all_stats[-1]['total_cache_hits'] = cache.hits
        all_stats[-1]['total_cache_misses'] = cache.misses
        all_stats[-1]['cache_hit_rate'] = cache.get_hit_rate()

    return best_overall, best_energy_overall, all_stats


def memetic_tabu_search_gpu(
    n: int,
    population_size: int = 20,
    max_generations: int = 50,
    p_mutate: float = 0.1,
    tabu_iterations: int = 100,
    tabu_tenure: int = 7,
    initial_population: Optional[List[List[int]]] = None,
    use_shared_cache: bool = True,
    verbose: bool = False,
) -> Tuple[List[int], float, List[List[int]], List[float], Dict]:
    """
    GPU-Accelerated Memetic Tabu Search.

    Combines evolutionary search with GPU-accelerated local search.

    Args:
        n: Sequence length
        population_size: Size of the population
        max_generations: Maximum generations
        p_mutate: Mutation probability
        tabu_iterations: Iterations per tabu search
        tabu_tenure: Tabu tenure
        initial_population: Optional pre-seeded population (e.g., from quantum)
        use_shared_cache: Whether to use shared exploration cache
        verbose: Print progress

    Returns:
        Tuple of (best_seq, best_energy, final_pop, final_energies, metrics)
    """
    start_time = time.time()

    # Initialize population
    if initial_population is not None:
        population = np.array([seq[:] for seq in initial_population[:population_size]], dtype=np.float32)
        while len(population) < population_size:
            population = np.vstack([population, np.array(random_sequence(n), dtype=np.float32)])
    else:
        population = np.array([random_sequence(n) for _ in range(population_size)], dtype=np.float32)

    # Batch compute initial energies
    if GPU_AVAILABLE:
        energies = calculate_energy_batch_gpu(population)
    else:
        energies = calculate_energy_batch_cpu(population)

    # Find initial best
    best_idx = np.argmin(energies)
    best_sequence = population[best_idx].copy()
    best_energy = energies[best_idx]

    # Create shared cache for this run
    cache = GPUEnergyCache(capacity=8192, sequence_length=n) if use_shared_cache else None

    # Seed cache with initial population
    if cache:
        for i in range(population_size):
            cache.insert(population[i], energies[i])

    metrics = {
        'generations': [],
        'best_energies': [best_energy],
        'mean_energies': [np.mean(energies)],
        'cache_hit_rates': [],
        'iteration_times': [],
    }

    if verbose:
        print(f"Initial: best={best_energy:.0f}, mean={np.mean(energies):.1f}")

    for generation in range(max_generations):
        gen_start = time.time()

        # Select parents (tournament selection)
        idx1, idx2 = np.random.choice(population_size, 2, replace=False)
        parent1 = population[idx1]
        parent2 = population[idx2]

        # Crossover (uniform)
        mask = np.random.random(n) > 0.5
        child = np.where(mask, parent1, parent2)

        # Mutation
        if np.random.random() < p_mutate:
            mut_mask = np.random.random(n) < 0.1
            child = np.where(mut_mask, -child, child)

        # Apply tabu search to child
        improved_child, child_energy, stats = tabu_search_gpu(
            child.astype(np.float32),
            max_iterations=tabu_iterations,
            tabu_tenure=tabu_tenure,
            cache=cache,
        )

        # Replace worst if child is better
        worst_idx = np.argmax(energies)
        if child_energy < energies[worst_idx]:
            population[worst_idx] = improved_child
            energies[worst_idx] = child_energy

        # Update global best
        if child_energy < best_energy:
            best_sequence = improved_child.copy()
            best_energy = child_energy
            if verbose:
                print(f"Gen {generation}: new best = {best_energy:.0f}")

        gen_time = time.time() - gen_start

        # Record metrics
        metrics['generations'].append(generation)
        metrics['best_energies'].append(best_energy)
        metrics['mean_energies'].append(np.mean(energies))
        metrics['iteration_times'].append(gen_time)
        if cache:
            metrics['cache_hit_rates'].append(cache.get_hit_rate())

    total_time = time.time() - start_time
    metrics['total_time'] = total_time
    metrics['final_cache_hit_rate'] = cache.get_hit_rate() if cache else 0.0

    if verbose:
        print(f"Completed in {total_time:.2f}s, final best = {best_energy:.0f}")
        if cache:
            print(f"Cache hit rate: {cache.get_hit_rate():.1%}")

    return (
        best_sequence.astype(int).tolist(),
        float(best_energy),
        [p.astype(int).tolist() for p in population],
        energies.tolist(),
        metrics,
    )


def benchmark_cpu_vs_gpu(n: int, population_size: int = 20, generations: int = 30) -> Dict:
    """
    Benchmark CPU vs GPU MTS performance.

    Args:
        n: Sequence length
        population_size: Population size
        generations: Number of generations

    Returns:
        Dictionary with benchmark results
    """
    from .mts import memetic_tabu_search as cpu_mts

    print(f"\nBenchmarking N={n}, pop={population_size}, gens={generations}")
    print("-" * 50)

    # CPU benchmark
    print("Running CPU MTS...")
    cpu_start = time.time()
    cpu_seq, cpu_energy, _, _ = cpu_mts(
        n,
        population_size=population_size,
        max_generations=generations,
        verbose=False,
    )
    cpu_time = time.time() - cpu_start

    # GPU benchmark (with cache)
    print("Running GPU MTS (with cache)...")
    gpu_start = time.time()
    gpu_seq, gpu_energy, _, _, gpu_metrics = memetic_tabu_search_gpu(
        n,
        population_size=population_size,
        max_generations=generations,
        use_shared_cache=True,
        verbose=False,
    )
    gpu_time = time.time() - gpu_start

    # GPU benchmark (without cache)
    print("Running GPU MTS (no cache)...")
    gpu_nc_start = time.time()
    gpu_nc_seq, gpu_nc_energy, _, _, _ = memetic_tabu_search_gpu(
        n,
        population_size=population_size,
        max_generations=generations,
        use_shared_cache=False,
        verbose=False,
    )
    gpu_nc_time = time.time() - gpu_nc_start

    results = {
        'n': n,
        'population_size': population_size,
        'generations': generations,
        'cpu_time': cpu_time,
        'cpu_energy': cpu_energy,
        'gpu_cache_time': gpu_time,
        'gpu_cache_energy': gpu_energy,
        'gpu_cache_hit_rate': gpu_metrics['final_cache_hit_rate'],
        'gpu_no_cache_time': gpu_nc_time,
        'gpu_no_cache_energy': gpu_nc_energy,
        'speedup_with_cache': cpu_time / gpu_time if gpu_time > 0 else float('inf'),
        'speedup_no_cache': cpu_time / gpu_nc_time if gpu_nc_time > 0 else float('inf'),
    }

    print(f"\nResults:")
    print(f"  CPU:              {cpu_time:.2f}s, E={cpu_energy:.0f}")
    print(f"  GPU (with cache): {gpu_time:.2f}s, E={gpu_energy:.0f}, hit rate={results['gpu_cache_hit_rate']:.1%}")
    print(f"  GPU (no cache):   {gpu_nc_time:.2f}s, E={gpu_nc_energy:.0f}")
    print(f"  Speedup:          {results['speedup_with_cache']:.2f}x (with cache)")

    return results
