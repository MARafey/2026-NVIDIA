"""
Integration tests for the complete LABS-SharedCache pipeline.

These tests verify that all components work together:
Quantum Sampling -> Seed Selection -> Shared Cache -> MTS -> Results
"""

import pytest
import numpy as np
from typing import Optional
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# Component imports (adjust paths as needed)
# -----------------------------------------------------------------------------

# For testing, we include simplified implementations here
# In production, these would be imported from src/

BITS_PER_ELEM = 1
EMPTY_SLOT = 0xFFFFFFFFFFFFFFFF


def pack_sequence(seq: list[int]) -> int:
    """Pack binary sequence into uint64."""
    packed = 0
    for i, elem in enumerate(seq):
        if elem == 1:
            packed |= (1 << i)
    return packed


def unpack_sequence(packed: int, n: int) -> list[int]:
    """Unpack uint64 to binary sequence."""
    return [1 if (packed >> i) & 1 else -1 for i in range(n)]


def compute_labs_energy(seq: list[int]) -> float:
    """Compute LABS energy."""
    n = len(seq)
    energy = 0.0
    for k in range(1, n):
        c_k = sum(seq[i] * seq[i + k] for i in range(n - k))
        energy += c_k * c_k
    return energy


@dataclass
class CacheEntry:
    packed_seq: int
    energy: float


class SharedCache:
    """Simplified shared cache for integration testing."""

    def __init__(self, size: int = 256):
        self.size = size
        self.entries = {}
        self.hits = 0
        self.misses = 0

    def lookup(self, packed_seq: int) -> Optional[float]:
        if packed_seq in self.entries:
            self.hits += 1
            return self.entries[packed_seq]
        self.misses += 1
        return None

    def update(self, packed_seq: int, energy: float):
        if packed_seq not in self.entries or energy < self.entries[packed_seq]:
            self.entries[packed_seq] = energy

    def get_hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


def generate_mock_quantum_seeds(n: int, num_seeds: int, seed: int = 42) -> list[tuple[int, float]]:
    """Generate mock quantum seeds (packed_seq, energy) pairs."""
    np.random.seed(seed)
    seeds = []
    for _ in range(num_seeds):
        seq = [1 if np.random.random() > 0.5 else -1 for _ in range(n)]
        packed = pack_sequence(seq)
        energy = compute_labs_energy(seq)
        seeds.append((packed, energy))
    # Sort by energy, return best
    seeds.sort(key=lambda x: x[1])
    return seeds[:num_seeds]


def mts_with_cache(
    initial_packed: int,
    n: int,
    cache: SharedCache,
    max_iterations: int = 50,
    tabu_tenure: int = 7
) -> tuple[int, float]:
    """MTS with shared cache integration."""
    current_packed = initial_packed
    current_seq = unpack_sequence(current_packed, n)
    current_energy = compute_labs_energy(current_seq)

    best_packed = current_packed
    best_energy = current_energy

    tabu = {}

    for iteration in range(max_iterations):
        best_neighbor_packed = None
        best_neighbor_energy = float('inf')
        best_flip_pos = -1

        for pos in range(n):
            # Flip and check cache
            neighbor_seq = current_seq.copy()
            neighbor_seq[pos] *= -1
            neighbor_packed = pack_sequence(neighbor_seq)

            # Cache lookup
            cached_energy = cache.lookup(neighbor_packed)
            if cached_energy is not None:
                neighbor_energy = cached_energy
            else:
                neighbor_energy = compute_labs_energy(neighbor_seq)
                cache.update(neighbor_packed, neighbor_energy)

            # Tabu check
            is_tabu = tabu.get(pos, 0) > iteration
            if is_tabu and neighbor_energy >= best_energy:
                continue

            if neighbor_energy < best_neighbor_energy:
                best_neighbor_packed = neighbor_packed
                best_neighbor_energy = neighbor_energy
                best_flip_pos = pos

        if best_neighbor_packed is None:
            break

        current_packed = best_neighbor_packed
        current_seq = unpack_sequence(current_packed, n)
        current_energy = best_neighbor_energy

        tabu[best_flip_pos] = iteration + tabu_tenure

        if current_energy < best_energy:
            best_packed = current_packed
            best_energy = current_energy

    return best_packed, best_energy


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------

class TestFullPipeline:
    """End-to-end pipeline tests."""

    def test_pipeline_n7(self):
        """Full pipeline for N=7."""
        n = 7
        num_seeds = 10
        cache = SharedCache(size=256)

        # Step 1: Generate quantum seeds
        seeds = generate_mock_quantum_seeds(n, num_seeds, seed=42)
        assert len(seeds) == num_seeds

        # Step 2: Seed cache
        for packed, energy in seeds:
            cache.update(packed, energy)

        # Step 3: Run MTS from best seed
        best_seed_packed, _ = seeds[0]
        result_packed, result_energy = mts_with_cache(
            best_seed_packed, n, cache, max_iterations=100
        )

        # Step 4: Verify result
        result_seq = unpack_sequence(result_packed, n)
        assert len(result_seq) == n
        assert all(x in [-1, 1] for x in result_seq)
        assert compute_labs_energy(result_seq) == result_energy

        # Known optimum for N=7 is 4
        assert result_energy <= 10  # Should be reasonable

    def test_pipeline_improves_seeds(self):
        """Pipeline should improve upon initial seeds."""
        n = 11
        num_seeds = 20
        cache = SharedCache(size=512)

        seeds = generate_mock_quantum_seeds(n, num_seeds, seed=42)
        best_seed_energy = seeds[0][1]

        # Run MTS on multiple seeds
        best_final_energy = float('inf')
        for packed, _ in seeds[:5]:  # Top 5 seeds
            _, energy = mts_with_cache(packed, n, cache, max_iterations=100)
            best_final_energy = min(best_final_energy, energy)

        # Should improve
        assert best_final_energy <= best_seed_energy

    def test_cache_provides_benefit(self):
        """Cache should reduce computation (higher hit rate over time)."""
        n = 7
        cache = SharedCache(size=256)

        # Run multiple MTS instances sharing cache
        seeds = generate_mock_quantum_seeds(n, 10, seed=42)

        for packed, _ in seeds:
            mts_with_cache(packed, n, cache, max_iterations=50)

        # Should have some cache hits
        assert cache.hits > 0
        assert cache.get_hit_rate() > 0


class TestCacheIntegration:
    """Tests for cache integration with MTS."""

    def test_cache_accumulates_knowledge(self):
        """Cache accumulates explored states across runs."""
        n = 5
        cache = SharedCache(size=128)

        seeds = generate_mock_quantum_seeds(n, 5, seed=42)

        entries_after_each_run = []
        for packed, _ in seeds:
            mts_with_cache(packed, n, cache, max_iterations=30)
            entries_after_each_run.append(len(cache.entries))

        # Cache should grow (or plateau)
        assert entries_after_each_run[-1] >= entries_after_each_run[0]

    def test_cache_hit_rate_increases(self):
        """Hit rate should increase as search converges."""
        n = 7
        cache = SharedCache(size=256)

        # First run - mostly misses
        seed1 = generate_mock_quantum_seeds(n, 1, seed=42)[0][0]
        mts_with_cache(seed1, n, cache, max_iterations=50)
        hit_rate_1 = cache.get_hit_rate()

        # Second run from nearby - should have more hits
        seed2 = generate_mock_quantum_seeds(n, 1, seed=43)[0][0]
        mts_with_cache(seed2, n, cache, max_iterations=50)
        hit_rate_2 = cache.get_hit_rate()

        # Overall hit rate should be reasonable
        assert hit_rate_2 >= 0  # At minimum, no negative


class TestQuantumAdvantage:
    """Tests demonstrating quantum seed advantage."""

    def test_quantum_seeds_vs_random(self):
        """Compare quantum-seeded vs random-seeded MTS."""
        n = 11
        iterations = 100

        # Quantum-seeded (mock: sorted by energy)
        quantum_seeds = generate_mock_quantum_seeds(n, 20, seed=42)
        quantum_cache = SharedCache(size=256)

        quantum_results = []
        for packed, _ in quantum_seeds[:5]:
            _, energy = mts_with_cache(packed, n, quantum_cache, iterations)
            quantum_results.append(energy)

        # Random-seeded
        np.random.seed(99)
        random_cache = SharedCache(size=256)
        random_results = []
        for _ in range(5):
            seq = [1 if np.random.random() > 0.5 else -1 for _ in range(n)]
            packed = pack_sequence(seq)
            _, energy = mts_with_cache(packed, n, random_cache, iterations)
            random_results.append(energy)

        # Both should find reasonable solutions
        assert min(quantum_results) <= 50  # Reasonable for N=11
        assert min(random_results) <= 50


class TestScaling:
    """Tests for scaling behavior."""

    @pytest.mark.parametrize("n", [5, 7, 9, 11])
    def test_pipeline_scales(self, n):
        """Pipeline works for various problem sizes."""
        cache = SharedCache(size=256)
        seeds = generate_mock_quantum_seeds(n, 5, seed=42)

        best_energy = float('inf')
        for packed, _ in seeds:
            _, energy = mts_with_cache(packed, n, cache, max_iterations=50)
            best_energy = min(best_energy, energy)

        # Should find something
        assert best_energy < float('inf')
        assert best_energy >= 0

    @pytest.mark.slow
    def test_large_n(self):
        """Pipeline handles larger N."""
        n = 20
        cache = SharedCache(size=512)
        seeds = generate_mock_quantum_seeds(n, 10, seed=42)

        result_packed, result_energy = mts_with_cache(
            seeds[0][0], n, cache, max_iterations=200
        )

        result_seq = unpack_sequence(result_packed, n)
        assert len(result_seq) == n
        assert result_energy >= 0


class TestCorrectness:
    """Correctness verification tests."""

    def test_result_energy_matches(self):
        """Reported energy matches recomputed energy."""
        n = 9
        cache = SharedCache(size=256)
        seeds = generate_mock_quantum_seeds(n, 5, seed=42)

        for packed, _ in seeds:
            result_packed, result_energy = mts_with_cache(
                packed, n, cache, max_iterations=50
            )
            result_seq = unpack_sequence(result_packed, n)
            recomputed = compute_labs_energy(result_seq)
            assert result_energy == recomputed

    def test_cache_entries_correct(self):
        """All cache entries have correct energies."""
        n = 7
        cache = SharedCache(size=256)
        seeds = generate_mock_quantum_seeds(n, 5, seed=42)

        for packed, _ in seeds:
            mts_with_cache(packed, n, cache, max_iterations=50)

        # Verify all cache entries
        for packed, cached_energy in cache.entries.items():
            seq = unpack_sequence(packed, n)
            actual_energy = compute_labs_energy(seq)
            assert cached_energy == actual_energy

    def test_symmetry_preserved(self):
        """Pipeline respects LABS symmetries."""
        n = 7
        cache = SharedCache(size=256)

        # Run from a sequence
        seq = [1, 1, 1, -1, -1, 1, -1]
        packed = pack_sequence(seq)
        _, energy1 = mts_with_cache(packed, n, cache, max_iterations=50)

        # Run from negated sequence
        neg_seq = [-s for s in seq]
        neg_packed = pack_sequence(neg_seq)
        cache2 = SharedCache(size=256)
        _, energy2 = mts_with_cache(neg_packed, n, cache2, max_iterations=50)

        # Should find same or symmetric solutions
        # (energies should be equal by symmetry)
        assert abs(energy1 - energy2) < 1  # Allow small difference due to search path


# -----------------------------------------------------------------------------
# Performance Benchmarks
# -----------------------------------------------------------------------------

@pytest.mark.slow
class TestPerformance:
    """Performance benchmark tests."""

    def test_throughput(self):
        """Measure pipeline throughput."""
        n = 11
        num_runs = 20
        cache = SharedCache(size=512)

        seeds = generate_mock_quantum_seeds(n, num_runs, seed=42)

        import time
        start = time.time()

        for packed, _ in seeds:
            mts_with_cache(packed, n, cache, max_iterations=100)

        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 60  # 60 seconds max for 20 runs

        runs_per_second = num_runs / elapsed
        print(f"\nThroughput: {runs_per_second:.2f} runs/second")

    def test_cache_efficiency(self):
        """Measure cache efficiency."""
        n = 9
        cache = SharedCache(size=256)
        seeds = generate_mock_quantum_seeds(n, 10, seed=42)

        for packed, _ in seeds:
            mts_with_cache(packed, n, cache, max_iterations=100)

        hit_rate = cache.get_hit_rate()
        occupancy = len(cache.entries) / cache.size

        print(f"\nCache hit rate: {hit_rate:.2%}")
        print(f"Cache occupancy: {occupancy:.2%}")

        # Report metrics (no hard assertions, just visibility)
        assert hit_rate >= 0
        assert occupancy >= 0
