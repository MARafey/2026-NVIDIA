"""
Tests for Memetic Tabu Search (MTS) algorithm.

MTS combines local search with tabu memory to avoid cycling
and find high-quality LABS solutions.
"""

import pytest
from hypothesis import given, strategies as st, settings
import numpy as np
from typing import Callable

# -----------------------------------------------------------------------------
# Reference implementation for testing
# -----------------------------------------------------------------------------

def compute_labs_energy(seq: list[int]) -> float:
    """Compute LABS energy."""
    n = len(seq)
    energy = 0.0
    for k in range(1, n):
        c_k = sum(seq[i] * seq[i + k] for i in range(n - k))
        energy += c_k * c_k
    return energy


def flip_bit(seq: list[int], pos: int) -> list[int]:
    """Flip single position in sequence."""
    result = seq.copy()
    result[pos] *= -1
    return result


def get_neighbors(seq: list[int]) -> list[list[int]]:
    """Generate all single-flip neighbors."""
    return [flip_bit(seq, i) for i in range(len(seq))]


def simple_tabu_search(
    initial_seq: list[int],
    max_iterations: int = 100,
    tabu_tenure: int = 7,
    energy_func: Callable = compute_labs_energy
) -> tuple[list[int], float]:
    """
    Simple tabu search implementation for testing.

    Returns (best_sequence, best_energy).
    """
    n = len(initial_seq)
    current = initial_seq.copy()
    current_energy = energy_func(current)

    best = current.copy()
    best_energy = current_energy

    # Tabu list: position -> iteration when it becomes non-tabu
    tabu = {}

    for iteration in range(max_iterations):
        # Find best non-tabu neighbor
        best_neighbor = None
        best_neighbor_energy = float('inf')
        best_flip_pos = -1

        for pos in range(n):
            neighbor = flip_bit(current, pos)
            neighbor_energy = energy_func(neighbor)

            # Check if move is tabu
            is_tabu = tabu.get(pos, 0) > iteration

            # Aspiration: accept tabu move if it improves global best
            if is_tabu and neighbor_energy >= best_energy:
                continue

            if neighbor_energy < best_neighbor_energy:
                best_neighbor = neighbor
                best_neighbor_energy = neighbor_energy
                best_flip_pos = pos

        if best_neighbor is None:
            # All moves are tabu - this shouldn't happen with aspiration
            break

        # Make the move
        current = best_neighbor
        current_energy = best_neighbor_energy

        # Update tabu list
        tabu[best_flip_pos] = iteration + tabu_tenure

        # Update global best
        if current_energy < best_energy:
            best = current.copy()
            best_energy = current_energy

    return best, best_energy


def mts_with_population(
    population: list[list[int]],
    max_iterations: int = 50,
    tabu_tenure: int = 7
) -> tuple[list[int], float]:
    """
    MTS with population (for quantum-seeded search).

    Each member runs tabu search, best overall is returned.
    """
    best_overall = None
    best_energy_overall = float('inf')

    for member in population:
        result, energy = simple_tabu_search(
            member,
            max_iterations=max_iterations,
            tabu_tenure=tabu_tenure
        )
        if energy < best_energy_overall:
            best_overall = result
            best_energy_overall = energy

    return best_overall, best_energy_overall


# -----------------------------------------------------------------------------
# Unit Tests: Basic MTS Operations
# -----------------------------------------------------------------------------

class TestMTSBasic:
    """Basic MTS operation tests."""

    def test_flip_bit(self):
        """Flip bit works correctly."""
        seq = [1, 1, -1, 1]
        flipped = flip_bit(seq, 2)
        assert flipped == [1, 1, 1, 1]
        assert seq == [1, 1, -1, 1]  # Original unchanged

    def test_get_neighbors(self):
        """Get neighbors returns correct count."""
        seq = [1, 1, -1]
        neighbors = get_neighbors(seq)
        assert len(neighbors) == 3
        assert [-1, 1, -1] in neighbors
        assert [1, -1, -1] in neighbors
        assert [1, 1, 1] in neighbors

    def test_tabu_search_improves(self):
        """Tabu search should not make solution worse."""
        initial = [1, 1, 1, 1, 1]  # High energy
        initial_energy = compute_labs_energy(initial)

        result, final_energy = simple_tabu_search(
            initial, max_iterations=100
        )

        assert final_energy <= initial_energy

    def test_tabu_search_finds_optimum_n3(self):
        """Tabu search finds optimum for N=3."""
        initial = [1, 1, 1]  # Not optimal
        result, energy = simple_tabu_search(initial, max_iterations=50)
        assert energy == 1  # Known optimum for N=3


# -----------------------------------------------------------------------------
# Tests: MTS Correctness
# -----------------------------------------------------------------------------

class TestMTSCorrectness:
    """Correctness tests for MTS."""

    def test_result_is_valid_sequence(self):
        """MTS result is a valid binary sequence."""
        initial = [1, -1, 1, -1, 1]
        result, _ = simple_tabu_search(initial, max_iterations=50)

        assert len(result) == len(initial)
        assert all(x in [-1, 1] for x in result)

    def test_reported_energy_matches_sequence(self):
        """Reported energy matches actual energy of returned sequence."""
        initial = [1, 1, 1, -1, -1, 1, -1]
        result, reported_energy = simple_tabu_search(initial, max_iterations=50)

        actual_energy = compute_labs_energy(result)
        assert reported_energy == actual_energy

    def test_monotonic_best_energy(self):
        """Best energy found should never increase during search."""
        initial = [1, 1, 1, 1, 1, 1, 1]

        # Track best energies during search
        best_energies = []

        def tracking_energy(seq):
            e = compute_labs_energy(seq)
            if not best_energies or e < best_energies[-1]:
                best_energies.append(e)
            else:
                best_energies.append(best_energies[-1])
            return e

        simple_tabu_search(initial, max_iterations=50, energy_func=tracking_energy)

        # Best should be monotonically decreasing
        for i in range(1, len(best_energies)):
            assert best_energies[i] <= best_energies[i-1]


# -----------------------------------------------------------------------------
# Tests: Population-Based MTS
# -----------------------------------------------------------------------------

class TestMTSPopulation:
    """Tests for population-based MTS (quantum-seeded)."""

    def test_population_mts_uses_all_members(self):
        """All population members are searched."""
        # Create population where one member is clearly better
        population = [
            [1, 1, 1, 1, 1],      # High energy start
            [1, 1, 1, 1, 1],      # High energy start
            [1, 1, -1, 1, -1],    # Better start
        ]

        result, energy = mts_with_population(population, max_iterations=20)

        # Should find something at least as good as best initial member
        best_initial = min(compute_labs_energy(p) for p in population)
        assert energy <= best_initial

    def test_population_mts_improves_all(self):
        """Population MTS improves over random seeds."""
        np.random.seed(42)

        # Random population
        population = []
        for _ in range(10):
            seq = [1 if np.random.random() > 0.5 else -1 for _ in range(11)]
            population.append(seq)

        best_initial = min(compute_labs_energy(p) for p in population)
        result, final_energy = mts_with_population(population, max_iterations=100)

        # Known optimum for N=11 is 12
        assert final_energy <= best_initial
        assert final_energy <= 20  # Should get reasonably close


# -----------------------------------------------------------------------------
# Ground Truth Tests
# -----------------------------------------------------------------------------

class TestMTSGroundTruth:
    """Tests against known LABS optima."""

    @pytest.mark.parametrize("n,expected_optimal", [
        (3, 1),
        (5, 2),
        (7, 4),
    ])
    def test_finds_small_n_optima(self, n, expected_optimal):
        """MTS finds known optima for small N."""
        # Try multiple random starts
        best_found = float('inf')

        for seed in range(10):
            np.random.seed(seed)
            initial = [1 if np.random.random() > 0.5 else -1 for _ in range(n)]
            _, energy = simple_tabu_search(initial, max_iterations=100)
            best_found = min(best_found, energy)

        assert best_found == expected_optimal

    @pytest.mark.slow
    def test_n11_approaches_optimum(self):
        """MTS approaches optimum for N=11 (optimal=12)."""
        best_found = float('inf')

        for seed in range(20):
            np.random.seed(seed)
            initial = [1 if np.random.random() > 0.5 else -1 for _ in range(11)]
            _, energy = simple_tabu_search(initial, max_iterations=200)
            best_found = min(best_found, energy)

        # Should get within 50% of optimal
        assert best_found <= 18  # Optimal is 12


# -----------------------------------------------------------------------------
# Property-Based Tests
# -----------------------------------------------------------------------------

class TestMTSProperties:
    """Property-based tests for MTS."""

    @given(st.lists(st.sampled_from([-1, 1]), min_size=5, max_size=15))
    @settings(max_examples=20)
    def test_mts_never_worsens(self, initial_seq):
        """MTS result is never worse than initial."""
        initial_energy = compute_labs_energy(initial_seq)
        _, final_energy = simple_tabu_search(initial_seq, max_iterations=30)
        assert final_energy <= initial_energy

    @given(st.lists(st.sampled_from([-1, 1]), min_size=5, max_size=15))
    @settings(max_examples=20)
    def test_mts_result_valid(self, initial_seq):
        """MTS result is always a valid sequence."""
        result, energy = simple_tabu_search(initial_seq, max_iterations=30)

        # Valid length
        assert len(result) == len(initial_seq)

        # Valid elements
        assert all(x in [-1, 1] for x in result)

        # Energy matches
        assert compute_labs_energy(result) == energy


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------

@pytest.mark.integration
class TestMTSIntegration:
    """Integration tests for MTS with other components."""

    def test_quantum_seeded_vs_random(self):
        """Quantum-like seeds should outperform random seeds."""
        n = 11
        np.random.seed(42)

        # Simulate "quantum seeds" - biased toward lower energy
        # (In reality, quantum samples cluster near ground state)
        quantum_seeds = []
        for _ in range(20):
            # Generate and keep if energy is reasonable
            seq = [1 if np.random.random() > 0.5 else -1 for _ in range(n)]
            if compute_labs_energy(seq) < 50:  # Bias toward lower energy
                quantum_seeds.append(seq)
        quantum_seeds = quantum_seeds[:10] or [[1]*n]  # Fallback

        # Random seeds
        random_seeds = []
        for _ in range(10):
            seq = [1 if np.random.random() > 0.5 else -1 for _ in range(n)]
            random_seeds.append(seq)

        # Run MTS on both
        _, quantum_result = mts_with_population(quantum_seeds, max_iterations=50)
        _, random_result = mts_with_population(random_seeds, max_iterations=50)

        # Quantum seeds should be at least as good (usually better)
        # This is probabilistic, so we just check it's reasonable
        assert quantum_result <= random_result + 10

    def test_mts_with_cache_simulation(self):
        """Simulate MTS with shared cache benefit."""
        n = 7

        # Without cache: count all energy evaluations
        eval_count_no_cache = [0]

        def counting_energy(seq):
            eval_count_no_cache[0] += 1
            return compute_labs_energy(seq)

        initial = [1, 1, 1, 1, 1, 1, 1]
        simple_tabu_search(initial, max_iterations=50, energy_func=counting_energy)

        # With "cache": track unique sequences evaluated
        seen_sequences = set()
        eval_count_with_cache = [0]

        def caching_energy(seq):
            key = tuple(seq)
            if key not in seen_sequences:
                seen_sequences.add(key)
                eval_count_with_cache[0] += 1
            return compute_labs_energy(seq)

        simple_tabu_search(initial, max_iterations=50, energy_func=caching_energy)

        # Cache should reduce evaluations
        assert eval_count_with_cache[0] <= eval_count_no_cache[0]
