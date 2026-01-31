"""
Tests for LABS energy calculation.

The LABS energy is defined as E(S) = sum_{k=1}^{N-1} C_k^2
where C_k = sum_{i=0}^{N-1-k} S_i * S_{i+k} is the autocorrelation at lag k.
"""

import pytest
from hypothesis import given, strategies as st
import numpy as np

# -----------------------------------------------------------------------------
# Reference implementation for testing
# -----------------------------------------------------------------------------

def compute_autocorrelation(seq: list[int], k: int) -> int:
    """Compute autocorrelation C_k for lag k."""
    n = len(seq)
    return sum(seq[i] * seq[i + k] for i in range(n - k))


def compute_labs_energy(seq: list[int]) -> float:
    """
    Compute LABS energy: E(S) = sum_{k=1}^{N-1} C_k^2
    """
    n = len(seq)
    energy = 0.0
    for k in range(1, n):
        c_k = compute_autocorrelation(seq, k)
        energy += c_k * c_k
    return energy


def compute_labs_energy_vectorized(seq: np.ndarray) -> float:
    """Vectorized energy computation for performance comparison."""
    n = len(seq)
    energy = 0.0
    for k in range(1, n):
        c_k = np.dot(seq[:-k], seq[k:])
        energy += c_k * c_k
    return energy


# -----------------------------------------------------------------------------
# Unit Tests: Basic Energy Calculation
# -----------------------------------------------------------------------------

class TestEnergyBasic:
    """Basic unit tests for energy calculation."""

    def test_energy_simple_n3(self):
        """Test energy for N=3 sequence."""
        # S = [1, 1, -1]
        # C_1 = 1*1 + 1*(-1) = 0
        # C_2 = 1*(-1) = -1
        # E = 0^2 + (-1)^2 = 1
        seq = [1, 1, -1]
        assert compute_labs_energy(seq) == 1.0

    def test_energy_all_ones(self):
        """All +1 sequence has maximum autocorrelation."""
        seq = [1, 1, 1, 1]
        # C_1 = 3, C_2 = 2, C_3 = 1
        # E = 9 + 4 + 1 = 14
        assert compute_labs_energy(seq) == 14.0

    def test_energy_alternating(self):
        """Alternating sequence [1, -1, 1, -1, ...]."""
        seq = [1, -1, 1, -1]
        # C_1 = -1 -1 -1 = -3
        # C_2 = 1 + 1 = 2
        # C_3 = -1
        # E = 9 + 4 + 1 = 14
        energy = compute_labs_energy(seq)
        assert energy == 14.0

    def test_energy_single_element(self):
        """Single element has no lags, so energy = 0."""
        seq = [1]
        assert compute_labs_energy(seq) == 0.0

    def test_energy_two_elements(self):
        """Two elements, one lag."""
        seq = [1, 1]
        # C_1 = 1*1 = 1, E = 1
        assert compute_labs_energy(seq) == 1.0

        seq = [1, -1]
        # C_1 = 1*(-1) = -1, E = 1
        assert compute_labs_energy(seq) == 1.0


# -----------------------------------------------------------------------------
# Property Tests: Physics Invariants
# -----------------------------------------------------------------------------

class TestEnergySymmetries:
    """Tests for physical symmetries of LABS energy."""

    @given(st.lists(st.sampled_from([-1, 1]), min_size=3, max_size=25))
    def test_negation_symmetry(self, seq):
        """E(S) must equal E(-S)."""
        negated = [-s for s in seq]
        e_original = compute_labs_energy(seq)
        e_negated = compute_labs_energy(negated)
        assert e_original == e_negated, f"Negation symmetry failed: E({seq})={e_original} != E({negated})={e_negated}"

    @given(st.lists(st.sampled_from([-1, 1]), min_size=3, max_size=25))
    def test_reversal_symmetry(self, seq):
        """E(S) must equal E(reverse(S))."""
        reversed_seq = seq[::-1]
        e_original = compute_labs_energy(seq)
        e_reversed = compute_labs_energy(reversed_seq)
        assert e_original == e_reversed, f"Reversal symmetry failed: E({seq})={e_original} != E({reversed_seq})={e_reversed}"

    @given(st.lists(st.sampled_from([-1, 1]), min_size=3, max_size=25))
    def test_energy_non_negative(self, seq):
        """Energy must always be >= 0 (sum of squares)."""
        energy = compute_labs_energy(seq)
        assert energy >= 0, f"Negative energy: E({seq}) = {energy}"

    @given(st.lists(st.sampled_from([-1, 1]), min_size=3, max_size=25))
    def test_energy_is_integer(self, seq):
        """Energy must be an integer (sum of integer squares)."""
        energy = compute_labs_energy(seq)
        assert energy == int(energy), f"Non-integer energy: E({seq}) = {energy}"


# -----------------------------------------------------------------------------
# Ground Truth Tests
# -----------------------------------------------------------------------------

class TestKnownOptima:
    """Tests against known optimal LABS solutions."""

    def test_known_optima(self, known_optima):
        """Verify energy calculation matches known optimal values."""
        for n, (expected_energy, optimal_seq) in known_optima.items():
            computed_energy = compute_labs_energy(optimal_seq)
            assert computed_energy == expected_energy, \
                f"N={n}: computed {computed_energy}, expected {expected_energy}"

    def test_n3_optimal(self):
        """N=3 optimal energy is 1."""
        # All optimal sequences for N=3
        optimal_seqs = [
            [1, 1, -1],
            [-1, -1, 1],  # negation
            [-1, 1, 1],   # reversal
            [1, -1, -1],  # negation of reversal
        ]
        for seq in optimal_seqs:
            assert compute_labs_energy(seq) == 1

    def test_n5_optimal(self):
        """N=5 optimal energy is 2."""
        seq = [1, 1, 1, -1, 1]
        assert compute_labs_energy(seq) == 2

    def test_n7_optimal(self):
        """N=7 optimal energy is 4."""
        seq = [1, 1, 1, -1, -1, 1, -1]
        assert compute_labs_energy(seq) == 4


# -----------------------------------------------------------------------------
# Autocorrelation Tests
# -----------------------------------------------------------------------------

class TestAutocorrelation:
    """Tests for autocorrelation calculation."""

    def test_autocorrelation_basic(self):
        """Basic autocorrelation tests."""
        seq = [1, 1, 1, 1]
        assert compute_autocorrelation(seq, 1) == 3
        assert compute_autocorrelation(seq, 2) == 2
        assert compute_autocorrelation(seq, 3) == 1

    def test_autocorrelation_alternating(self):
        """Autocorrelation for alternating sequence."""
        seq = [1, -1, 1, -1]
        assert compute_autocorrelation(seq, 1) == -3
        assert compute_autocorrelation(seq, 2) == 2
        assert compute_autocorrelation(seq, 3) == -1

    @given(st.lists(st.sampled_from([-1, 1]), min_size=3, max_size=20))
    def test_autocorrelation_bounds(self, seq):
        """C_k must be in range [-(N-k), N-k]."""
        n = len(seq)
        for k in range(1, n):
            c_k = compute_autocorrelation(seq, k)
            max_val = n - k
            assert -max_val <= c_k <= max_val, \
                f"C_{k} = {c_k} out of bounds [-{max_val}, {max_val}]"


# -----------------------------------------------------------------------------
# Implementation Consistency Tests
# -----------------------------------------------------------------------------

class TestImplementationConsistency:
    """Tests comparing different implementations."""

    @given(st.lists(st.sampled_from([-1, 1]), min_size=3, max_size=25))
    def test_list_vs_numpy(self, seq):
        """List and NumPy implementations should match."""
        seq_np = np.array(seq)
        e_list = compute_labs_energy(seq)
        e_numpy = compute_labs_energy_vectorized(seq_np)
        assert abs(e_list - e_numpy) < 1e-10, \
            f"Implementation mismatch: list={e_list}, numpy={e_numpy}"


# -----------------------------------------------------------------------------
# Performance Baseline Tests
# -----------------------------------------------------------------------------

@pytest.mark.slow
class TestEnergyPerformance:
    """Performance tests for energy calculation."""

    def test_energy_n30(self):
        """Energy calculation completes for N=30."""
        np.random.seed(42)
        seq = [1 if x > 0.5 else -1 for x in np.random.random(30)]
        energy = compute_labs_energy(seq)
        assert energy >= 0

    def test_energy_n50(self):
        """Energy calculation completes for N=50."""
        np.random.seed(42)
        seq = [1 if x > 0.5 else -1 for x in np.random.random(50)]
        energy = compute_labs_energy(seq)
        assert energy >= 0
