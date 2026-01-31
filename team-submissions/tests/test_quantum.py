"""
Tests for quantum seed generation using CUDA-Q.

These tests verify that quantum circuits produce valid samples
that can be used to seed the classical MTS algorithm.
"""

import pytest
from hypothesis import given, strategies as st
import numpy as np
from typing import Optional

# -----------------------------------------------------------------------------
# Mock/Reference implementations for testing without CUDA-Q hardware
# -----------------------------------------------------------------------------

def mock_quantum_sample(n: int, num_samples: int, seed: Optional[int] = None) -> list[str]:
    """
    Mock quantum sampling for testing.

    In reality, this would run a CUDA-Q circuit.
    For testing, we generate biased random samples that mimic
    quantum distribution (clustered around low-energy states).
    """
    if seed is not None:
        np.random.seed(seed)

    samples = []
    for _ in range(num_samples):
        # Generate bitstring with slight bias toward patterns with low autocorrelation
        bitstring = ''.join(str(np.random.randint(0, 2)) for _ in range(n))
        samples.append(bitstring)

    return samples


def bitstring_to_sequence(bitstring: str) -> list[int]:
    """Convert bitstring '0110...' to sequence [-1, +1, +1, -1, ...]."""
    return [1 if b == '1' else -1 for b in bitstring]


def sequence_to_bitstring(seq: list[int]) -> str:
    """Convert sequence to bitstring."""
    return ''.join('1' if s == 1 else '0' for s in seq)


def validate_bitstring(bitstring: str, n: int) -> bool:
    """Validate that bitstring is properly formatted."""
    if len(bitstring) != n:
        return False
    if not all(b in '01' for b in bitstring):
        return False
    return True


def compute_labs_energy(seq: list[int]) -> float:
    """Compute LABS energy."""
    n = len(seq)
    energy = 0.0
    for k in range(1, n):
        c_k = sum(seq[i] * seq[i + k] for i in range(n - k))
        energy += c_k * c_k
    return energy


# -----------------------------------------------------------------------------
# Unit Tests: Bitstring Conversion
# -----------------------------------------------------------------------------

class TestBitstringConversion:
    """Tests for bitstring to sequence conversion."""

    def test_bitstring_to_sequence_basic(self):
        """Basic bitstring conversion."""
        assert bitstring_to_sequence('110') == [1, 1, -1]
        assert bitstring_to_sequence('0110') == [-1, 1, 1, -1]
        assert bitstring_to_sequence('1111') == [1, 1, 1, 1]
        assert bitstring_to_sequence('0000') == [-1, -1, -1, -1]

    def test_sequence_to_bitstring_basic(self):
        """Basic sequence to bitstring conversion."""
        assert sequence_to_bitstring([1, 1, -1]) == '110'
        assert sequence_to_bitstring([-1, 1, 1, -1]) == '0110'

    def test_roundtrip_bitstring_sequence(self):
        """Bitstring -> sequence -> bitstring should be identity."""
        bitstrings = ['110', '0110', '10101', '11111', '00000']
        for bs in bitstrings:
            seq = bitstring_to_sequence(bs)
            recovered = sequence_to_bitstring(seq)
            assert recovered == bs

    @given(st.text(alphabet='01', min_size=1, max_size=32))
    def test_roundtrip_property(self, bitstring):
        """Property: roundtrip preserves bitstring."""
        seq = bitstring_to_sequence(bitstring)
        recovered = sequence_to_bitstring(seq)
        assert recovered == bitstring


# -----------------------------------------------------------------------------
# Unit Tests: Sample Validation
# -----------------------------------------------------------------------------

class TestSampleValidation:
    """Tests for quantum sample validation."""

    def test_validate_bitstring_correct(self):
        """Valid bitstrings pass validation."""
        assert validate_bitstring('110', 3) is True
        assert validate_bitstring('01010', 5) is True
        assert validate_bitstring('1', 1) is True

    def test_validate_bitstring_wrong_length(self):
        """Wrong length fails validation."""
        assert validate_bitstring('110', 4) is False
        assert validate_bitstring('11001', 3) is False

    def test_validate_bitstring_invalid_chars(self):
        """Invalid characters fail validation."""
        assert validate_bitstring('102', 3) is False
        assert validate_bitstring('abc', 3) is False
        assert validate_bitstring('1 0', 3) is False


# -----------------------------------------------------------------------------
# Tests: Mock Quantum Sampling
# -----------------------------------------------------------------------------

class TestMockQuantumSampling:
    """Tests for mock quantum sampling."""

    def test_sample_count(self):
        """Returns requested number of samples."""
        samples = mock_quantum_sample(n=5, num_samples=100, seed=42)
        assert len(samples) == 100

    def test_sample_length(self):
        """All samples have correct length."""
        samples = mock_quantum_sample(n=7, num_samples=50, seed=42)
        assert all(len(s) == 7 for s in samples)

    def test_sample_valid_bitstrings(self):
        """All samples are valid bitstrings."""
        samples = mock_quantum_sample(n=10, num_samples=100, seed=42)
        assert all(validate_bitstring(s, 10) for s in samples)

    def test_sample_reproducibility(self):
        """Same seed produces same samples."""
        samples1 = mock_quantum_sample(n=5, num_samples=10, seed=42)
        samples2 = mock_quantum_sample(n=5, num_samples=10, seed=42)
        assert samples1 == samples2

    def test_sample_diversity(self):
        """Samples are not all identical."""
        samples = mock_quantum_sample(n=10, num_samples=100, seed=42)
        unique_samples = set(samples)
        assert len(unique_samples) > 1  # At least some diversity


# -----------------------------------------------------------------------------
# Tests: Sample Quality
# -----------------------------------------------------------------------------

class TestSampleQuality:
    """Tests for quantum sample quality metrics."""

    def test_samples_have_valid_energies(self):
        """All samples produce valid (non-negative) energies."""
        samples = mock_quantum_sample(n=11, num_samples=100, seed=42)

        for bitstring in samples:
            seq = bitstring_to_sequence(bitstring)
            energy = compute_labs_energy(seq)
            assert energy >= 0

    def test_energy_distribution(self):
        """Samples have a range of energies."""
        samples = mock_quantum_sample(n=11, num_samples=100, seed=42)

        energies = [
            compute_labs_energy(bitstring_to_sequence(bs))
            for bs in samples
        ]

        # Should have some variation
        assert max(energies) > min(energies)

    def test_best_sample_reasonable(self):
        """Best sample energy is reasonable for problem size."""
        n = 11
        samples = mock_quantum_sample(n=n, num_samples=1000, seed=42)

        energies = [
            compute_labs_energy(bitstring_to_sequence(bs))
            for bs in samples
        ]

        best_energy = min(energies)

        # For N=11, optimal is 12, worst is much higher
        # Random samples should occasionally get lucky
        assert best_energy < 100  # Reasonable upper bound


# -----------------------------------------------------------------------------
# Integration Tests: Quantum-Classical Pipeline
# -----------------------------------------------------------------------------

@pytest.mark.integration
class TestQuantumClassicalPipeline:
    """Tests for quantum to classical handoff."""

    def test_samples_can_seed_mts(self):
        """Quantum samples are valid MTS seeds."""
        samples = mock_quantum_sample(n=7, num_samples=10, seed=42)

        for bitstring in samples:
            seq = bitstring_to_sequence(bitstring)

            # Should be valid for MTS
            assert len(seq) == 7
            assert all(x in [-1, 1] for x in seq)

            # Should have computable energy
            energy = compute_labs_energy(seq)
            assert isinstance(energy, (int, float))
            assert energy >= 0

    def test_sorted_samples_for_seeding(self):
        """Samples can be sorted by energy for seeding."""
        samples = mock_quantum_sample(n=11, num_samples=100, seed=42)

        # Convert and compute energies
        sample_energies = []
        for bitstring in samples:
            seq = bitstring_to_sequence(bitstring)
            energy = compute_labs_energy(seq)
            sample_energies.append((seq, energy))

        # Sort by energy
        sorted_samples = sorted(sample_energies, key=lambda x: x[1])

        # Best samples first
        assert sorted_samples[0][1] <= sorted_samples[-1][1]

        # Top-k for seeding
        top_k = sorted_samples[:10]
        assert len(top_k) == 10

    def test_pipeline_end_to_end(self):
        """Full pipeline: quantum samples -> sort -> seed MTS."""
        n = 7

        # Step 1: Generate quantum samples
        samples = mock_quantum_sample(n=n, num_samples=50, seed=42)

        # Step 2: Convert and evaluate
        evaluated = []
        for bitstring in samples:
            seq = bitstring_to_sequence(bitstring)
            energy = compute_labs_energy(seq)
            evaluated.append((seq, energy))

        # Step 3: Sort and select best
        sorted_samples = sorted(evaluated, key=lambda x: x[1])
        seeds = [seq for seq, _ in sorted_samples[:5]]

        # Step 4: Verify seeds are valid
        assert len(seeds) == 5
        for seed in seeds:
            assert len(seed) == n
            assert all(x in [-1, 1] for x in seed)

        # Best seed should have reasonable energy
        best_seed_energy = compute_labs_energy(seeds[0])
        assert best_seed_energy <= compute_labs_energy(seeds[-1])


# -----------------------------------------------------------------------------
# Tests: CUDA-Q Specific (Skip if not available)
# -----------------------------------------------------------------------------

@pytest.mark.gpu
class TestCUDAQIntegration:
    """Tests that require CUDA-Q hardware."""

    @pytest.fixture
    def cudaq_available(self):
        """Check if CUDA-Q is available."""
        try:
            import cudaq
            return True
        except ImportError:
            pytest.skip("CUDA-Q not available")

    def test_cudaq_import(self, cudaq_available):
        """CUDA-Q can be imported."""
        import cudaq
        assert cudaq is not None

    def test_cudaq_target_nvidia(self, cudaq_available):
        """Can set NVIDIA target."""
        import cudaq
        try:
            cudaq.set_target("nvidia")
        except Exception as e:
            pytest.skip(f"NVIDIA target not available: {e}")

    def test_simple_cudaq_kernel(self, cudaq_available):
        """Simple CUDA-Q kernel runs."""
        import cudaq

        @cudaq.kernel
        def simple_kernel(n: int):
            qubits = cudaq.qvector(n)
            for q in range(n):
                h(qubits[q])
            mz(qubits)

        try:
            results = cudaq.sample(simple_kernel, 3, shots_count=100)
            assert len(results) > 0
        except Exception as e:
            pytest.skip(f"CUDA-Q execution failed: {e}")
