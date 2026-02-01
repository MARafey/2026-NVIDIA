#!/usr/bin/env python3
"""
Direct test runner for LABS Solver - bypasses pytest configuration issues.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from labs_solver.energy import calculate_energy, compute_autocorrelation
from labs_solver.mts import memetic_tabu_search, tabu_search, random_sequence, combine, mutate
from labs_solver.utils import (
    bitstring_to_sequence,
    sequence_to_bitstring,
    pack_sequence,
    unpack_sequence,
    get_interactions,
)


def run_test(name, test_func):
    """Run a test and report result."""
    try:
        test_func()
        print(f"  [PASS] {name}")
        return True
    except AssertionError as e:
        print(f"  [FAIL] {name}: {e}")
        return False
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        return False


# =============================================================================
# Energy Tests
# =============================================================================

def test_energy_n3():
    """N=3 optimal energy is 1."""
    seq = [1, 1, -1]
    assert calculate_energy(seq) == 1.0

def test_energy_n5():
    """N=5 optimal energy is 2."""
    seq = [1, 1, 1, -1, 1]
    assert calculate_energy(seq) == 2.0

def test_energy_n7():
    """N=7 optimal energy is 3."""
    seq = [1, 1, 1, -1, -1, 1, -1]
    assert calculate_energy(seq) == 3.0

def test_energy_negation_symmetry():
    """E(S) must equal E(-S)."""
    for _ in range(10):
        seq = random_sequence(10)
        negated = [-s for s in seq]
        assert calculate_energy(seq) == calculate_energy(negated)

def test_energy_reversal_symmetry():
    """E(S) must equal E(reverse(S))."""
    for _ in range(10):
        seq = random_sequence(10)
        reversed_seq = seq[::-1]
        assert calculate_energy(seq) == calculate_energy(reversed_seq)

def test_energy_non_negative():
    """Energy must be >= 0."""
    for _ in range(20):
        seq = random_sequence(15)
        assert calculate_energy(seq) >= 0

def test_energy_is_integer():
    """Energy must be an integer."""
    for _ in range(20):
        seq = random_sequence(15)
        energy = calculate_energy(seq)
        assert energy == int(energy)


# =============================================================================
# Bit Packing Tests
# =============================================================================

def test_bitstring_conversion():
    """Bitstring to sequence round-trip."""
    seq = [1, 1, -1, 1, -1]
    bs = sequence_to_bitstring(seq)
    recovered = bitstring_to_sequence(bs)
    assert seq == recovered

def test_pack_unpack():
    """Pack and unpack round-trip."""
    for n in [5, 10, 20, 32]:
        seq = random_sequence(n)
        packed = pack_sequence(seq)
        unpacked = unpack_sequence(packed, n)
        assert seq == unpacked, f"Failed for N={n}"

def test_pack_uniqueness():
    """Different sequences must have different packed values."""
    seq1 = [1, 1, 1, -1, -1]
    seq2 = [1, -1, 1, -1, -1]
    assert pack_sequence(seq1) != pack_sequence(seq2)


# =============================================================================
# Interaction Indices Tests
# =============================================================================

def test_interactions_n5():
    """Test interaction indices for N=5."""
    G2, G4 = get_interactions(5)
    expected_G2 = [[0, 1], [0, 2], [1, 2], [2, 3]]
    expected_G4 = [[0, 1, 2, 3], [0, 1, 3, 4], [1, 2, 3, 4]]
    assert G2 == expected_G2, f"G2 mismatch: {G2}"
    assert G4 == expected_G4, f"G4 mismatch: {G4}"

def test_interactions_bounds():
    """All indices must be within bounds."""
    for n in [6, 8, 10, 15]:
        G2, G4 = get_interactions(n)
        for pair in G2:
            assert all(0 <= idx < n for idx in pair)
        for quad in G4:
            assert all(0 <= idx < n for idx in quad)


# =============================================================================
# MTS Tests
# =============================================================================

def test_tabu_search_improves():
    """Tabu search should not worsen the solution."""
    for _ in range(5):
        seq = random_sequence(10)
        initial_energy = calculate_energy(seq)
        improved, final_energy = tabu_search(seq)
        assert final_energy <= initial_energy

def test_mts_finds_optimal_n5():
    """MTS should find optimal for N=5."""
    found_optimal = False
    for _ in range(5):
        seq, energy, _, _ = memetic_tabu_search(5, population_size=10, max_generations=30)
        if energy == 2.0:
            found_optimal = True
            break
    assert found_optimal, "Failed to find optimal for N=5"

def test_mts_finds_optimal_n7():
    """MTS should find optimal for N=7."""
    found_optimal = False
    for _ in range(5):
        seq, energy, _, _ = memetic_tabu_search(7, population_size=15, max_generations=40)
        if energy == 3.0:  # Correct optimal energy for N=7
            found_optimal = True
            break
    assert found_optimal, "Failed to find optimal for N=7"


# =============================================================================
# Run All Tests
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  LABS Solver Test Suite")
    print("=" * 60)

    results = []

    print("\n[Energy Tests]")
    results.append(run_test("energy_n3", test_energy_n3))
    results.append(run_test("energy_n5", test_energy_n5))
    results.append(run_test("energy_n7", test_energy_n7))
    results.append(run_test("energy_negation_symmetry", test_energy_negation_symmetry))
    results.append(run_test("energy_reversal_symmetry", test_energy_reversal_symmetry))
    results.append(run_test("energy_non_negative", test_energy_non_negative))
    results.append(run_test("energy_is_integer", test_energy_is_integer))

    print("\n[Bit Packing Tests]")
    results.append(run_test("bitstring_conversion", test_bitstring_conversion))
    results.append(run_test("pack_unpack", test_pack_unpack))
    results.append(run_test("pack_uniqueness", test_pack_uniqueness))

    print("\n[Interaction Indices Tests]")
    results.append(run_test("interactions_n5", test_interactions_n5))
    results.append(run_test("interactions_bounds", test_interactions_bounds))

    print("\n[MTS Tests]")
    results.append(run_test("tabu_search_improves", test_tabu_search_improves))
    results.append(run_test("mts_finds_optimal_n5", test_mts_finds_optimal_n5))
    results.append(run_test("mts_finds_optimal_n7", test_mts_finds_optimal_n7))

    # Summary
    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{total} tests passed")
    print("=" * 60)

    if passed == total:
        print("\n  All tests PASSED!")
        return 0
    else:
        print(f"\n  {total - passed} test(s) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
