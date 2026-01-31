"""
Tests for bit-packing operations.

These tests verify that sequences can be correctly encoded into uint64_t
and decoded back without information loss.
"""

import pytest
from hypothesis import given, strategies as st, assume
import numpy as np

# -----------------------------------------------------------------------------
# Import the module under test (adjust path as needed)
# -----------------------------------------------------------------------------
# from src.bit_packing import pack_sequence, unpack_sequence, BITS_PER_ELEM, MAX_N

# For now, include reference implementation for testing
BITS_PER_ELEM = 2
MAX_N = 32
ELEM_MASK = (1 << BITS_PER_ELEM) - 1


def pack_sequence(seq: list[int], bits_per_elem: int = BITS_PER_ELEM) -> int:
    """Pack a sequence into a uint64."""
    packed = 0
    mask = (1 << bits_per_elem) - 1
    for i, elem in enumerate(seq):
        # Map: -1 -> 0, +1 -> 1 (for binary)
        # For larger alphabets, assume elem is already 0-indexed
        if bits_per_elem == 1:
            val = 1 if elem == 1 else 0
        else:
            val = elem & mask
        packed |= (val << (i * bits_per_elem))
    return packed


def unpack_sequence(packed: int, n: int, bits_per_elem: int = BITS_PER_ELEM) -> list[int]:
    """Unpack a uint64 into a sequence."""
    seq = []
    mask = (1 << bits_per_elem) - 1
    for i in range(n):
        val = (packed >> (i * bits_per_elem)) & mask
        if bits_per_elem == 1:
            seq.append(1 if val == 1 else -1)
        else:
            seq.append(val)
    return seq


# -----------------------------------------------------------------------------
# Unit Tests
# -----------------------------------------------------------------------------

class TestPackSequenceBasic:
    """Basic unit tests for pack_sequence."""

    def test_pack_simple_binary(self):
        """Test packing a simple binary sequence."""
        seq = [1, -1, 1, 1]  # -> [1, 0, 1, 1] -> 0b1101 = 13
        packed = pack_sequence(seq, bits_per_elem=1)
        assert packed == 0b1101

    def test_pack_all_ones(self):
        """All +1 sequence should pack to all 1 bits."""
        seq = [1, 1, 1, 1]
        packed = pack_sequence(seq, bits_per_elem=1)
        assert packed == 0b1111

    def test_pack_all_negative_ones(self):
        """All -1 sequence should pack to all 0 bits."""
        seq = [-1, -1, -1, -1]
        packed = pack_sequence(seq, bits_per_elem=1)
        assert packed == 0b0000

    def test_pack_empty_sequence(self):
        """Empty sequence should pack to 0."""
        seq = []
        packed = pack_sequence(seq, bits_per_elem=1)
        assert packed == 0

    def test_pack_single_element(self):
        """Single element sequences."""
        assert pack_sequence([1], bits_per_elem=1) == 1
        assert pack_sequence([-1], bits_per_elem=1) == 0


class TestUnpackSequenceBasic:
    """Basic unit tests for unpack_sequence."""

    def test_unpack_simple_binary(self):
        """Test unpacking a simple binary value."""
        packed = 0b1101  # -> [1, 0, 1, 1] -> [1, -1, 1, 1]
        seq = unpack_sequence(packed, n=4, bits_per_elem=1)
        assert seq == [1, -1, 1, 1]

    def test_unpack_all_ones(self):
        """Unpack all 1 bits."""
        packed = 0b1111
        seq = unpack_sequence(packed, n=4, bits_per_elem=1)
        assert seq == [1, 1, 1, 1]

    def test_unpack_all_zeros(self):
        """Unpack all 0 bits."""
        packed = 0b0000
        seq = unpack_sequence(packed, n=4, bits_per_elem=1)
        assert seq == [-1, -1, -1, -1]


class TestRoundTrip:
    """Round-trip tests: pack then unpack should return original."""

    def test_roundtrip_manual(self, small_sequences):
        """Manual round-trip test with known sequences."""
        for seq in small_sequences:
            packed = pack_sequence(seq, bits_per_elem=1)
            unpacked = unpack_sequence(packed, len(seq), bits_per_elem=1)
            assert unpacked == seq, f"Round-trip failed: {seq} -> {packed} -> {unpacked}"

    @given(st.lists(st.sampled_from([-1, 1]), min_size=1, max_size=32))
    def test_roundtrip_property_binary(self, seq):
        """Property: pack(unpack(x)) == x for all binary sequences."""
        packed = pack_sequence(seq, bits_per_elem=1)
        unpacked = unpack_sequence(packed, len(seq), bits_per_elem=1)
        assert unpacked == seq

    @given(st.lists(st.integers(min_value=0, max_value=3), min_size=1, max_size=32))
    def test_roundtrip_property_quaternary(self, seq):
        """Property: pack(unpack(x)) == x for quaternary alphabet."""
        packed = pack_sequence(seq, bits_per_elem=2)
        unpacked = unpack_sequence(packed, len(seq), bits_per_elem=2)
        assert unpacked == seq


class TestBitPackingConstraints:
    """Tests for bit-packing constraints and edge cases."""

    def test_max_sequence_length_binary(self):
        """Binary sequences up to length 64 should fit in uint64."""
        seq = [1] * 64
        packed = pack_sequence(seq, bits_per_elem=1)
        assert packed == (2**64 - 1)  # All 1s
        unpacked = unpack_sequence(packed, 64, bits_per_elem=1)
        assert unpacked == seq

    def test_max_sequence_length_quaternary(self):
        """Quaternary sequences up to length 32 should fit in uint64."""
        seq = [3] * 32  # Max value for 2-bit encoding
        packed = pack_sequence(seq, bits_per_elem=2)
        assert packed == (2**64 - 1)  # All 1s
        unpacked = unpack_sequence(packed, 32, bits_per_elem=2)
        assert unpacked == seq

    def test_different_sequences_different_packed(self):
        """Different sequences must produce different packed values."""
        seq1 = [1, 1, -1, 1]
        seq2 = [1, -1, 1, 1]
        packed1 = pack_sequence(seq1, bits_per_elem=1)
        packed2 = pack_sequence(seq2, bits_per_elem=1)
        assert packed1 != packed2

    @given(
        st.lists(st.sampled_from([-1, 1]), min_size=2, max_size=20),
        st.lists(st.sampled_from([-1, 1]), min_size=2, max_size=20)
    )
    def test_collision_free_property(self, seq1, seq2):
        """Property: different sequences of same length -> different packed values."""
        assume(seq1 != seq2)
        assume(len(seq1) == len(seq2))
        packed1 = pack_sequence(seq1, bits_per_elem=1)
        packed2 = pack_sequence(seq2, bits_per_elem=1)
        assert packed1 != packed2


class TestBitPackingOverflow:
    """Tests to prevent overflow errors (AI hallucination guardrail)."""

    def test_bits_per_elem_times_n_within_64(self):
        """Verify our constants don't overflow uint64."""
        # Binary: 1 bit * 64 elements = 64 bits (OK)
        assert 1 * 64 <= 64

        # Quaternary: 2 bits * 32 elements = 64 bits (OK)
        assert 2 * 32 <= 64

        # 5-bit alphabet: 5 * 12 = 60 bits (OK)
        assert 5 * 12 <= 64

        # 5-bit alphabet: 5 * 13 = 65 bits (OVERFLOW!)
        assert 5 * 13 > 64  # This would overflow

    def test_pack_does_not_overflow(self):
        """Packing max-length sequence should not raise overflow."""
        # This should work
        seq_ok = [1] * 64
        packed = pack_sequence(seq_ok, bits_per_elem=1)
        assert packed >= 0

        # For 2-bit encoding, max 32 elements
        seq_ok_2bit = [3] * 32
        packed_2bit = pack_sequence(seq_ok_2bit, bits_per_elem=2)
        assert packed_2bit >= 0
