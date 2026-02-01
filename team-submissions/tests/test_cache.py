"""
Tests for shared exploration cache.

The cache stores (packed_sequence -> energy) mappings with:
- O(1) hash-based lookup
- "Keep better energy" collision policy
- Tiered structure: protected quantum seeds + organic growth
"""

import pytest
from hypothesis import given, strategies as st, assume
import numpy as np
from typing import Optional
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# Reference implementation for testing
# -----------------------------------------------------------------------------

EMPTY_SLOT = 0xFFFFFFFFFFFFFFFF  # Max uint64 as sentinel


@dataclass
class CacheEntry:
    """Single cache entry."""
    packed_seq: int
    energy: float

    @classmethod
    def empty(cls) -> 'CacheEntry':
        return cls(packed_seq=EMPTY_SLOT, energy=float('inf'))


class SharedCache:
    """
    Simulated shared memory cache for testing.

    Implements the "keep better energy" collision policy.
    """

    def __init__(self, size: int = 2048, protected_ratio: float = 0.25):
        self.size = size
        self.protected_size = int(size * protected_ratio)
        self.entries = [CacheEntry.empty() for _ in range(size)]
        self.hits = 0
        self.misses = 0

    def _hash(self, packed_seq: int) -> int:
        """Simple modulo hash."""
        return packed_seq % self.size

    def lookup(self, packed_seq: int) -> Optional[float]:
        """
        Look up energy for a packed sequence.
        Returns energy if found, None if not.
        """
        slot = self._hash(packed_seq)
        entry = self.entries[slot]

        if entry.packed_seq == packed_seq:
            self.hits += 1
            return entry.energy
        else:
            self.misses += 1
            return None

    def update(self, packed_seq: int, energy: float) -> bool:
        """
        Update cache with (sequence, energy) pair.

        Policy:
        - Empty slot: insert
        - Same sequence: update if energy is better
        - Different sequence (collision): keep better energy
        - Protected region: only update same sequence

        Returns True if inserted/updated, False if rejected.
        """
        slot = self._hash(packed_seq)
        entry = self.entries[slot]

        # Protected region (quantum seeds)
        if slot < self.protected_size:
            if entry.packed_seq == EMPTY_SLOT:
                # Empty protected slot - insert
                self.entries[slot] = CacheEntry(packed_seq, energy)
                return True
            elif entry.packed_seq == packed_seq:
                # Same sequence - update if better
                if energy < entry.energy:
                    self.entries[slot].energy = energy
                    return True
            # Different sequence in protected region - reject
            return False

        # Organic region
        if entry.packed_seq == EMPTY_SLOT:
            # Empty slot - insert
            self.entries[slot] = CacheEntry(packed_seq, energy)
            return True
        elif entry.packed_seq == packed_seq:
            # Same sequence - update if better
            if energy < entry.energy:
                self.entries[slot].energy = energy
                return True
            return False
        else:
            # Collision - keep better energy
            if energy < entry.energy:
                self.entries[slot] = CacheEntry(packed_seq, energy)
                return True
            return False

    def seed(self, seeds: list[tuple[int, float]]):
        """Initialize cache with quantum seeds in protected region."""
        for i, (packed_seq, energy) in enumerate(seeds):
            if i >= self.protected_size:
                break
            # Use hash to determine slot (like normal lookup/update)
            slot = self._hash(packed_seq)
            self.entries[slot] = CacheEntry(packed_seq, energy)

    def get_hit_rate(self) -> float:
        """Return cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_occupancy(self) -> float:
        """Return fraction of slots occupied."""
        occupied = sum(1 for e in self.entries if e.packed_seq != EMPTY_SLOT)
        return occupied / self.size


# -----------------------------------------------------------------------------
# Unit Tests: Basic Cache Operations
# -----------------------------------------------------------------------------

class TestCacheBasicOperations:
    """Basic cache operation tests."""

    def test_insert_and_lookup(self):
        """Basic insert and lookup."""
        cache = SharedCache(size=1024, protected_ratio=0.0)

        packed_seq = 12345
        energy = 42.0

        # Insert
        result = cache.update(packed_seq, energy)
        assert result is True

        # Lookup
        retrieved = cache.lookup(packed_seq)
        assert retrieved == energy

    def test_lookup_empty(self):
        """Lookup on empty cache returns None."""
        cache = SharedCache(size=1024)
        result = cache.lookup(12345)
        assert result is None

    def test_lookup_miss(self):
        """Lookup for non-existent key returns None."""
        cache = SharedCache(size=1024, protected_ratio=0.0)
        cache.update(12345, 10.0)

        result = cache.lookup(99999)  # Different key
        assert result is None

    def test_update_same_sequence_better_energy(self):
        """Updating same sequence with better energy succeeds."""
        cache = SharedCache(size=1024, protected_ratio=0.0)

        packed_seq = 12345
        cache.update(packed_seq, 100.0)
        cache.update(packed_seq, 50.0)  # Better

        assert cache.lookup(packed_seq) == 50.0

    def test_update_same_sequence_worse_energy(self):
        """Updating same sequence with worse energy is rejected."""
        cache = SharedCache(size=1024, protected_ratio=0.0)

        packed_seq = 12345
        cache.update(packed_seq, 50.0)
        cache.update(packed_seq, 100.0)  # Worse

        assert cache.lookup(packed_seq) == 50.0  # Still 50


# -----------------------------------------------------------------------------
# Tests: "Keep Better Energy" Collision Policy
# -----------------------------------------------------------------------------

class TestCollisionPolicy:
    """Tests for collision handling with 'keep better energy' policy."""

    def test_collision_keeps_better(self):
        """On collision, keep the entry with lower energy."""
        # Use tiny cache to force collisions
        cache = SharedCache(size=4, protected_ratio=0.0)

        # Find two sequences that collide
        # seq1 and seq1 + size will hash to same slot
        seq1 = 1
        seq2 = 1 + 4  # Same slot

        cache.update(seq1, 100.0)
        cache.update(seq2, 50.0)  # Better energy, should evict seq1

        # seq2 should be in cache (better energy)
        assert cache.lookup(seq2) == 50.0
        # seq1 was evicted
        assert cache.lookup(seq1) is None

    def test_collision_rejects_worse(self):
        """On collision, reject entry with higher energy."""
        cache = SharedCache(size=4, protected_ratio=0.0)

        seq1 = 1
        seq2 = 1 + 4  # Same slot

        cache.update(seq1, 50.0)
        result = cache.update(seq2, 100.0)  # Worse energy

        assert result is False  # Rejected
        assert cache.lookup(seq1) == 50.0  # Original still there
        assert cache.lookup(seq2) is None  # New one not inserted

    @given(st.integers(min_value=0, max_value=1000000))
    def test_collision_deterministic(self, base_seq):
        """Collision resolution is deterministic."""
        cache1 = SharedCache(size=16, protected_ratio=0.0)
        cache2 = SharedCache(size=16, protected_ratio=0.0)

        seq1 = base_seq
        seq2 = base_seq + 16  # Collides

        # Same operations on both caches
        for cache in [cache1, cache2]:
            cache.update(seq1, 100.0)
            cache.update(seq2, 50.0)

        # Results should match
        assert cache1.lookup(seq1) == cache2.lookup(seq1)
        assert cache1.lookup(seq2) == cache2.lookup(seq2)


# -----------------------------------------------------------------------------
# Tests: Protected Region (Quantum Seeds)
# -----------------------------------------------------------------------------

class TestProtectedRegion:
    """Tests for protected quantum seed region."""

    def test_seed_initialization(self):
        """Seeds are placed in protected region."""
        cache = SharedCache(size=100, protected_ratio=0.25)

        # Use seeds that hash to different slots (0, 1, 2, ... 9)
        seeds = [(i, float(i * 10)) for i in range(10)]
        cache.seed(seeds)

        # All seeds should be retrievable
        for packed_seq, energy in seeds:
            assert cache.lookup(packed_seq) == energy

    def test_protected_region_no_eviction(self):
        """Protected region entries cannot be evicted by collisions."""
        cache = SharedCache(size=100, protected_ratio=0.25)
        # Protected region is slots 0-24

        # Seed slot 0
        cache.entries[0] = CacheEntry(packed_seq=1000, energy=50.0)

        # Try to insert different sequence that hashes to slot 0
        # with better energy
        collision_seq = 100  # 100 % 100 = 0, same slot
        result = cache.update(collision_seq, 10.0)  # Better energy

        # Should be rejected (protected region)
        assert result is False
        assert cache.entries[0].packed_seq == 1000  # Original preserved

    def test_protected_region_same_seq_update(self):
        """Protected entries can be updated with better energy for same sequence."""
        cache = SharedCache(size=100, protected_ratio=0.25)

        cache.entries[0] = CacheEntry(packed_seq=1000, energy=50.0)

        # Update same sequence with better energy
        result = cache.update(1000, 30.0)

        assert result is True
        assert cache.entries[0].energy == 30.0

    def test_organic_region_allows_eviction(self):
        """Organic region allows eviction based on energy."""
        cache = SharedCache(size=100, protected_ratio=0.25)
        # Organic region starts at slot 25

        organic_slot = 50
        seq1 = organic_slot  # Hashes to slot 50
        seq2 = organic_slot + 100  # Also hashes to slot 50

        cache.update(seq1, 100.0)
        cache.update(seq2, 50.0)  # Better, should evict

        assert cache.lookup(seq2) == 50.0


# -----------------------------------------------------------------------------
# Tests: Cache Statistics
# -----------------------------------------------------------------------------

class TestCacheStatistics:
    """Tests for cache statistics tracking."""

    def test_hit_rate_calculation(self):
        """Hit rate is correctly calculated."""
        cache = SharedCache(size=1024, protected_ratio=0.0)

        cache.update(1, 10.0)
        cache.update(2, 20.0)

        # 2 hits
        cache.lookup(1)
        cache.lookup(2)

        # 2 misses
        cache.lookup(3)
        cache.lookup(4)

        assert cache.get_hit_rate() == 0.5

    def test_occupancy_calculation(self):
        """Occupancy is correctly calculated."""
        cache = SharedCache(size=100, protected_ratio=0.0)

        for i in range(25):
            cache.update(i, float(i))

        assert cache.get_occupancy() == 0.25


# -----------------------------------------------------------------------------
# Property-Based Tests
# -----------------------------------------------------------------------------

class TestCacheProperties:
    """Property-based tests for cache invariants."""

    @given(st.lists(st.tuples(
        st.integers(min_value=0, max_value=10000),
        st.floats(min_value=0.0, max_value=1000.0, allow_nan=False)
    ), min_size=1, max_size=100))
    def test_lookup_after_update_consistent(self, insertions):
        """After update, lookup returns correct or None (if evicted)."""
        cache = SharedCache(size=128, protected_ratio=0.0)

        for packed_seq, energy in insertions:
            cache.update(packed_seq, energy)

        # For each unique sequence, lookup should return a valid energy
        # (may not be original if updated with better)
        unique_seqs = set(packed_seq for packed_seq, _ in insertions)

        for seq in unique_seqs:
            result = cache.lookup(seq)
            if result is not None:
                assert result >= 0  # Energy is non-negative

    @given(st.integers(min_value=0, max_value=10000))
    def test_energy_monotonic_improvement(self, packed_seq):
        """For same sequence, energy can only decrease."""
        cache = SharedCache(size=1024, protected_ratio=0.0)

        energies = [100.0, 80.0, 90.0, 50.0, 60.0, 30.0]

        for energy in energies:
            cache.update(packed_seq, energy)

        # Final energy should be minimum
        assert cache.lookup(packed_seq) == 30.0


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------

class TestCacheIntegration:
    """Integration tests simulating real usage patterns."""

    def test_mts_simulation(self):
        """Simulate MTS-like access pattern."""
        cache = SharedCache(size=256, protected_ratio=0.25)

        # Seed with "quantum samples"
        seeds = [(i, 100.0 - i) for i in range(20)]  # Lower energy for higher i
        cache.seed(seeds)

        # Simulate MTS iterations
        np.random.seed(42)
        for _ in range(1000):
            # Generate random sequence
            seq = np.random.randint(0, 10000)
            energy = np.random.uniform(0, 200)

            # Try lookup first
            cached = cache.lookup(seq)
            if cached is None:
                # Miss - compute and cache
                cache.update(seq, energy)

        # Should have some hits after warmup
        assert cache.get_hit_rate() > 0

    def test_convergent_search_pattern(self):
        """Hit rate should increase as search converges."""
        cache = SharedCache(size=128, protected_ratio=0.0)

        # Simulate converging search - sequences cluster around "good" region
        hit_rates = []

        for phase in range(5):
            # Each phase, sequences cluster more tightly
            spread = 100 // (phase + 1)

            for _ in range(100):
                seq = np.random.randint(0, spread)
                energy = float(seq)  # Lower seq = lower energy

                cached = cache.lookup(seq)
                if cached is None:
                    cache.update(seq, energy)

            hit_rates.append(cache.get_hit_rate())

        # Hit rate should generally increase
        assert hit_rates[-1] > hit_rates[0]
