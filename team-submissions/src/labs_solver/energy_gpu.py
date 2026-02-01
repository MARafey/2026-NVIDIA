"""
GPU-Accelerated LABS Energy Calculation using CuPy

Provides vectorized and batched energy computation on GPU for
significant speedup over CPU implementation.
"""

from typing import List, Union
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def calculate_energy_gpu(sequence: Union[List[int], np.ndarray]) -> float:
    """
    Calculate LABS energy on GPU.

    E(S) = sum_{k=1}^{N-1} C_k^2
    where C_k = sum_{i=0}^{N-1-k} S_i * S_{i+k}

    Args:
        sequence: List or array of +1/-1 values

    Returns:
        The LABS energy
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available for GPU computation")

    seq = cp.asarray(sequence, dtype=cp.float32)
    n = len(seq)
    energy = 0.0

    for k in range(1, n):
        c_k = cp.dot(seq[:-k], seq[k:])
        energy += float(c_k * c_k)

    return energy


def calculate_energy_batch_gpu(sequences: np.ndarray) -> np.ndarray:
    """
    Calculate LABS energy for a batch of sequences on GPU.

    This is the key acceleration: computing energy for many sequences
    simultaneously exploits GPU parallelism.

    Args:
        sequences: 2D array of shape (batch_size, sequence_length)
                   with +1/-1 values

    Returns:
        1D array of energies for each sequence
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available for GPU computation")

    # Transfer to GPU
    seqs = cp.asarray(sequences, dtype=cp.float32)
    batch_size, n = seqs.shape

    # Initialize energy array
    energies = cp.zeros(batch_size, dtype=cp.float32)

    # Compute autocorrelations for each lag
    for k in range(1, n):
        # Vectorized autocorrelation across batch
        # seqs[:, :-k] has shape (batch, n-k)
        # seqs[:, k:] has shape (batch, n-k)
        c_k = cp.sum(seqs[:, :-k] * seqs[:, k:], axis=1)
        energies += c_k * c_k

    return cp.asnumpy(energies)


def calculate_energy_batch_cpu(sequences: np.ndarray) -> np.ndarray:
    """
    Calculate LABS energy for a batch of sequences on CPU (fallback).

    Args:
        sequences: 2D array of shape (batch_size, sequence_length)

    Returns:
        1D array of energies
    """
    batch_size, n = sequences.shape
    energies = np.zeros(batch_size, dtype=np.float32)

    for k in range(1, n):
        c_k = np.sum(sequences[:, :-k] * sequences[:, k:], axis=1)
        energies += c_k * c_k

    return energies


def get_all_neighbors_batch(sequences: np.ndarray) -> np.ndarray:
    """
    Generate all single-flip neighbors for a batch of sequences.

    For each sequence, generates N neighbors (one for each position).

    Args:
        sequences: 2D array of shape (batch_size, n)

    Returns:
        3D array of shape (batch_size, n, n) where [i, j, :] is the
        j-th neighbor of the i-th sequence
    """
    batch_size, n = sequences.shape

    # Expand sequences to (batch_size, n, n) - broadcast for all neighbors
    expanded = np.tile(sequences[:, np.newaxis, :], (1, n, 1))

    # Create flip mask: flip position j for neighbor j
    flip_mask = np.eye(n, dtype=sequences.dtype) * -2

    # Apply flips: seq[j] -> seq[j] * -1 is same as seq + flip_mask[j] * seq[j]
    # More directly: multiply by -1 at diagonal positions
    neighbors = expanded.copy()
    for j in range(n):
        neighbors[:, j, j] *= -1

    return neighbors


def evaluate_neighbors_gpu(sequences: np.ndarray) -> tuple:
    """
    Evaluate all neighbors for a batch of sequences on GPU.

    This is the main optimization: instead of evaluating neighbors
    one at a time, we evaluate all neighbors for all sequences in parallel.

    Args:
        sequences: 2D array of shape (batch_size, n)

    Returns:
        Tuple of (neighbor_energies, neighbors) where:
        - neighbor_energies: (batch_size, n) array of energies
        - neighbors: (batch_size, n, n) array of neighbor sequences
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")

    batch_size, n = sequences.shape

    # Generate all neighbors
    neighbors = get_all_neighbors_batch(sequences)

    # Reshape to (batch_size * n, n) for batch energy calculation
    neighbors_flat = neighbors.reshape(-1, n)

    # Calculate energies on GPU
    energies_flat = calculate_energy_batch_gpu(neighbors_flat)

    # Reshape back to (batch_size, n)
    neighbor_energies = energies_flat.reshape(batch_size, n)

    return neighbor_energies, neighbors


class GPUEnergyCache:
    """
    GPU-side cache for storing computed energies.

    Implements the shared exploration cache from the PRD:
    - Hash table mapping packed sequences to energies
    - O(1) lookup before computation
    - Keeps lower energy on collision
    """

    def __init__(self, capacity: int = 4096, sequence_length: int = 32):
        """
        Initialize cache.

        Args:
            capacity: Number of cache entries
            sequence_length: Length of sequences to cache
        """
        self.capacity = capacity
        self.seq_len = sequence_length

        if CUPY_AVAILABLE:
            # Store packed sequences and energies
            self.packed_seqs = cp.zeros(capacity, dtype=cp.uint64)
            self.energies = cp.full(capacity, cp.inf, dtype=cp.float32)
            self.valid = cp.zeros(capacity, dtype=cp.bool_)
        else:
            self.packed_seqs = np.zeros(capacity, dtype=np.uint64)
            self.energies = np.full(capacity, np.inf, dtype=np.float32)
            self.valid = np.zeros(capacity, dtype=bool)

        self.hits = 0
        self.misses = 0

    def _pack_sequence(self, seq: np.ndarray) -> int:
        """Pack sequence into uint64."""
        packed = 0
        for i, s in enumerate(seq):
            if s == -1:
                packed |= (1 << i)
        return packed

    def _hash(self, packed: int) -> int:
        """Simple hash function."""
        return packed % self.capacity

    def lookup(self, seq: np.ndarray) -> tuple:
        """
        Look up sequence in cache.

        Returns:
            Tuple of (found, energy)
        """
        packed = self._pack_sequence(seq)
        idx = self._hash(packed)

        if CUPY_AVAILABLE:
            valid = bool(self.valid[idx].get())
            stored_packed = int(self.packed_seqs[idx].get())
            energy = float(self.energies[idx].get())
        else:
            valid = self.valid[idx]
            stored_packed = self.packed_seqs[idx]
            energy = self.energies[idx]

        if valid and stored_packed == packed:
            self.hits += 1
            return True, energy
        else:
            self.misses += 1
            return False, 0.0

    def insert(self, seq: np.ndarray, energy: float):
        """
        Insert or update sequence in cache.

        Uses "keep better" policy: only overwrites if new energy is lower.
        """
        packed = self._pack_sequence(seq)
        idx = self._hash(packed)

        if CUPY_AVAILABLE:
            current_valid = bool(self.valid[idx].get())
            current_energy = float(self.energies[idx].get())
            current_packed = int(self.packed_seqs[idx].get())
        else:
            current_valid = self.valid[idx]
            current_energy = self.energies[idx]
            current_packed = self.packed_seqs[idx]

        # Insert if empty slot or same sequence with worse energy
        if not current_valid or current_packed != packed or energy < current_energy:
            if CUPY_AVAILABLE:
                self.packed_seqs[idx] = cp.uint64(packed)
                self.energies[idx] = cp.float32(energy)
                self.valid[idx] = True
            else:
                self.packed_seqs[idx] = packed
                self.energies[idx] = energy
                self.valid[idx] = True

    def get_hit_rate(self) -> float:
        """Return cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def clear(self):
        """Clear the cache."""
        if CUPY_AVAILABLE:
            self.valid[:] = False
            self.energies[:] = cp.inf
        else:
            self.valid[:] = False
            self.energies[:] = np.inf
        self.hits = 0
        self.misses = 0
