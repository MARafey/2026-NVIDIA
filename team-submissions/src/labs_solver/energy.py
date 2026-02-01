"""
LABS Energy Calculation

The LABS (Low Autocorrelation Binary Sequences) energy is defined as:
    E(S) = sum_{k=1}^{N-1} C_k^2

where C_k is the autocorrelation at lag k:
    C_k = sum_{i=0}^{N-1-k} S_i * S_{i+k}

Lower energy indicates better sequences for radar/communications applications.
"""

from typing import List, Union
import numpy as np


def compute_autocorrelation(seq: List[int], k: int) -> int:
    """
    Compute autocorrelation C_k for a given lag k.

    Args:
        seq: Binary sequence of +1/-1 values
        k: Lag value (1 <= k < N)

    Returns:
        Autocorrelation value C_k
    """
    n = len(seq)
    return sum(seq[i] * seq[i + k] for i in range(n - k))


def calculate_energy(sequence: Union[List[int], np.ndarray]) -> float:
    """
    Calculate the LABS energy for a binary sequence.

    E(S) = sum_{k=1}^{N-1} C_k^2
    where C_k = sum_{i=0}^{N-1-k} S_i * S_{i+k}

    Args:
        sequence: List or array of +1/-1 values

    Returns:
        The LABS energy (always non-negative integer, returned as float)
    """
    n = len(sequence)
    energy = 0.0
    for k in range(1, n):
        c_k = sum(sequence[i] * sequence[i + k] for i in range(n - k))
        energy += c_k * c_k
    return energy


def calculate_energy_vectorized(seq: np.ndarray) -> float:
    """
    Vectorized energy computation using NumPy.

    Faster for larger sequences.

    Args:
        seq: NumPy array of +1/-1 values

    Returns:
        The LABS energy
    """
    n = len(seq)
    energy = 0.0
    for k in range(1, n):
        c_k = np.dot(seq[:-k], seq[k:])
        energy += c_k * c_k
    return energy


def merit_factor(sequence: Union[List[int], np.ndarray]) -> float:
    """
    Calculate the merit factor F = N^2 / (2 * E(S)).

    Higher merit factor indicates better sequences.
    The best known merit factors approach ~12.3 for large N.

    Args:
        sequence: Binary sequence of +1/-1 values

    Returns:
        Merit factor (infinity if energy is 0)
    """
    n = len(sequence)
    energy = calculate_energy(sequence)
    if energy == 0:
        return float('inf')
    return (n * n) / (2.0 * energy)


def approximation_ratio(sequence: Union[List[int], np.ndarray], optimal_energy: float) -> float:
    """
    Calculate approximation ratio = optimal_energy / found_energy.

    A ratio of 1.0 means optimal solution found.

    Args:
        sequence: Binary sequence of +1/-1 values
        optimal_energy: Known or estimated optimal energy

    Returns:
        Approximation ratio in (0, 1]
    """
    found_energy = calculate_energy(sequence)
    if found_energy == 0:
        return 1.0 if optimal_energy == 0 else float('inf')
    return optimal_energy / found_energy
