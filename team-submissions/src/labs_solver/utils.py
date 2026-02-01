"""
Utility Functions for LABS Solver

Includes:
- Bitstring/sequence conversion
- Bit-packing for cache-efficient storage
- Interaction index computation for quantum circuits
"""

from typing import List, Tuple
from math import floor, sin, cos, pi


def bitstring_to_sequence(bitstring: str) -> List[int]:
    """
    Convert a bitstring (0/1) to a sequence (+1/-1).

    Convention: 0 -> +1, 1 -> -1

    Args:
        bitstring: String of '0' and '1' characters

    Returns:
        List of +1/-1 values
    """
    return [1 if b == '0' else -1 for b in bitstring]


def sequence_to_bitstring(sequence: List[int]) -> str:
    """
    Convert a sequence (+1/-1) to a bitstring (0/1).

    Convention: +1 -> '0', -1 -> '1'

    Args:
        sequence: List of +1/-1 values

    Returns:
        String of '0' and '1' characters
    """
    return ''.join('0' if s == 1 else '1' for s in sequence)


def pack_sequence(sequence: List[int]) -> int:
    """
    Pack a binary sequence into a single integer for efficient storage.

    Uses 1 bit per element: +1 -> 0, -1 -> 1
    Supports sequences up to length 64.

    Args:
        sequence: List of +1/-1 values (length <= 64)

    Returns:
        Packed integer representation
    """
    if len(sequence) > 64:
        raise ValueError(f"Sequence length {len(sequence)} exceeds maximum 64")

    packed = 0
    for i, s in enumerate(sequence):
        if s == -1:
            packed |= (1 << i)
    return packed


def unpack_sequence(packed: int, length: int) -> List[int]:
    """
    Unpack an integer back to a binary sequence.

    Args:
        packed: Packed integer representation
        length: Length of the original sequence

    Returns:
        List of +1/-1 values
    """
    sequence = []
    for i in range(length):
        if (packed >> i) & 1:
            sequence.append(-1)
        else:
            sequence.append(1)
    return sequence


def get_interactions(n: int) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Generate interaction indices G2 (2-body) and G4 (4-body) for LABS Hamiltonian.

    These indices define which qubits interact in the counteradiabatic circuit.

    From the Trotterized evolution equation:
    - G2: pairs [i, i+k] for 2-body ZZ interactions
    - G4: quads [i, i+t, i+k, i+k+t] for 4-body ZZZZ interactions

    Args:
        n: Sequence length (number of qubits)

    Returns:
        Tuple of (G2, G4) where:
        - G2: List of [i, j] pairs (0-indexed)
        - G4: List of [i, j, k, l] quads (0-indexed)
    """
    G2 = []
    G4 = []

    # G2: Two-body terms
    # From equation: prod_{i=1}^{N-2} prod_{k=1}^{floor((N-i)/2)}
    # Converting to 0-based indexing
    for i in range(n - 2):  # i from 0 to N-3
        i_1based = i + 1
        max_k = floor((n - i_1based) / 2)
        for k in range(1, max_k + 1):
            j = i + k
            G2.append([i, j])

    # G4: Four-body terms
    # From equation: prod_{i=1}^{N-3} prod_{t=1}^{floor((N-i-1)/2)} prod_{k=t+1}^{N-i-t}
    for i in range(n - 3):  # i from 0 to N-4
        i_1based = i + 1
        max_t = floor((n - i_1based - 1) / 2)
        for t in range(1, max_t + 1):
            max_k = n - i_1based - t
            for k in range(t + 1, max_k + 1):
                # Four qubit indices (0-based)
                idx0 = i
                idx1 = i + t
                idx2 = i + k
                idx3 = i + k + t
                G4.append([idx0, idx1, idx2, idx3])

    return G2, G4


def compute_topology_overlaps(G2: List[List[int]], G4: List[List[int]]) -> dict:
    """
    Compute topological invariants for Gamma2 calculation.

    Args:
        G2: 2-body interaction indices
        G4: 4-body interaction indices

    Returns:
        Dictionary with keys '22', '44', '24' containing overlap counts
    """
    def count_matches(list_a, list_b):
        set_b = set(tuple(sorted(x)) for x in list_b)
        return sum(1 for item in list_a if tuple(sorted(item)) in set_b)

    return {
        '22': count_matches(G2, G2),  # Self overlap = len(G2)
        '44': count_matches(G4, G4),  # Self overlap = len(G4)
        '24': 0,  # 2-body vs 4-body overlap usually 0
    }


def compute_theta(
    t: float,
    dt: float,
    total_time: float,
    n: int,
    G2: List[List[int]],
    G4: List[List[int]],
) -> float:
    """
    Compute theta parameter for counteradiabatic evolution.

    Uses sinusoidal annealing schedule: lambda(t) = sin^2(pi * t / 2T)

    Args:
        t: Current time
        dt: Time step
        total_time: Total evolution time T
        n: Number of qubits
        G2: 2-body interaction indices
        G4: 4-body interaction indices

    Returns:
        Theta value for this time step
    """
    if total_time == 0:
        return 0.0

    # Sinusoidal schedule
    arg = (pi * t) / (2.0 * total_time)
    lam = sin(arg) ** 2
    lam_dot = (pi / (2.0 * total_time)) * sin((pi * t) / total_time)

    # Gamma 1 (Eq 16) - assumes h^x = 1, h^b = 0
    term_g1_2 = 16 * len(G2) * 2
    term_g1_4 = 64 * len(G4) * 4
    Gamma1 = term_g1_2 + term_g1_4

    # Gamma 2 (Eq 17)
    sum_G2 = len(G2) * (lam ** 2 * 2)
    sum_G4 = 4 * len(G4) * (16 * (lam ** 2) + 8 * ((1 - lam) ** 2))

    I_vals = compute_topology_overlaps(G2, G4)
    term_topology = 4 * (lam ** 2) * (4 * I_vals['24'] + I_vals['22']) + 64 * (lam ** 2) * I_vals['44']

    Gamma2 = -256 * (term_topology + sum_G2 + sum_G4)

    # Alpha & Theta
    if abs(Gamma2) < 1e-12:
        alpha = 0.0
    else:
        alpha = -Gamma1 / Gamma2

    return dt * alpha * lam_dot
