"""
LABS Solver - Quantum-Enhanced Memetic Tabu Search for LABS Problem

This package implements a hybrid quantum-classical solver for the
Low Autocorrelation Binary Sequences (LABS) problem using:
- Counteradiabatic quantum optimization with CUDA-Q
- Memetic Tabu Search (MTS) for classical refinement
- Shared exploration cache for GPU acceleration (Phase 2)
"""

from .energy import calculate_energy, compute_autocorrelation
from .mts import (
    memetic_tabu_search,
    tabu_search,
    random_sequence,
    combine,
    mutate,
)
from .quantum import (
    get_interactions,
    sample_quantum_population,
    quantum_enhanced_mts,
)
from .utils import (
    bitstring_to_sequence,
    sequence_to_bitstring,
    pack_sequence,
    unpack_sequence,
)

__version__ = "0.1.0"
__all__ = [
    "calculate_energy",
    "compute_autocorrelation",
    "memetic_tabu_search",
    "tabu_search",
    "random_sequence",
    "combine",
    "mutate",
    "get_interactions",
    "sample_quantum_population",
    "quantum_enhanced_mts",
    "bitstring_to_sequence",
    "sequence_to_bitstring",
    "pack_sequence",
    "unpack_sequence",
]
