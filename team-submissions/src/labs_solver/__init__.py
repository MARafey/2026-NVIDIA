"""
LABS Solver - Quantum-Enhanced Memetic Tabu Search for LABS Problem

This package implements a hybrid quantum-classical solver for the
Low Autocorrelation Binary Sequences (LABS) problem using:
- Counteradiabatic quantum optimization with CUDA-Q
- Memetic Tabu Search (MTS) for classical refinement
- Shared exploration cache for GPU acceleration (Phase 2)
"""

from .energy import calculate_energy, compute_autocorrelation, merit_factor
from .mts import (
    memetic_tabu_search,
    tabu_search,
    random_sequence,
    combine,
    mutate,
)
from .utils import (
    bitstring_to_sequence,
    sequence_to_bitstring,
    pack_sequence,
    unpack_sequence,
    get_interactions,
)

# Conditional imports for quantum (requires CUDA-Q)
try:
    from .quantum import (
        sample_quantum_population,
        quantum_enhanced_mts,
        CUDAQ_AVAILABLE,
    )
except ImportError:
    CUDAQ_AVAILABLE = False
    sample_quantum_population = None
    quantum_enhanced_mts = None

# Conditional imports for GPU (requires CuPy)
try:
    from .energy_gpu import (
        calculate_energy_gpu,
        calculate_energy_batch_gpu,
        GPUEnergyCache,
        CUPY_AVAILABLE,
    )
    from .mts_gpu import (
        memetic_tabu_search_gpu,
        tabu_search_gpu,
        benchmark_cpu_vs_gpu,
    )
except ImportError:
    CUPY_AVAILABLE = False
    calculate_energy_gpu = None
    calculate_energy_batch_gpu = None
    GPUEnergyCache = None
    memetic_tabu_search_gpu = None
    tabu_search_gpu = None
    benchmark_cpu_vs_gpu = None

__version__ = "0.2.0"
__all__ = [
    # Core
    "calculate_energy",
    "compute_autocorrelation",
    "merit_factor",
    # MTS
    "memetic_tabu_search",
    "tabu_search",
    "random_sequence",
    "combine",
    "mutate",
    # Quantum
    "get_interactions",
    "sample_quantum_population",
    "quantum_enhanced_mts",
    "CUDAQ_AVAILABLE",
    # Utils
    "bitstring_to_sequence",
    "sequence_to_bitstring",
    "pack_sequence",
    "unpack_sequence",
    # GPU
    "calculate_energy_gpu",
    "calculate_energy_batch_gpu",
    "GPUEnergyCache",
    "memetic_tabu_search_gpu",
    "tabu_search_gpu",
    "benchmark_cpu_vs_gpu",
    "CUPY_AVAILABLE",
]
