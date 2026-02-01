"""
Quantum Counteradiabatic Optimization for LABS using CUDA-Q

Implements the Trotterized counteradiabatic circuit from:
"Scaling advantage with quantum-enhanced memetic tabu search for LABS"

Key components:
- 2-qubit R_YZ and R_ZY gates
- 4-qubit R_YZZZ, R_ZYZZ, R_ZZYZ, R_ZZZY gates
- Full Trotterized evolution circuit
- Quantum-seeded population generation
"""

from typing import List, Tuple, Optional
import numpy as np

try:
    import cudaq
    CUDAQ_AVAILABLE = True
except ImportError:
    CUDAQ_AVAILABLE = False
    print("Warning: CUDA-Q not available. Quantum functions will not work.")

from .energy import calculate_energy
from .mts import memetic_tabu_search, random_sequence
from .utils import bitstring_to_sequence, get_interactions, compute_theta


# =============================================================================
# CUDA-Q Kernels
# =============================================================================

if CUDAQ_AVAILABLE:

    @cudaq.kernel
    def rzz(q0: cudaq.qubit, q1: cudaq.qubit, theta: float):
        """
        Apply R_ZZ(theta) gate: exp(-i * theta/2 * Z_0 Z_1)
        Implementation: CNOT - RZ - CNOT
        """
        x.ctrl(q0, q1)
        rz(theta, q1)
        x.ctrl(q0, q1)

    @cudaq.kernel
    def r_yz(q0: cudaq.qubit, q1: cudaq.qubit, theta: float):
        """
        Apply R_YZ(theta) gate: exp(-i * theta/2 * Y_0 Z_1)
        RX(pi/2) on q0 -> RZZ(theta) -> RX(-pi/2) on q0
        """
        rx(np.pi / 2, q0)
        rzz(q0, q1, theta)
        rx(-np.pi / 2, q0)

    @cudaq.kernel
    def r_zy(q0: cudaq.qubit, q1: cudaq.qubit, theta: float):
        """
        Apply R_ZY(theta) gate: exp(-i * theta/2 * Z_0 Y_1)
        RX(pi/2) on q1 -> RZZ(theta) -> RX(-pi/2) on q1
        """
        rx(np.pi / 2, q1)
        rzz(q0, q1, theta)
        rx(-np.pi / 2, q1)

    @cudaq.kernel
    def r_yzzz(q0: cudaq.qubit, q1: cudaq.qubit, q2: cudaq.qubit, q3: cudaq.qubit, theta: float):
        """
        Apply R_YZZZ(theta) gate: exp(-i * theta/2 * Y_0 Z_1 Z_2 Z_3)
        """
        rx(np.pi / 2, q0)
        x.ctrl(q0, q1)
        x.ctrl(q1, q2)
        x.ctrl(q2, q3)
        rz(theta, q3)
        x.ctrl(q2, q3)
        x.ctrl(q1, q2)
        x.ctrl(q0, q1)
        rx(-np.pi / 2, q0)

    @cudaq.kernel
    def r_zyzz(q0: cudaq.qubit, q1: cudaq.qubit, q2: cudaq.qubit, q3: cudaq.qubit, theta: float):
        """
        Apply R_ZYZZ(theta) gate: exp(-i * theta/2 * Z_0 Y_1 Z_2 Z_3)
        """
        rx(np.pi / 2, q1)
        x.ctrl(q0, q1)
        x.ctrl(q1, q2)
        x.ctrl(q2, q3)
        rz(theta, q3)
        x.ctrl(q2, q3)
        x.ctrl(q1, q2)
        x.ctrl(q0, q1)
        rx(-np.pi / 2, q1)

    @cudaq.kernel
    def r_zzyz(q0: cudaq.qubit, q1: cudaq.qubit, q2: cudaq.qubit, q3: cudaq.qubit, theta: float):
        """
        Apply R_ZZYZ(theta) gate: exp(-i * theta/2 * Z_0 Z_1 Y_2 Z_3)
        """
        rx(np.pi / 2, q2)
        x.ctrl(q0, q1)
        x.ctrl(q1, q2)
        x.ctrl(q2, q3)
        rz(theta, q3)
        x.ctrl(q2, q3)
        x.ctrl(q1, q2)
        x.ctrl(q0, q1)
        rx(-np.pi / 2, q2)

    @cudaq.kernel
    def r_zzzy(q0: cudaq.qubit, q1: cudaq.qubit, q2: cudaq.qubit, q3: cudaq.qubit, theta: float):
        """
        Apply R_ZZZY(theta) gate: exp(-i * theta/2 * Z_0 Z_1 Z_2 Y_3)
        """
        rx(np.pi / 2, q3)
        x.ctrl(q0, q1)
        x.ctrl(q1, q2)
        x.ctrl(q2, q3)
        rz(theta, q3)
        x.ctrl(q2, q3)
        x.ctrl(q1, q2)
        x.ctrl(q0, q1)
        rx(-np.pi / 2, q3)

    @cudaq.kernel
    def trotterized_circuit(
        n: int,
        G2: list[list[int]],
        G4: list[list[int]],
        steps: int,
        dt: float,
        T: float,
        thetas: list[float],
    ):
        """
        Complete Trotterized circuit for counteradiabatic LABS optimization.

        Args:
            n: Number of qubits (sequence length)
            G2: List of 2-body interaction indices [[i, j], ...]
            G4: List of 4-body interaction indices [[i, j, k, l], ...]
            steps: Number of Trotter steps
            dt: Time step
            T: Total evolution time
            thetas: List of theta values for each Trotter step
        """
        # Allocate qubits
        reg = cudaq.qvector(n)

        # Initialize in |+>^N state (ground state of H_i = sum_i sigma_x)
        h(reg)

        # Apply Trotter steps
        for step in range(steps):
            theta = thetas[step]

            # Apply 2-body terms: R_YZ and R_ZY for each pair in G2
            theta_2body = 4.0 * theta

            for pair in G2:
                i = pair[0]
                j = pair[1]
                r_yz(reg[i], reg[j], theta_2body)
                r_zy(reg[i], reg[j], theta_2body)

            # Apply 4-body terms
            theta_4body = 8.0 * theta

            for quad in G4:
                i0 = quad[0]
                i1 = quad[1]
                i2 = quad[2]
                i3 = quad[3]
                r_yzzz(reg[i0], reg[i1], reg[i2], reg[i3], theta_4body)
                r_zyzz(reg[i0], reg[i1], reg[i2], reg[i3], theta_4body)
                r_zzyz(reg[i0], reg[i1], reg[i2], reg[i3], theta_4body)
                r_zzzy(reg[i0], reg[i1], reg[i2], reg[i3], theta_4body)


# =============================================================================
# Quantum Sampling Functions
# =============================================================================

def sample_quantum_population(
    n: int,
    population_size: int,
    n_steps: int = 1,
    T: float = 1.0,
    shots: Optional[int] = None,
) -> List[List[int]]:
    """
    Sample a population from the quantum counteradiabatic circuit.

    Args:
        n: Sequence length
        population_size: Number of sequences to return
        n_steps: Number of Trotter steps
        T: Total evolution time
        shots: Number of shots (default: 2 * population_size)

    Returns:
        List of sequences (each sequence is a list of +1/-1)
    """
    if not CUDAQ_AVAILABLE:
        raise RuntimeError("CUDA-Q is not available")

    if shots is None:
        shots = population_size * 2

    dt = T / n_steps
    G2, G4 = get_interactions(n)

    # Compute thetas for each step
    thetas = []
    for step in range(1, n_steps + 1):
        t = step * dt
        theta_val = compute_theta(t, dt, T, n, G2, G4)
        thetas.append(theta_val)

    # Sample from the circuit
    result = cudaq.sample(
        trotterized_circuit, n, G2, G4, n_steps, dt, T, thetas,
        shots_count=shots
    )

    # Convert samples to sequences
    population = []
    for bitstring in result.keys():
        if len(population) >= population_size:
            break
        seq = bitstring_to_sequence(bitstring)
        population.append(seq)

    # Fill remaining with random if needed
    while len(population) < population_size:
        population.append(random_sequence(n))

    return population


def quantum_enhanced_mts(
    n: int,
    population_size: int = 20,
    max_generations: int = 50,
    p_mutate: float = 0.1,
    n_trotter_steps: int = 1,
    T: float = 1.0,
    tabu_iterations: int = 100,
    tabu_tenure: int = 7,
    verbose: bool = False,
) -> Tuple[List[int], float, List[List[int]], List[float]]:
    """
    Quantum-Enhanced Memetic Tabu Search.

    Uses quantum sampling to initialize the population, then runs MTS.

    Args:
        n: Sequence length
        population_size: Size of the population
        max_generations: Maximum number of generations
        p_mutate: Mutation probability
        n_trotter_steps: Number of Trotter steps for quantum circuit
        T: Total evolution time for quantum circuit
        tabu_iterations: Max iterations for each tabu search
        tabu_tenure: Tabu tenure
        verbose: Print progress information

    Returns:
        Tuple of (best_sequence, best_energy, population, energies)
    """
    if verbose:
        print(f"Sampling quantum population for N={n}...")

    # Get quantum-seeded population
    quantum_population = sample_quantum_population(
        n, population_size, n_trotter_steps, T
    )

    if verbose:
        energies = [calculate_energy(seq) for seq in quantum_population]
        print(f"Initial quantum population: min={min(energies):.1f}, mean={np.mean(energies):.1f}")

    # Run MTS with quantum-seeded population
    return memetic_tabu_search(
        n=n,
        population_size=population_size,
        max_generations=max_generations,
        p_mutate=p_mutate,
        tabu_iterations=tabu_iterations,
        tabu_tenure=tabu_tenure,
        initial_population=quantum_population,
        verbose=verbose,
    )
