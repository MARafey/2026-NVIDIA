"""
Memetic Tabu Search (MTS) for LABS Problem

Implements the classical MTS algorithm which combines:
- Population-based evolutionary search
- Local search via tabu search
- Crossover and mutation operators

Reference: "Scaling advantage with quantum-enhanced memetic tabu search"
"""

import random
from typing import List, Tuple, Optional
import numpy as np

from .energy import calculate_energy


def random_sequence(n: int) -> List[int]:
    """
    Generate a random binary sequence of length N.

    Args:
        n: Sequence length

    Returns:
        List of +1/-1 values
    """
    return [random.choice([1, -1]) for _ in range(n)]


def combine(parent1: List[int], parent2: List[int]) -> List[int]:
    """
    Combine two parent sequences using uniform crossover.

    Each position is randomly selected from either parent.

    Args:
        parent1: First parent sequence
        parent2: Second parent sequence

    Returns:
        Child sequence
    """
    return [random.choice([p1, p2]) for p1, p2 in zip(parent1, parent2)]


def mutate(sequence: List[int], p_mutate: float = 0.1) -> List[int]:
    """
    Mutate a sequence by flipping each bit with given probability.

    Args:
        sequence: Input sequence
        p_mutate: Probability of flipping each bit

    Returns:
        Mutated sequence (new list)
    """
    return [s if random.random() > p_mutate else -s for s in sequence]


def tabu_search(
    sequence: List[int],
    max_iterations: int = 100,
    tabu_tenure: int = 7,
) -> Tuple[List[int], float]:
    """
    Perform tabu search starting from the given sequence.

    Tabu search is a local search method that keeps a "tabu list" of recently
    visited moves to avoid cycling back to the same solutions.

    Args:
        sequence: Initial sequence to start search from
        max_iterations: Maximum number of iterations
        tabu_tenure: How long a move stays in the tabu list

    Returns:
        Tuple of (best_sequence, best_energy)
    """
    n = len(sequence)
    current = sequence.copy()
    current_energy = calculate_energy(current)

    best = current.copy()
    best_energy = current_energy

    # Tabu list: position -> iteration when it becomes non-tabu
    tabu_list = {}

    for iteration in range(max_iterations):
        # Find the best non-tabu move (flip a single bit)
        best_move = None
        best_move_energy = float('inf')

        for i in range(n):
            # Try flipping bit i
            current[i] = -current[i]
            new_energy = calculate_energy(current)
            current[i] = -current[i]  # Flip back

            # Check if move is tabu (unless it gives a new best solution)
            is_tabu = tabu_list.get(i, 0) > iteration
            aspiration = new_energy < best_energy  # Aspiration criterion

            if (not is_tabu or aspiration) and new_energy < best_move_energy:
                best_move = i
                best_move_energy = new_energy

        if best_move is None:
            break  # No valid moves

        # Apply the best move
        current[best_move] = -current[best_move]
        current_energy = best_move_energy

        # Update tabu list
        tabu_list[best_move] = iteration + tabu_tenure

        # Update best solution if improved
        if current_energy < best_energy:
            best = current.copy()
            best_energy = current_energy

    return best, best_energy


def memetic_tabu_search(
    n: int,
    population_size: int = 20,
    max_generations: int = 50,
    p_mutate: float = 0.1,
    tabu_iterations: int = 100,
    tabu_tenure: int = 7,
    initial_population: Optional[List[List[int]]] = None,
    verbose: bool = False,
) -> Tuple[List[int], float, List[List[int]], List[float]]:
    """
    Memetic Tabu Search for the LABS problem.

    Combines evolutionary search with local tabu search refinement.

    Args:
        n: Sequence length
        population_size: Size of the population
        max_generations: Maximum number of generations
        p_mutate: Mutation probability
        tabu_iterations: Max iterations for each tabu search
        tabu_tenure: Tabu tenure for tabu search
        initial_population: Optional pre-generated population (e.g., from quantum)
        verbose: Print progress information

    Returns:
        Tuple of (best_sequence, best_energy, final_population, final_energies)
    """
    # Initialize population
    if initial_population is not None:
        population = [seq.copy() for seq in initial_population[:population_size]]
        # Fill remaining with random if needed
        while len(population) < population_size:
            population.append(random_sequence(n))
    else:
        population = [random_sequence(n) for _ in range(population_size)]

    energies = [calculate_energy(seq) for seq in population]

    # Find initial best
    best_idx = int(np.argmin(energies))
    best_sequence = population[best_idx].copy()
    best_energy = energies[best_idx]

    if verbose:
        print(f"Initial best energy: {best_energy}")

    for generation in range(max_generations):
        # Select parents (tournament selection)
        idx1, idx2 = random.sample(range(population_size), 2)
        parent1 = population[idx1]
        parent2 = population[idx2]

        # Create child through combination
        child = combine(parent1, parent2)

        # Mutate with probability
        if random.random() < p_mutate:
            child = mutate(child, p_mutate=0.1)

        # Apply tabu search to child
        improved_child, child_energy = tabu_search(
            child, max_iterations=tabu_iterations, tabu_tenure=tabu_tenure
        )

        # Update population if child is better than worst member
        worst_idx = int(np.argmax(energies))
        if child_energy < energies[worst_idx]:
            population[worst_idx] = improved_child
            energies[worst_idx] = child_energy

        # Update global best
        if child_energy < best_energy:
            best_sequence = improved_child.copy()
            best_energy = child_energy
            if verbose:
                print(f"Generation {generation}: new best energy = {best_energy}")

    return best_sequence, best_energy, population, energies
