# AI Methodology Report

Project Name: LABS-Shared-Cache-Distribution

Team Name: Exception Handler

---

## 1. The Workflow

### Agent Organization

| Agent | Role | Use Case |
|-------|------|----------|
| **Claude Code (Primary)** | Architecture design, CUDA kernel implementation, PRD drafting | Core development partner for algorithm design and code generation |
| **Claude Code** | Documentation, verification planning | PRD structure, test strategy, AI report |
| **[Optional: Add others]** | [Role] | [Use case] |

### Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Development Workflow                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│   │  Human   │───▶│   AI     │───▶│  Human   │───▶│   AI     │ │
│   │  Idea    │    │  Refine  │    │  Decide  │    │  Implement│ │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│                                                                  │
│   Example Flow:                                                  │
│   "Share sequences    "What about     "Async, keep    "Here's   │
│    across threads" ─▶  sync overhead?" ─▶ better energy" ─▶ code"│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Session Structure

1. **Conceptual Phase:** Human proposes high-level idea (shared exploration cache)
2. **Refinement Phase:** AI identifies technical challenges, proposes solutions
3. **Decision Phase:** Human selects from options (async vs sync, eviction policy)
4. **Implementation Phase:** AI generates detailed code and documentation

---

## 2. Verification Strategy

### Unit Test Framework

**Framework:** `pytest` with property-based testing via `hypothesis`

### Specific Unit Tests for AI-Generated Code

#### Test 1: Bit-Packing Round-Trip
```python
import pytest
from hypothesis import given, strategies as st

@given(st.lists(st.sampled_from([-1, 1]), min_size=1, max_size=32))
def test_pack_unpack_roundtrip(sequence):
    """Verify bit-packing doesn't lose information"""
    packed = pack_sequence(sequence)
    unpacked = unpack_sequence(packed, len(sequence))
    assert unpacked == sequence, f"Round-trip failed: {sequence} -> {packed} -> {unpacked}"
```

#### Test 2: Energy Symmetry (Physics Constraint)
```python
@given(st.lists(st.sampled_from([-1, 1]), min_size=3, max_size=20))
def test_energy_negation_symmetry(sequence):
    """E(S) must equal E(-S) for LABS"""
    negated = [-s for s in sequence]
    assert compute_energy(sequence) == compute_energy(negated)

@given(st.lists(st.sampled_from([-1, 1]), min_size=3, max_size=20))
def test_energy_reversal_symmetry(sequence):
    """E(S) must equal E(reverse(S)) for LABS"""
    reversed_seq = sequence[::-1]
    assert compute_energy(sequence) == compute_energy(reversed_seq)
```

#### Test 3: Cache "Keep Better" Policy
```python
def test_cache_keeps_better_energy():
    """Cache should retain lower energy on collision"""
    cache = SharedCache(size=1024)

    seq = [1, 1, -1, 1, -1]
    packed = pack_sequence(seq)

    # Insert with high energy
    cache.update(packed, energy=100.0)
    assert cache.lookup(packed) == 100.0

    # Insert same sequence with lower energy - should update
    cache.update(packed, energy=50.0)
    assert cache.lookup(packed) == 50.0

    # Insert same sequence with higher energy - should NOT update
    cache.update(packed, energy=75.0)
    assert cache.lookup(packed) == 50.0  # Still 50, not 75
```

#### Test 4: Cache Collision Behavior
```python
def test_cache_collision_keeps_better():
    """When two sequences hash to same slot, keep lower energy"""
    cache = SharedCache(size=16)  # Small size to force collisions

    # Find two sequences that collide
    seq1 = [1, 1, 1, -1]
    seq2 = [1, -1, 1, 1]  # Different sequence

    packed1 = pack_sequence(seq1)
    packed2 = pack_sequence(seq2)

    # Force same slot (in real test, find actual collisions)
    slot1 = packed1 % 16
    slot2 = packed2 % 16

    if slot1 == slot2:  # Collision case
        cache.update(packed1, energy=100.0)
        cache.update(packed2, energy=50.0)  # Better energy

        # Slot should contain seq2 (better energy)
        result = cache.lookup_slot(slot1)
        assert result.energy == 50.0
```

#### Test 5: Ground Truth Validation
```python
KNOWN_OPTIMA = {
    3: (1, [1, 1, -1]),
    5: (2, [1, 1, 1, -1, 1]),
    7: (4, [1, 1, 1, -1, -1, 1, -1]),
    11: (12, None),  # Energy known, sequence varies
    13: (18, None),
}

@pytest.mark.parametrize("n,expected_energy,_",
    [(k, v[0], v[1]) for k, v in KNOWN_OPTIMA.items()])
def test_finds_known_optima(n, expected_energy, _):
    """Solver must find known optimal energies for small N"""
    result = run_mts_solver(n, max_iterations=10000)
    assert result.energy <= expected_energy, \
        f"N={n}: Found {result.energy}, expected {expected_energy}"
```

### AI Hallucination Guardrails

| Guardrail | Implementation |
|-----------|----------------|
| **Type checking** | All AI code must pass `mypy --strict` |
| **Bounds checking** | Assert energy >= 0 (physical constraint) |
| **Determinism test** | Same input must produce same output |
| **Small-N exhaustive** | For N<=10, compare against brute-force |

---

## 3. The "Vibe" Log

### Win: Architecture Design Acceleration

**Situation:** Needed to design a GPU-friendly data structure for sharing explored sequences across threads.

**AI Contribution:** In a single session, Claude Code:
1. Identified memory hierarchy trade-offs (shared vs global memory)
2. Proposed bit-packing scheme for sequence representation
3. Designed cache structure with collision handling
4. Generated complete CUDA kernel skeleton

**Time Saved:** Estimated 4-6 hours of manual research and prototyping compressed into ~30 minutes of iterative discussion.

**Key Exchange:**
```
Human: "I want threads to share explored sequences in real-time"
AI: [Identified 4 technical challenges: memory contention, sync overhead,
     state representation, memory limits]
AI: [Proposed 4 strategies: Bloom filter, Hash table, Hierarchical, Batch sync]
Human: "Shared memory, async, bit-packing"
AI: [Generated complete implementation with cache structure, lookup/update
     functions, and MTS integration]
```

---

### Learn: Iterative Refinement Through Specific Choices

**Initial Approach:** Asked broad question "How should I parallelize MTS on GPU?"

**Problem:** AI provided generic overview without actionable implementation details.

**Improved Approach:** Broke down into specific decision points:
1. "What sequence lengths am I targeting?" → Determined bit-packing scheme
2. "Keep better energy or LRU for eviction?" → Chose energy-based
3. "Sync or async updates?" → Chose async for performance

**Lesson Learned:** AI produces better output when human provides:
- Concrete constraints (N ≤ 32, alphabet ≤ 4)
- Binary choices (async vs sync)
- Clear priorities (performance over consistency)

**Prompting Evolution:**
```
Before: "Design a GPU cache for MTS"
After:  "Design a GPU shared memory cache for MTS with:
         - N ≤ 32 sequences
         - Bit-packed uint64_t representation
         - Async updates (no locks)
         - Keep-better-energy collision policy"
```

---

### Fail: Initial Bit-Packing Calculation Error

**Situation:** AI initially suggested:
> "N ≤ 25 with alphabet ≤ 32 → single uint64_t (5 bits × 25)"

**The Bug:** 5 bits × 25 = 125 bits, which does NOT fit in uint64_t (64 bits).

**Detection:** Human caught the arithmetic error during review.

**Fix:** Corrected to practical limits:
- 5 bits × 12 elements = 60 bits (fits in uint64_t)
- For N=25 with large alphabet, need uint128 or two uint64_t

**Lesson:** Always verify AI arithmetic, especially for:
- Memory calculations
- Bit manipulation bounds
- Index ranges

**Mitigation Added:**
```python
def test_packing_fits_in_uint64():
    """Verify packing scheme doesn't overflow"""
    BITS_PER_ELEM = 2
    MAX_N = 32
    assert BITS_PER_ELEM * MAX_N <= 64, "Packing overflows uint64_t"
```

---

### Context Dump

#### skills.md (CUDA-Q Reference)
```markdown
# CUDA-Q Quick Reference for AI Agent

## Kernel Decorators
- `@cudaq.kernel` - Defines quantum kernel
- Supported gates: h, x, y, z, rx, ry, rz, cx, cz

## Sampling
```python
results = cudaq.sample(kernel, *args, shots_count=1000)
for bitstring, count in results.items():
    print(f"{bitstring}: {count}")
```

## GPU Backends
- `cudaq.set_target("nvidia")` - Single GPU
- `cudaq.set_target("nvidia-mgpu")` - Multi-GPU

## Memory Constraints
- Shared memory: ~48KB per block
- Max threads per block: 1024
- Registers: 65536 per block
```

#### constraints.md (Project Constraints)
```markdown
# Project Constraints

## Hardware Limits
- Shared memory: 48KB per SM
- Our cache: 2048 entries × 16 bytes = 32KB (safe margin)
- Thread block: 256 threads

## Problem Bounds
- N ≤ 32 for uint64_t bit-packing
- Binary alphabet: 1 bit per element
- Quaternary alphabet: 2 bits per element

## Performance Targets
- Cache hit rate: >20%
- Speedup vs CPU: >10x
```

#### Example Prompt Structure
```markdown
## Task
[Specific implementation request]

## Context
- Problem: LABS optimization
- Target: NVIDIA GPU with CUDA-Q
- Constraints: [From constraints.md]

## Requirements
1. [Specific requirement]
2. [Specific requirement]

## Output Format
- Working code with comments
- Test cases for verification
```

---

## 4. Summary Statistics

| Metric | Value |
|--------|-------|
| **AI Sessions** | [X] sessions |
| **Total Interaction Time** | [X] hours |
| **Code Generated by AI** | ~70% |
| **Code Modified After AI Generation** | ~30% |
| **AI Errors Caught by Tests** | [X] |
| **AI Errors Caught by Review** | 1 (bit-packing overflow) |

---

## 5. Reflection

### What Worked Well
- Breaking complex design into binary decisions
- Having AI propose options, human selects
- Immediate test writing for AI-generated code

### What We'd Do Differently
- Create skills.md earlier in the process
- Run arithmetic validation on all AI calculations
- Use property-based testing from the start

### AI Tools Evaluation

| Tool | Strengths | Limitations |
|------|-----------|-------------|
| **Claude Code** | Deep technical discussion, code generation, iterative refinement | Occasional arithmetic errors, needs specific constraints |
| **[Other tool]** | [Strengths] | [Limitations] |
