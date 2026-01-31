# Product Requirements Document (PRD)

**Project Name:** `LABS-Shared-Cache-Distribution`

**Team Name:** `Exception Handler`

**GitHub Repository:** https://github.com/MARafey/2026-NVIDIA

---

## 1. Team Roles

**Working Solo :*)**

---

## 2. The Architecture
**Owner:** Project Lead

### Choice of Quantum Algorithm

* **Algorithm:** Counteradiabatic Quantum Optimization with Trotterized Evolution

* **Motivation:**
    * The counteradiabatic approach provides more efficient gate counts than QAOA for the LABS problem by using an auxiliary gauge potential to suppress diabatic transitions during evolution.
    * Research demonstrates O(1.24^N) scaling versus O(1.34^N) for classical-only MTS, providing a concrete quantum advantage target.
    * The algorithm produces high-quality samples that serve as seeds for classical refinement, fitting naturally into our hybrid pipeline.

### Hybrid Architecture: Shared Exploration Cache

Our key innovation is a **collaborative parallel search with shared exploration state** on the GPU:

```
┌────────────────── Thread Block ──────────────────┐
│  ┌─────────────────────────────────────────────┐ │
│  │      Shared Memory Cache (~48KB)            │ │
│  │  ┌─────────────────────────────────────┐    │ │
│  │  │  Hash Table: packed_seq → energy    │    │ │
│  │  │  Tiered: Protected Seeds + Organic  │    │ │
│  │  └─────────────────────────────────────┘    │ │
│  └─────────────────────────────────────────────┘ │
│       ▲ async      ▲ async      ▲ async          │
│    Thread 0    Thread 1    Thread 2   ...        │
└──────────────────────────────────────────────────┘
```

**Core Concept:** All threads within a block share discovered sequences and their energies in real-time. When a thread evaluates a neighbor:
1. Check cache first - if hit, skip computation entirely
2. If miss, compute energy and broadcast to cache for others
3. Async updates with "keep better energy" collision policy

This eliminates redundant computation as threads converge toward similar high-quality regions.

### Literature Review

* **Reference:** "Scaling advantage in approximate optimization with quantum enhanced Memetic Tabu Search" - Kipu Quantum, University of Basque Country, NVIDIA
* **Relevance:** Provides the theoretical foundation for our counteradiabatic approach and establishes the O(1.24^N) scaling target. Demonstrates that quantum samples improve MTS convergence.

* **Reference:** "Parallel MTS" - JPMorgan Chase
* **Relevance:** Validates GPU parallelization of MTS; our shared cache approach extends this by adding inter-thread collaboration.

---

## 3. The Acceleration Strategy
**Owner:** GPU Acceleration PIC

### Quantum Acceleration (CUDA-Q)

* **Strategy:**
    * Use `nvidia` GPU backend for quantum circuit simulation
    * Implement counteradiabatic kernel with Trotterized 2-qubit (ZZ) and 4-qubit interactions
    * Generate 1024+ samples per run to seed the classical search
    * For larger N, target `nvidia-mgpu` backend to distribute simulation across multiple GPUs

* **Sample Generation Pipeline:**
    ```
    CUDA-Q Circuit → Measurement Samples → Sort by Energy → Top-K Seeds → GPU MTS
    ```

### Classical Acceleration (MTS)

* **Strategy: Shared Memory Exploration Cache**

| Component | Implementation |
|-----------|----------------|
| **Bit-Packing** | Sequences encoded as `uint64_t` (2-bit per element for alphabet ≤ 4, supporting N ≤ 32) |
| **Cache Structure** | 2048 entries × 16 bytes = 32KB in shared memory |
| **Lookup** | O(1) hash-based lookup before every energy computation |
| **Update Policy** | Async writes, keep lower energy on collision |
| **Initialization** | Tiered: 25% protected quantum seeds + 75% organic growth |

* **Expected Speedup Sources:**
    1. **Cache hits:** Eliminate redundant energy calculations (estimated 10-50% hit rate as search converges)
    2. **Bit-packing:** Fast sequence comparison via integer operations
    3. **Shared memory:** ~100x bandwidth vs global memory
    4. **Parallel blocks:** 128+ blocks exploring different regions simultaneously

### Hardware Targets

* **Dev Environment:** qBraid (CPU) for logic validation and unit tests
* **Initial GPU Testing:** Brev L4 for cache implementation and small-N benchmarks
* **Production Benchmarks:** Brev A100-80GB for final N=32 performance runs

---

## 4. The Verification Plan
**Owner:** Quality Assurance PIC

### Unit Testing Strategy

* **Framework:** `pytest` with `hypothesis` for property-based testing
* **AI Hallucination Guardrails:**
    * All AI-generated CUDA kernels must pass property tests before integration
    * Energy outputs validated against theoretical bounds: E(S) ≥ 0 for all sequences
    * Cache operations tested in isolation before integration with MTS

### Core Correctness Checks

* **Check 1 (Symmetry):**
    * LABS sequence S and its negation -S must have identical energies
    * Test: `assert energy(S) == energy(-S)` for 1000 random sequences

* **Check 2 (Reflection Symmetry):**
    * Reversed sequence must have same energy
    * Test: `assert energy(S) == energy(S[::-1])` for 1000 random sequences

* **Check 3 (Ground Truth - Small N):**
    | N | Known Optimal Energy | Optimal Sequence |
    |---|---------------------|------------------|
    | 3 | 1 | [1, 1, -1] |
    | 5 | 2 | [1, 1, 1, -1, 1] |
    | 7 | 4 | [1, 1, 1, -1, -1, 1, -1] |

    * Test: Verify our solver finds these exact solutions for small N

* **Check 4 (Bit-Packing Round-Trip):**
    * Test: `unpack(pack(S)) == S` for all test sequences
    * Test: `pack(S1) != pack(S2)` when S1 != S2 (no collisions for same-length sequences)

* **Check 5 (Cache Consistency):**
    * After `cache_update(seq, energy)`, subsequent `cache_lookup(seq)` returns correct energy
    * "Keep better" policy: inserting higher energy never overwrites lower energy

### Integration Tests

* **Quantum-Classical Pipeline:**
    * Verify quantum samples have valid format (correct length, binary values)
    * Verify MTS improves upon quantum seed energies (monotonic improvement)

---

## 5. Execution Strategy & Success Metrics
**Owner:** Technical Marketing PIC

### Agentic Workflow

* **IDE:** Claude Code / Cursor with CUDA-Q documentation context
* **Workflow:**
    1. Architect drafts component interface (function signatures, data structures)
    2. AI generates implementation with explicit test requirements
    3. QA PIC runs test suite; failures fed back to AI for iteration
    4. GPU PIC validates performance on target hardware
* **Context Management:**
    * `skills.md` containing CUDA-Q API reference and kernel patterns
    * `constraints.md` documenting shared memory limits and thread block sizes

### Success Metrics

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| **Approximation Ratio** | > 0.95 for N=25 | > 0.98 for N=25 |
| **Cache Hit Rate** | > 20% in converged search | > 40% |
| **Speedup vs CPU MTS** | 10x | 50x |
| **Max Problem Size** | N=28 | N=32 |
| **Quantum Seed Improvement** | MTS improves seed energy by > 10% | > 20% |

### Visualization Plan

* **Plot 1: Cache Effectiveness**
    * X-axis: MTS iteration
    * Y-axis: Cache hit rate (%)
    * Shows how collaboration increases as threads converge

* **Plot 2: Quantum Seed Value**
    * Comparison: Quantum-seeded MTS vs Random-seeded MTS
    * Metric: Iterations to reach target energy

* **Plot 3: Scaling Analysis**
    * X-axis: Problem size N
    * Y-axis: Time to solution (log scale)
    * Compare: CPU baseline, GPU without cache, GPU with shared cache

* **Plot 4: Energy Convergence**
    * X-axis: Wall-clock time
    * Y-axis: Best energy found
    * Multiple traces for different seeding strategies

---

## 6. Resource Management Plan
**Owner:** GPU Acceleration PIC

### Credit Conservation Strategy

| Phase | Environment | GPU Usage | Estimated Cost |
|-------|-------------|-----------|----------------|
| **Logic Development** | qBraid (CPU) | None | Free |
| **Unit Testing** | qBraid (CPU) | None | Free |
| **Initial GPU Port** | Brev L4 | 2-3 hours | ~$2-3 |
| **Cache Tuning** | Brev L4 | 2-3 hours | ~$2-3 |
| **Performance Benchmarks** | Brev A100 | 2 hours | ~$8-10 |
| **Final Runs** | Brev A100 | 1 hour | ~$4-5 |
| **Buffer** | - | - | ~$2 |
| **Total** | - | - | **~$20** |

### Operational Procedures

* All development and debugging done on qBraid CPU until tests pass
* GPU instances manually shutdown during breaks (GPU PIC responsibility)
* Brev instance spinup requires Slack notification to team
* A100 usage reserved for final 3 hours only
* Emergency: If credits drop below $5, stop all GPU work and assess

### Checkpoints

1. **Before L4:** All unit tests pass on CPU
2. **Before A100:** Cache implementation validated on L4, benchmarks planned
3. **Final runs:** Pre-written benchmark scripts, no debugging on A100

---

## Appendix: Technical Specifications

### Cache Entry Structure
```cpp
struct __align__(16) CacheEntry {
    uint64_t packed_seq;   // Bit-packed sequence (8 bytes)
    float energy;          // Computed LABS energy (4 bytes)
    float padding;         // Alignment (4 bytes)
};  // 16 bytes total, 2048 entries = 32KB
```

### Bit-Packing Scheme
```
N ≤ 32, alphabet ≤ 4:  2 bits/element → uint64_t
N ≤ 25, alphabet ≤ 32: 5 bits/element → uint64_t

Example (N=8, binary):
Sequence: [+1, -1, +1, +1, -1, -1, +1, -1]
Encoded:  [1,  0,  1,  1,  0,  0,  1,  0]
Packed:   0b01001101 = 77
```

### Kernel Launch Configuration
```cpp
num_blocks = 128          // Parallel search regions
threads_per_block = 256   // Threads sharing cache
max_iterations = 1000     // MTS iterations per thread
cache_size = 2048         // Entries per block
```
