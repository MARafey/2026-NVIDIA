# Test Suite Documentation

**Project Name:** LABS-SharedCache
**Team Name:** [Your Team Name]

---

## 1. Verification Strategy Overview

Our verification strategy follows a **layered testing pyramid**:

```
                    ┌─────────────┐
                    │ Integration │  ← Full pipeline tests
                    ├─────────────┤
                 ┌──┴─────────────┴──┐
                 │  Property-Based   │  ← Physics invariants
                 ├───────────────────┤
              ┌──┴───────────────────┴──┐
              │      Unit Tests         │  ← Individual functions
              └─────────────────────────┘
```

### Testing Principles

1. **Physics First:** Verify physical invariants (symmetries, bounds) before implementation details
2. **Ground Truth Anchoring:** Validate against known optimal solutions for small N
3. **Property-Based Testing:** Use `hypothesis` to generate edge cases automatically
4. **AI Guardrails:** Every AI-generated function requires a test before integration

---

## 2. Test Categories & Coverage

### Coverage Matrix

| Component | Unit Tests | Property Tests | Integration | Ground Truth |
|-----------|:----------:|:--------------:|:-----------:|:------------:|
| Bit-Packing | 5 | 2 | - | - |
| Energy Calculation | 3 | 3 | - | 5 |
| Cache Operations | 6 | 2 | 1 | - |
| MTS Algorithm | 2 | 1 | 2 | 3 |
| Quantum Seeds | 2 | 1 | 1 | - |
| Full Pipeline | - | - | 2 | 1 |

### Test Selection Rationale

| Test Category | Why These Tests? |
|---------------|------------------|
| **Bit-packing round-trip** | Core data structure - corruption here breaks everything |
| **Energy symmetries** | Physics constraints - if violated, algorithm is fundamentally wrong |
| **Known optima** | Ground truth anchors - confirms we're solving the right problem |
| **Cache collision** | Critical for correctness - wrong policy = incorrect results |
| **Quantum sample validity** | Integration point - bad seeds propagate errors downstream |

---

## 3. Test Files Structure

```
team-submissions/
├── TEST_SUITE.md           # This documentation
└── tests/
    ├── __init__.py
    ├── conftest.py         # Pytest fixtures and configuration
    ├── test_bit_packing.py # Sequence encoding tests
    ├── test_energy.py      # LABS energy calculation tests
    ├── test_cache.py       # Shared cache operation tests
    ├── test_mts.py         # Memetic Tabu Search tests
    ├── test_quantum.py     # Quantum seed generation tests
    └── test_integration.py # Full pipeline tests
```

---

## 4. Running the Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run only fast tests (skip slow integration)
pytest tests/ -v -m "not slow"

# Run property-based tests with more examples
pytest tests/ -v --hypothesis-seed=42 --hypothesis-profile=thorough
```

---

## 5. Continuous Integration Hooks

### Pre-commit Checks
```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: pytest-quick
      name: Quick Tests
      entry: pytest tests/ -v -m "not slow" --tb=short
      language: system
      pass_filenames: false
```

### Before GPU Migration Checklist
- [ ] All unit tests pass
- [ ] All property tests pass with 1000+ examples
- [ ] Ground truth tests pass for N=3,5,7,11,13
- [ ] Cache tests pass with simulated concurrency

---

## 6. Known Limitations & Future Tests

### Current Limitations
- GPU kernel tests require CUDA hardware (mocked on CPU)
- Async cache tests use simulated concurrency, not true GPU threads
- Large N tests (N>20) are slow, marked as `@pytest.mark.slow`

### Planned Additional Tests
- [ ] Multi-block communication tests
- [ ] Memory leak detection for long runs
- [ ] Performance regression tests
- [ ] Numerical stability tests for energy computation
