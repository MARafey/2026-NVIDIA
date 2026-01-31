"""
Pytest configuration and shared fixtures for LABS-SharedCache test suite.
"""

import pytest
from hypothesis import settings, Verbosity

# -----------------------------------------------------------------------------
# Hypothesis Profiles
# -----------------------------------------------------------------------------

settings.register_profile("fast", max_examples=50)
settings.register_profile("thorough", max_examples=1000)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose)

# Default to fast for regular development
settings.load_profile("fast")


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def small_sequences():
    """Known small sequences for quick testing."""
    return [
        [1, 1, -1],
        [1, -1, 1],
        [-1, 1, 1],
        [1, 1, 1, -1],
        [1, 1, -1, 1],
        [1, -1, 1, 1],
    ]


@pytest.fixture
def known_optima():
    """
    Known optimal LABS solutions.
    Format: {N: (optimal_energy, [one optimal sequence])}
    """
    return {
        3: (1, [1, 1, -1]),
        5: (2, [1, 1, 1, -1, 1]),
        7: (4, [1, 1, 1, -1, -1, 1, -1]),
        11: (12, [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1]),
        13: (18, [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]),
    }


@pytest.fixture
def cache_config():
    """Default cache configuration for tests."""
    return {
        "size": 2048,
        "bits_per_elem": 2,
        "max_n": 32,
    }


# -----------------------------------------------------------------------------
# Markers
# -----------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m not slow')")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU hardware")
    config.addinivalue_line("markers", "integration: marks integration tests")
