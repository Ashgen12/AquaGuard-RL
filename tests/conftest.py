# tests/conftest.py
"""pytest fixtures for AquaGuard-RL tests."""

import sys
import os
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def env():
    """Create and return a fresh AquaGuardEnvironment instance."""
    from server.aquaguard_environment import AquaGuardEnvironment
    return AquaGuardEnvironment()


@pytest.fixture
def env_reset(env):
    """Return environment after reset with baseline task."""
    obs = env.reset(task="baseline", seed=42)
    return env, obs


@pytest.fixture
def default_action():
    """Return a default valid AquaGuardAction."""
    from models import AquaGuardAction
    return AquaGuardAction()


@pytest.fixture
def conservative_action():
    """Return a water-conservative action."""
    from models import AquaGuardAction
    return AquaGuardAction(
        crop_allocation={
            "rice": 0.15, "wheat": 0.20, "millet": 0.30,
            "pulses": 0.20, "oilseeds": 0.10, "vegetables": 0.05,
        },
        water_quotas={"zone_a": 600, "zone_b": 550, "zone_c": 500},
        irrigation_methods={"zone_a": "drip", "zone_b": "drip", "zone_c": "drip"},
        extraction_limits={"zone_a": 15.0, "zone_b": 12.0, "zone_c": 10.0},
        subsidy_adjustments={"rice": -0.15, "wheat": -0.05, "millet": 0.15, "pulses": 0.10},
        justification="Reducing water-intensive crops, deploying drip irrigation.",
    )