# src/aquaguard_env/__init__.py
"""
AquaGuard-RL: India Groundwater & Agricultural Resource Management Environment.

A Mini-RL Environment for the Meta PyTorch OpenEnv Hackathon that simulates
India's groundwater and agricultural resource management crisis.

An LLM agent acts as a District Agricultural Commissioner, making seasonal
policy decisions across 3 interconnected zones to balance:
- Groundwater sustainability (prevent aquifer depletion)
- Food security (meet district food requirements)
- Farmer welfare (keep income above poverty line)
- Crop diversity (encourage water-efficient cropping patterns)

Example usage:
    from aquaguard_env import AquaGuardEnv, AquaGuardAction

    env = AquaGuardEnv("http://localhost:8000")
    obs = env.reset(task="baseline", seed=42)
    action = AquaGuardAction(
        crop_allocation={"rice": 0.20, "wheat": 0.25, "millet": 0.25,
                         "pulses": 0.15, "oilseeds": 0.10, "vegetables": 0.05},
        water_quotas={"zone_a": 750, "zone_b": 700, "zone_c": 600},
        irrigation_methods={"zone_a": "drip", "zone_b": "sprinkler", "zone_c": "drip"},
        extraction_limits={"zone_a": 20.0, "zone_b": 25.0, "zone_c": 15.0},
        subsidy_adjustments={"rice": -0.15, "millet": 0.10, "pulses": 0.10},
        justification="Reducing water-intensive rice from 30% to 20% because..."
    )
    obs = env.step(action)
    print(f"Reward: {obs.reward:.2f}, GW depth: {obs.shared_aquifer_level_m:.1f}m")

For standalone use (no HTTP server):
    from aquaguard_env.server.aquaguard_environment import AquaGuardEnvironment
    env = AquaGuardEnvironment()
    obs = env.reset(task="baseline")
"""

from .models import (
    AquaGuardAction,
    AquaGuardObservation,
    AquaGuardState,
    ZoneObservation,
    CropObservation,
)
from .client import AquaGuardEnv

__all__ = [
    "AquaGuardAction",
    "AquaGuardObservation",
    "AquaGuardState",
    "ZoneObservation",
    "CropObservation",
    "AquaGuardEnv",
]

__version__ = "1.0.0"
__author__ = "ashunaukari01@gmail.com"
__license__ = "MIT"