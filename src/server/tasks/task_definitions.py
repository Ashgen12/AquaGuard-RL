# src/server/tasks/task_definitions.py
"""
Task configurations for AquaGuard-RL environment.

Defines 5 tasks with different initial conditions, objectives, and difficulty levels.
Each task tests different aspects of the agent's ability to manage the agricultural system.

Tasks:
    1. baseline        — Stable management (EASY)
    2. crisis          — Aquifer crisis recovery (HARD)
    3. policy_shift    — Green Revolution crop transition (MEDIUM)
    4. climate_shock   — Drought year management (VERY HARD)
    5. multi_district  — Cross-district equity coordination (EXPERT)
"""

from __future__ import annotations

from typing import Dict, Any

# ─── Task Configuration Type ──────────────────────────────────────────────────

# Each task is a dict with keys:
#   name: str                   — task identifier
#   description: str            — human-readable description
#   difficulty: str             — EASY/MEDIUM/HARD/VERY_HARD/EXPERT
#   max_steps: int              — maximum seasons per episode
#   zone_a_gw_depth: float      — initial zone A groundwater depth (meters)
#   zone_b_gw_depth: float      — initial zone B groundwater depth (meters)
#   zone_c_gw_depth: float      — initial zone C groundwater depth (meters)
#   initial_allocation: dict    — initial crop allocation fractions
#   farmer_income_ratio: float  — initial income as multiple of poverty line
#   food_security_ratio: float  — initial food security ratio
#   reward_weights: dict        — per-objective reward weights
#   success_criteria: dict      — thresholds for episode success
#   special_conditions: dict    — task-specific simulation modifiers

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {

    # ── Task 1: Baseline ──────────────────────────────────────────────────────

    "baseline": {
        "name": "baseline",
        "description": (
            "Manage a 3-zone agricultural district for 10 seasons (approximately 3.3 years) "
            "without depleting the groundwater aquifer below critical levels. "
            "Starting conditions are typical of a healthy North Indian agricultural district. "
            "The agent must maintain groundwater sustainability, food security, and farmer "
            "welfare simultaneously while improving crop diversity."
        ),
        "difficulty": "EASY",
        "max_steps": 10,

        # Initial conditions — healthy starting state
        "zone_a_gw_depth": 22.0,     # Punjab-type: good aquifer
        "zone_b_gw_depth": 26.0,     # Haryana-type: moderate stress
        "zone_c_gw_depth": 30.0,     # Rajasthan-type: approaching warning
        "initial_allocation": {
            "rice": 0.30, "wheat": 0.30, "millet": 0.15,
            "pulses": 0.15, "oilseeds": 0.07, "vegetables": 0.03,
        },
        "farmer_income_ratio": 1.80,    # 80% above poverty line
        "food_security_ratio": 1.15,    # 15% surplus

        # Reward weights
        "reward_weights": {
            "groundwater": 0.35,
            "food_security": 0.30,
            "farmer_income": 0.25,
            "crop_diversity": 0.10,
        },

        # Success criteria
        "success_criteria": {
            "max_final_gw_depth_m": 38.0,         # All zones ≤ 38m at end
            "min_food_security_rate": 0.80,        # ≥80% of steps meet food target
            "max_poverty_fraction": 0.35,          # Poverty fraction < 35% throughout
            "min_cumulative_reward": 40.0,         # Cumulative reward > 40.0
        },

        # No special conditions for baseline
        "special_conditions": {},
        "food_requirement_multiplier": 1.0,
        "rainfall_shock_factor": 1.0,
    },

    # ── Task 2: Crisis Recovery ───────────────────────────────────────────────

    "crisis": {
        "name": "crisis",
        "description": (
            "The district's aquifer is nearly depleted. Zone C is already at 37m depth "
            "(near the critical 40m threshold). Zone B is at 35m. "
            "Recover groundwater levels while maintaining food production and farmer welfare. "
            "Initial crop pattern is water-intensive (rice 40%, wheat 35%). "
            "The agent must urgently reduce water extraction without causing food crisis "
            "or farmer income collapse."
        ),
        "difficulty": "HARD",
        "max_steps": 12,

        # Initial conditions — crisis state
        "zone_a_gw_depth": 30.0,     # Stressed but manageable
        "zone_b_gw_depth": 35.0,     # Warning zone
        "zone_c_gw_depth": 37.0,     # Near-critical — danger zone
        "initial_allocation": {
            "rice": 0.40, "wheat": 0.35, "millet": 0.10,
            "pulses": 0.08, "oilseeds": 0.05, "vegetables": 0.02,
        },
        "farmer_income_ratio": 1.20,    # Only 20% above poverty line
        "food_security_ratio": 0.95,    # Slight deficit

        # Higher groundwater weight in crisis
        "reward_weights": {
            "groundwater": 0.50,
            "food_security": 0.25,
            "farmer_income": 0.20,
            "crop_diversity": 0.05,
        },

        # Success criteria
        "success_criteria": {
            "zone_c_recovery_m": 33.0,          # Zone C recovers to ≤33m
            "max_any_zone_gw_depth": 50.0,      # No zone collapses
            "min_food_security_all_steps": 0.85, # Allow some reduction
            "min_cumulative_reward": 20.0,
        },

        "special_conditions": {},
        "food_requirement_multiplier": 1.0,
        "rainfall_shock_factor": 1.0,
    },

    # ── Task 3: Policy Shift ──────────────────────────────────────────────────

    "policy_shift": {
        "name": "policy_shift",
        "description": (
            "India's Green Revolution legacy: rice and wheat occupy 70% of arable land, "
            "driven by MSP incentives that make water-intensive monocultures economically rational. "
            "The agent must transition to diversified cropping (Shannon diversity index ≥ 1.2) "
            "over 8 seasons WITHOUT causing a farmer income crisis. "
            "Transition cannot be too fast — farmers can only shift crops gradually (max 8pp per step). "
            "The challenge is making the transition economically viable for farmers while "
            "improving water sustainability."
        ),
        "difficulty": "MEDIUM",
        "max_steps": 8,

        # Initial conditions — Green Revolution lock-in
        "zone_a_gw_depth": 28.0,
        "zone_b_gw_depth": 30.0,
        "zone_c_gw_depth": 32.0,
        "initial_allocation": {
            "rice": 0.40, "wheat": 0.30, "millet": 0.08,
            "pulses": 0.10, "oilseeds": 0.08, "vegetables": 0.04,
        },
        "farmer_income_ratio": 1.50,
        "food_security_ratio": 1.08,

        # Higher diversity and income weights
        "reward_weights": {
            "groundwater": 0.25,
            "food_security": 0.25,
            "farmer_income": 0.30,
            "crop_diversity": 0.20,
        },

        # Transition speed constraint
        "max_rice_allocation_reduction_per_step": 0.08,
        "max_wheat_allocation_reduction_per_step": 0.08,

        # Success criteria
        "success_criteria": {
            "min_final_shannon_diversity": 1.2,  # Must achieve diversity target
            "max_poverty_fraction": 0.20,         # Poverty < 20% throughout
            "min_food_security_ratio": 0.90,      # No food crisis
        },

        "special_conditions": {},
        "food_requirement_multiplier": 1.0,
        "rainfall_shock_factor": 1.0,
    },

    # ── Task 4: Climate Shock ─────────────────────────────────────────────────

    "climate_shock": {
        "name": "climate_shock",
        "description": (
            "A severe El Niño drought year. Kharif rainfall is only 320mm (vs 800mm normal), "
            "and Rabi is also below average. The district starts from healthy conditions "
            "but must manage through 6 seasons of reduced rainfall without triggering "
            "aquifer collapse from panic groundwater extraction or a food security crisis. "
            "This tests adaptive crisis management under external shock."
        ),
        "difficulty": "VERY_HARD",
        "max_steps": 6,

        # Initial conditions — pre-drought healthy state
        "zone_a_gw_depth": 20.0,
        "zone_b_gw_depth": 24.0,
        "zone_c_gw_depth": 28.0,
        "initial_allocation": {
            "rice": 0.30, "wheat": 0.28, "millet": 0.18,
            "pulses": 0.12, "oilseeds": 0.08, "vegetables": 0.04,
        },
        "farmer_income_ratio": 1.60,
        "food_security_ratio": 1.12,

        # Higher groundwater and food weights (drought context)
        "reward_weights": {
            "groundwater": 0.40,
            "food_security": 0.35,
            "farmer_income": 0.20,
            "crop_diversity": 0.05,
        },

        # Success criteria (relaxed for drought conditions)
        "success_criteria": {
            "max_zone_gw_depth": 40.0,          # Prevent panic extraction
            "min_food_security_ratio": 0.75,     # Allow some reduction
            "max_poverty_fraction": 0.50,        # Some farmers will be hurt
            "zero_collapses": True,
        },

        # Climate shock: reduced rainfall
        "special_conditions": {
            "drought_active": True,
        },
        "food_requirement_multiplier": 1.0,
        "rainfall_shock_factor": 0.40,  # 40% of normal rainfall
        "rainfall_shock_by_season": {
            "kharif": 0.40,   # 320mm vs 800mm normal
            "rabi": 0.75,     # 90mm vs 120mm normal
            "zaid": 0.90,     # Near normal
        },
    },

    # ── Task 5: Multi-District Coordination ──────────────────────────────────

    "multi_district": {
        "name": "multi_district",
        "description": (
            "Three economically distinct districts share a single aquifer. "
            "Zone A: productive Northern Plains (rice surplus, high income). "
            "Zone B: Central Plains (wheat export, medium income). "
            "Zone C: Vulnerable Semi-Arid zone (dryland farming, lowest income, highest GW stress). "
            "The agent must balance all three zones with inter-district equity constraints: "
            "no zone's income should fall below 70% of the richest zone's income. "
            "This tests understanding of regional inequality and equitable resource distribution."
        ),
        "difficulty": "EXPERT",
        "max_steps": 15,

        # Initial conditions — distinct zone states
        "zone_a_gw_depth": 18.0,     # Zone A: strong aquifer
        "zone_b_gw_depth": 28.0,     # Zone B: moderate
        "zone_c_gw_depth": 36.0,     # Zone C: stressed
        "initial_allocation": {
            "rice": 0.30, "wheat": 0.25, "millet": 0.18,
            "pulses": 0.12, "oilseeds": 0.10, "vegetables": 0.05,
        },
        "farmer_income_ratio": 1.40,
        "food_security_ratio": 1.05,

        # Equity component added
        "reward_weights": {
            "groundwater": 0.30,
            "food_security": 0.25,
            "farmer_income": 0.25,
            "crop_diversity": 0.10,
            "equity": 0.10,          # Inter-zone income equity
        },

        # Equity constraint
        "inter_zone_income_ratio_min": 0.70,

        # Success criteria
        "success_criteria": {
            "aquifer_stable": True,
            "min_inter_zone_income_ratio": 0.65,
            "no_zone_food_deficit": True,
            "min_cumulative_reward": 60.0,
        },

        "special_conditions": {
            "equity_constraint_active": True,
        },
        "food_requirement_multiplier": 1.0,
        "rainfall_shock_factor": 1.0,
    },
}


def get_task_config(task_name: str) -> Dict[str, Any]:
    """
    Get task configuration by name with fallback to baseline.

    Args:
        task_name: Task identifier.

    Returns:
        Task configuration dictionary.
    """
    if task_name not in TASK_CONFIGS:
        import logging
        logging.getLogger(__name__).warning(
            f"Unknown task '{task_name}', falling back to 'baseline'"
        )
        return TASK_CONFIGS["baseline"]
    return TASK_CONFIGS[task_name]


AVAILABLE_TASKS = list(TASK_CONFIGS.keys())
TASK_DIFFICULTIES = {name: cfg["difficulty"] for name, cfg in TASK_CONFIGS.items()}