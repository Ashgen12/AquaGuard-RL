#!/usr/bin/env python3
"""
AquaGuard-RL Demo Heuristic Agent
==================================
Runs a complete baseline episode using a rule-based heuristic policy.
Demonstrates how to interact with the environment via HTTP client.

Usage:
    # Start server first:
    cd /path/to/AquaGuard-RL
    uvicorn server.app:app --host 0.0.0.0 --port 8000

    # Then run this script:
    python scripts/run_sample_agent.py
    python scripts/run_sample_agent.py --task crisis
    python scripts/run_sample_agent.py --task policy_shift --server http://localhost:8000
"""

import sys
import os
import argparse

from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from client import AquaGuardEnv
from models import AquaGuardAction


def heuristic_policy(obs) -> AquaGuardAction:
    """
    Simple heuristic policy for demonstration purposes.

    Decision logic:
    - If aquifer is critically stressed (> 85% of critical threshold):
        → Aggressively reduce rice/wheat, deploy drip, cut extraction
    - If aquifer is moderately stressed (> 65%):
        → Moderate reduction, sprinkler irrigation
    - If healthy:
        → Maintain with gradual diversification improvements
    """
    gw = obs.shared_aquifer_level_m
    critical = obs.critical_aquifer_threshold_m
    stress = gw / critical if critical > 0 else 0

    food_ok = obs.food_security_ratio >= 1.0

    if stress > 0.85:  # Critical: aggressive conservation
        rice_alloc = 0.15 if food_ok else 0.22
        return AquaGuardAction(
            crop_allocation={
                "rice": rice_alloc, "wheat": 0.18, "millet": 0.30,
                "pulses": 0.20, "oilseeds": 0.10, "vegetables": 0.05,
            },
            water_quotas={"zone_a": 580, "zone_b": 530, "zone_c": 480},
            irrigation_methods={"zone_a": "drip", "zone_b": "drip", "zone_c": "drip"},
            extraction_limits={"zone_a": 14.0, "zone_b": 11.0, "zone_c": 8.0},
            subsidy_adjustments={
                "rice": -0.20, "wheat": -0.10, "millet": 0.18,
                "pulses": 0.18, "oilseeds": 0.12, "vegetables": 0.0,
            },
            justification=(
                f"CRITICAL: Aquifer at {gw:.1f}m ({stress:.0%} of {critical:.0f}m critical). "
                f"Emergency conservation: rice cut to {rice_alloc:.0%} "
                f"(saves {(0.3 - rice_alloc) * 1200:.0f}mm water/season). "
                f"Drip irrigation deployed across all zones (45% water saving). "
                f"Extraction limits reduced to 14/11/8 m/season. "
                f"MSP subsidies shifted: -20% rice, +18% millet to incentivize farmers."
            ),
        )

    elif stress > 0.65:  # Warning: moderate measures
        return AquaGuardAction(
            crop_allocation={
                "rice": 0.23, "wheat": 0.22, "millet": 0.23,
                "pulses": 0.17, "oilseeds": 0.10, "vegetables": 0.05,
            },
            water_quotas={"zone_a": 720, "zone_b": 680, "zone_c": 620},
            irrigation_methods={
                "zone_a": "sprinkler", "zone_b": "sprinkler", "zone_c": "drip"
            },
            extraction_limits={"zone_a": 20.0, "zone_b": 17.0, "zone_c": 13.0},
            subsidy_adjustments={
                "rice": -0.10, "wheat": -0.05, "millet": 0.10,
                "pulses": 0.10, "oilseeds": 0.05, "vegetables": 0.0,
            },
            justification=(
                f"Warning: Aquifer at {gw:.1f}m ({stress:.0%} of critical). "
                f"Moderate action: 7pp shift from rice/wheat to millet/pulses. "
                f"Sprinkler irrigation in Zones A/B (30% water saving). "
                f"Food security {obs.food_security_ratio:.2f} maintained above threshold."
            ),
        )

    else:  # Healthy: gradual optimization
        return AquaGuardAction(
            crop_allocation={
                "rice": 0.27, "wheat": 0.26, "millet": 0.20,
                "pulses": 0.14, "oilseeds": 0.09, "vegetables": 0.04,
            },
            water_quotas={"zone_a": 850, "zone_b": 800, "zone_c": 740},
            irrigation_methods={
                "zone_a": "flood", "zone_b": "sprinkler", "zone_c": "sprinkler"
            },
            extraction_limits={"zone_a": 26.0, "zone_b": 23.0, "zone_c": 19.0},
            subsidy_adjustments={
                "millet": 0.06, "pulses": 0.06, "rice": -0.03,
            },
            justification=(
                f"Healthy aquifer at {gw:.1f}m. Proactive diversification: "
                f"millet/pulses nudged up 4-5pp. Zone B/C using sprinkler. "
                f"Shannon diversity {obs.shannon_diversity_index:.3f} — targeting improvement. "
                f"Food security {obs.food_security_ratio:.2f} safely above 1.0."
            ),
        )


def main():
    parser = argparse.ArgumentParser(description="AquaGuard-RL Demo Heuristic Agent")
    parser.add_argument("--task", default="baseline",
                        choices=["baseline", "crisis", "policy_shift", "climate_shock", "multi_district"],
                        help="Task to run (default: baseline)")
    parser.add_argument("--server", default="http://localhost:8000",
                        help="Environment server URL (default: http://localhost:8000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    print(f"\nAquaGuard-RL Demo — Task: {args.task} (seed={args.seed})")
    print("=" * 60)
    print(f"Connecting to: {args.server}")
    print()

    env = AquaGuardEnv(args.server)

    if not env.health_check():
        print("ERROR: Cannot connect to environment server.")
        print("Start it with:")
        print("  cd AquaGuard-RL")
        print("  uvicorn server.app:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    obs = env.reset(task=args.task, seed=args.seed)

    print(f"Task: {obs.task_name} | Season: {obs.season} | Year {obs.year}")
    print(f"Initial GW: {obs.shared_aquifer_level_m:.1f}m | "
          f"Food: {obs.food_security_ratio:.2f} | "
          f"Poverty: {obs.percent_farmers_below_poverty:.1f}%")
    print(f"Shannon diversity: {obs.shannon_diversity_index:.3f}")
    print()

    total_reward = 0.0
    step = 0

    while not obs.done:
        step += 1
        action = heuristic_policy(obs)
        obs = env.step(action)
        total_reward += obs.reward or 0.0

        print(f"Step {step:2d} [{obs.season:6s} Y{obs.year}]: "
              f"reward={obs.reward:+.2f} | "
              f"GW={obs.shared_aquifer_level_m:.1f}m | "
              f"food={obs.food_security_ratio:.2f} | "
              f"poverty={obs.percent_farmers_below_poverty:.1f}% | "
              f"H={obs.shannon_diversity_index:.3f}")

        if obs.grader_feedback:
            print(f"          → {obs.grader_feedback[:100]}")

    state = env.state
    print()
    print("=" * 60)
    print(f"Episode complete: {step} seasons")
    print(f"Total reward:          {total_reward:>10.2f}")
    print(f"Final GW depth (avg):  {obs.shared_aquifer_level_m:>10.1f} m")
    print(f"Final food security:   {obs.food_security_ratio:>10.2f}")
    print(f"Final poverty:         {obs.percent_farmers_below_poverty:>10.1f} %")
    print(f"Best Shannon diversity:{state.best_shannon_diversity:>10.3f}")
    print(f"Food security failures:{state.food_security_failures:>10d}")
    print(f"GW crisis triggered:   {str(state.groundwater_crisis_triggered):>10}")

    env.close()


if __name__ == "__main__":
    main()