# src/server/reward.py
"""
Multi-objective reward calculator for AquaGuard-RL.

Reward formula:
    R = w_gw * R_gw + w_food * R_food + w_income * R_income + w_diversity * R_diversity

Each component ∈ [−1, +1] before weighting.
Final reward is scaled to [−10, +10] for strong training signal.

Component weights (default):
    w_gw       = 0.35  (groundwater sustainability — highest: irreversible damage)
    w_food     = 0.30  (food security — national priority)
    w_income   = 0.25  (farmer welfare — social stability)
    w_diversity = 0.10 (crop diversity — resilience + sustainability)

Hard penalties:
    - Aquifer collapse (any zone): −5.0 (catastrophic, irreversible)
    - 3 consecutive food security failures: −3.0 (famine trigger)
"""

from __future__ import annotations

import math
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Maximum Shannon entropy for 6 crops with equal allocation
H_MAX = math.log(6)  # ≈ 1.7918


class RewardCalculator:
    """
    Computes the multi-objective weighted reward signal for AquaGuard-RL.

    The reward is designed to:
    1. Provide dense, informative gradient signal for every action
    2. Penalize irreversible outcomes heavily (aquifer collapse)
    3. Balance competing objectives via configurable weights
    4. Support per-task weight customization (e.g., higher GW weight for crisis task)
    """

    def compute(
        self,
        prev_avg_gw_depth: float,
        new_avg_gw_depth: float,
        zone_states: Dict[str, Dict],
        zone_data: Dict[str, Dict],
        food_security_ratio: float,
        poverty_fraction: float,
        shannon_diversity: float,
        any_zone_collapsed: bool,
        consecutive_food_failures: int,
        reward_weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Compute the scalar reward for one policy step.

        Args:
            prev_avg_gw_depth: Average groundwater depth BEFORE the action.
            new_avg_gw_depth: Average groundwater depth AFTER the action.
            zone_states: Current zone states (dict with gw_depth_m, etc.).
            zone_data: Static zone parameters (critical/collapse thresholds).
            food_security_ratio: Production / requirement ratio.
            poverty_fraction: Fraction of farmers below poverty line [0, 1].
            shannon_diversity: Shannon entropy of crop allocation.
            any_zone_collapsed: True if any zone exceeded collapse threshold.
            consecutive_food_failures: Number of consecutive steps with food ratio < 1.0.
            reward_weights: Per-objective weights (default: 0.35/0.30/0.25/0.10).

        Returns:
            Scalar reward in [−10, +10].
        """
        if reward_weights is None:
            reward_weights = {
                "groundwater": 0.35,
                "food_security": 0.30,
                "farmer_income": 0.25,
                "crop_diversity": 0.10,
            }

        w_gw = reward_weights.get("groundwater", 0.35)
        w_food = reward_weights.get("food_security", 0.30)
        w_income = reward_weights.get("farmer_income", 0.25)
        w_div = reward_weights.get("crop_diversity", 0.10)

        # Compute individual components ∈ [−1, +1]
        r_gw = self._groundwater_reward(prev_avg_gw_depth, new_avg_gw_depth, zone_states, zone_data)
        r_food = self._food_security_reward(food_security_ratio)
        r_income = self._farmer_income_reward(poverty_fraction)
        r_diversity = self._crop_diversity_reward(shannon_diversity)

        # Weighted base reward ∈ [−1, +1]
        base = w_gw * r_gw + w_food * r_food + w_income * r_income + w_div * r_diversity

        # Apply crisis multiplier (reduces reward signal during ongoing crises)
        multiplier = self._crisis_multiplier(zone_states, zone_data, food_security_ratio, poverty_fraction)
        base *= multiplier

        # Hard penalties (deducted after scaling to maintain [−10, +10] range)
        penalties = 0.0
        if any_zone_collapsed:
            penalties += 5.0   # Catastrophic irreversible damage
        if consecutive_food_failures >= 3:
            penalties += 3.0   # Famine threshold

        # Scale to [−10, +10] for training signal strength
        final = base * 10.0 - penalties
        final = max(-10.0, min(10.0, final))

        logger.debug(
            f"Reward: gw={r_gw:.3f} food={r_food:.3f} income={r_income:.3f} "
            f"div={r_diversity:.3f} | base={base:.3f} mult={multiplier:.3f} "
            f"penalties={penalties:.1f} → final={final:.3f}"
        )
        return final

    def _groundwater_reward(
        self,
        prev_avg_depth: float,
        new_avg_depth: float,
        zone_states: Dict[str, Dict],
        zone_data: Dict[str, Dict],
    ) -> float:
        """
        Reward component for groundwater conservation.

        Combines:
        1. Delta improvement: did the average depth improve this step?
        2. Zone health scores: are zones in safe ranges?

        Returns value in [−1, +1].
        """
        # Component 1: Improvement delta (did depth decrease = aquifer recovering?)
        delta = new_avg_depth - prev_avg_depth
        if delta < 0:
            # Recovering: depth decreased (positive, capped at 1.0 for 2m improvement)
            f_improvement = min(1.0, -delta / 2.0)
        elif delta > 0:
            # Depleting: depth increased (negative, capped at -1.0 for 5m depletion)
            f_improvement = max(-1.0, -delta / 5.0)
        else:
            f_improvement = 0.0

        # Component 2: Zone health scores (where are zones relative to critical threshold?)
        zone_scores = []
        for zone_id, zone_state in zone_states.items():
            depth = zone_state["gw_depth_m"]
            critical = zone_data[zone_id]["critical_threshold_m"]
            collapse = zone_data[zone_id]["collapse_threshold_m"]

            # Piecewise linear score based on depth bands
            if zone_state.get("is_collapsed", False):
                score = -1.0
            elif depth < 20:
                score = 1.0
            elif depth < 30:
                # Good → moderate: linear from 1.0 to 0.5
                score = 1.0 - (depth - 20) / 10.0 * 0.5
            elif depth < critical - 5:
                # Moderate → warning: linear from 0.5 to 0.0
                band = critical - 5 - 30
                if band > 0:
                    score = max(0.0, 0.5 - (depth - 30) / band * 0.5)
                else:
                    score = 0.0
            elif depth < critical:
                # Warning → danger: linear from 0.0 to -0.5
                score = -0.5 * (depth - (critical - 5)) / 5.0
            elif depth < collapse:
                score = -1.0
            else:
                score = -1.0

            zone_scores.append(score)

        f_zone = sum(zone_scores) / max(len(zone_scores), 1)

        # Combined: equal weight to improvement and zone health
        return 0.5 * f_improvement + 0.5 * f_zone

    def _food_security_reward(self, food_security_ratio: float) -> float:
        """
        Sigmoid reward for food security.

        Thresholds:
            ≥ 1.2: Excellent surplus → +1.0
            1.0–1.2: Meeting target → +0.5 to +1.0
            0.8–1.0: Mild shortfall → -0.5 to +0.5
            0.5–0.8: Serious shortfall → -0.75 to -0.5
            < 0.5: Famine threshold → -1.0

        Returns value in [−1, +1].
        """
        r = food_security_ratio
        if r >= 1.2:
            return 1.0
        elif r >= 1.0:
            return 0.5 + (r - 1.0) / 0.2 * 0.5
        elif r >= 0.8:
            # Linear interpolation: -0.5 at 0.8, +0.5 at 1.0
            return -0.5 + (r - 0.8) / 0.2 * 1.0
        elif r >= 0.5:
            return -0.75 + (r - 0.5) / 0.3 * 0.25
        else:
            return -1.0

    def _farmer_income_reward(self, poverty_fraction: float) -> float:
        """
        Linear reward based on poverty fraction.

        Formula: 1 - 2 × poverty_fraction

        Interpretation:
            poverty_fraction = 0.0 → +1.0 (no poverty)
            poverty_fraction = 0.5 → 0.0 (half in poverty)
            poverty_fraction = 1.0 → -1.0 (all in poverty)

        Extra penalty for widespread distress (> 70% poverty → social instability).

        Returns value in [−1, +1].
        """
        base = 1.0 - 2.0 * poverty_fraction

        # Extra penalty: crisis multiplier applied when widespread farmer distress
        if poverty_fraction > 0.70:
            base *= 1.5  # Extra 50% penalty for social instability risk

        return max(-1.0, min(1.0, base))

    def _crop_diversity_reward(self, shannon_index: float) -> float:
        """
        Reward for crop diversity based on Shannon entropy.

        Formula: 2 × (H / H_max) − 1

        Interpretation:
            H = 0 (monoculture) → -1.0
            H = log(6) (equal 6-crop distribution) → +1.0

        Returns value in [−1, +1].
        """
        normalized = min(1.0, shannon_index / H_MAX)
        return 2.0 * normalized - 1.0

    def _crisis_multiplier(
        self,
        zone_states: Dict[str, Dict],
        zone_data: Dict[str, Dict],
        food_ratio: float,
        poverty_frac: float,
    ) -> float:
        """
        Compute reward multiplier applied during ongoing crises.

        Reduces reward when crises are already underway (not the first time penalty
        is applied, but ongoing penalty for sustained bad states).

        Returns multiplier in [0.28, 1.0].
        """
        multiplier = 1.0

        # Danger zones: aquifer exceeding critical threshold
        danger_zones = sum(
            1 for zid, zs in zone_states.items()
            if zs["gw_depth_m"] > zone_data[zid]["critical_threshold_m"]
        )
        if danger_zones > 0:
            # Reduce by 20% per danger zone, min 40% of base
            multiplier *= max(0.4, 1.0 - 0.2 * danger_zones)

        # Serious food crisis
        if food_ratio < 0.7:
            multiplier *= 0.7

        # Widespread farmer poverty
        if poverty_frac > 0.8:
            multiplier *= 0.8

        return max(0.28, multiplier)

    def decompose(
        self,
        prev_avg_gw_depth: float,
        new_avg_gw_depth: float,
        zone_states: Dict[str, Dict],
        zone_data: Dict[str, Dict],
        food_security_ratio: float,
        poverty_fraction: float,
        shannon_diversity: float,
        reward_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Return decomposed reward components for analysis and debugging.

        Returns:
            Dictionary with keys: gw, food, income, diversity, composite, multiplier.
        """
        if reward_weights is None:
            reward_weights = {
                "groundwater": 0.35,
                "food_security": 0.30,
                "farmer_income": 0.25,
                "crop_diversity": 0.10,
            }

        r_gw = self._groundwater_reward(prev_avg_gw_depth, new_avg_gw_depth, zone_states, zone_data)
        r_food = self._food_security_reward(food_security_ratio)
        r_income = self._farmer_income_reward(poverty_fraction)
        r_div = self._crop_diversity_reward(shannon_diversity)
        mult = self._crisis_multiplier(zone_states, zone_data, food_security_ratio, poverty_fraction)

        return {
            "groundwater": r_gw,
            "food_security": r_food,
            "farmer_income": r_income,
            "crop_diversity": r_div,
            "crisis_multiplier": mult,
            "weighted_sum": (
                reward_weights.get("groundwater", 0.35) * r_gw
                + reward_weights.get("food_security", 0.30) * r_food
                + reward_weights.get("farmer_income", 0.25) * r_income
                + reward_weights.get("crop_diversity", 0.10) * r_div
            ),
        }