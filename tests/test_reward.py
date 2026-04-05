# tests/test_reward.py
"""Tests for the RewardCalculator multi-objective reward function."""

import sys
import os
import math
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from server.reward import RewardCalculator, H_MAX
from constants import ZONE_DATA


def make_zone_states(depths=None):
    """Helper to create zone_states dict."""
    if depths is None:
        depths = {"zone_a": 25.0, "zone_b": 28.0, "zone_c": 32.0}
    return {
        zid: {
            "gw_depth_m": depths.get(zid, 25.0),
            "is_collapsed": False,
        }
        for zid in ["zone_a", "zone_b", "zone_c"]
    }


class TestRewardCalculator:
    """Tests for RewardCalculator."""

    def setup_method(self):
        self.calc = RewardCalculator()

    def test_reward_in_bounds(self):
        """Reward must always be in [-10, +10]."""
        zone_states = make_zone_states()
        for food_ratio in [0.5, 1.0, 1.2]:
            for poverty in [0.1, 0.5, 0.9]:
                for shannon in [0.0, 1.0, 1.79]:
                    reward = self.calc.compute(
                        prev_avg_gw_depth=25.0,
                        new_avg_gw_depth=25.0,
                        zone_states=zone_states,
                        zone_data=ZONE_DATA,
                        food_security_ratio=food_ratio,
                        poverty_fraction=poverty,
                        shannon_diversity=shannon,
                        any_zone_collapsed=False,
                        consecutive_food_failures=0,
                    )
                    assert -10.0 <= reward <= 10.0, (
                        f"Reward {reward} out of bounds for food={food_ratio}, "
                        f"poverty={poverty}, shannon={shannon}"
                    )

    def test_reward_not_nan_or_inf(self):
        """Reward must never be NaN or infinite."""
        zone_states = make_zone_states()
        reward = self.calc.compute(
            prev_avg_gw_depth=25.0,
            new_avg_gw_depth=25.0,
            zone_states=zone_states,
            zone_data=ZONE_DATA,
            food_security_ratio=1.0,
            poverty_fraction=0.3,
            shannon_diversity=1.0,
            any_zone_collapsed=False,
            consecutive_food_failures=0,
        )
        assert not math.isnan(reward)
        assert not math.isinf(reward)

    def test_aquifer_collapse_penalty(self):
        """Aquifer collapse should reduce reward by 5 points."""
        zone_states = make_zone_states()
        reward_no_collapse = self.calc.compute(
            prev_avg_gw_depth=25.0,
            new_avg_gw_depth=25.0,
            zone_states=zone_states,
            zone_data=ZONE_DATA,
            food_security_ratio=1.2,
            poverty_fraction=0.1,
            shannon_diversity=1.5,
            any_zone_collapsed=False,
            consecutive_food_failures=0,
        )
        reward_with_collapse = self.calc.compute(
            prev_avg_gw_depth=25.0,
            new_avg_gw_depth=25.0,
            zone_states=zone_states,
            zone_data=ZONE_DATA,
            food_security_ratio=1.2,
            poverty_fraction=0.1,
            shannon_diversity=1.5,
            any_zone_collapsed=True,  # Collapse penalty
            consecutive_food_failures=0,
        )
        assert reward_with_collapse < reward_no_collapse

    def test_famine_penalty(self):
        """3 consecutive food failures should reduce reward."""
        zone_states = make_zone_states()
        reward_no_famine = self.calc.compute(
            prev_avg_gw_depth=25.0,
            new_avg_gw_depth=25.0,
            zone_states=zone_states,
            zone_data=ZONE_DATA,
            food_security_ratio=1.0,
            poverty_fraction=0.3,
            shannon_diversity=1.0,
            any_zone_collapsed=False,
            consecutive_food_failures=0,
        )
        reward_with_famine = self.calc.compute(
            prev_avg_gw_depth=25.0,
            new_avg_gw_depth=25.0,
            zone_states=zone_states,
            zone_data=ZONE_DATA,
            food_security_ratio=1.0,
            poverty_fraction=0.3,
            shannon_diversity=1.0,
            any_zone_collapsed=False,
            consecutive_food_failures=3,  # Famine
        )
        assert reward_with_famine < reward_no_famine

    def test_food_security_reward_monotone(self):
        """Higher food security ratio should give higher food reward."""
        ratios = [0.4, 0.7, 0.9, 1.0, 1.1, 1.2, 1.5]
        rewards = [self.calc._food_security_reward(r) for r in ratios]
        for i in range(1, len(rewards)):
            assert rewards[i] >= rewards[i-1], f"Not monotone at ratio={ratios[i]}"

    def test_food_reward_in_bounds(self):
        """Food security reward must be in [-1, +1]."""
        for ratio in [0.0, 0.5, 0.8, 1.0, 1.2, 2.0]:
            r = self.calc._food_security_reward(ratio)
            assert -1.0 <= r <= 1.0

    def test_income_reward_no_poverty_is_positive(self):
        """Zero poverty should give positive income reward."""
        r = self.calc._farmer_income_reward(0.0)
        assert r > 0.0

    def test_income_reward_all_poverty_is_negative(self):
        """100% poverty should give negative income reward."""
        r = self.calc._farmer_income_reward(1.0)
        assert r < 0.0

    def test_diversity_reward_monoculture_is_negative(self):
        """Zero Shannon diversity (monoculture) should give -1.0."""
        r = self.calc._crop_diversity_reward(0.0)
        assert r == -1.0

    def test_diversity_reward_max_diversity_is_positive(self):
        """Maximum Shannon diversity should give +1.0."""
        r = self.calc._crop_diversity_reward(H_MAX)
        assert abs(r - 1.0) < 0.01

    def test_aquifer_recovering_gives_positive_gw_reward(self):
        """Improving (shallowing) aquifer depth should give positive GW improvement reward."""
        zone_states = make_zone_states({"zone_a": 24.0, "zone_b": 26.0, "zone_c": 28.0})
        r = self.calc._groundwater_reward(
            prev_avg_depth=30.0,  # Was deeper
            new_avg_depth=26.0,   # Now shallower (recovering)
            zone_states=zone_states,
            zone_data=ZONE_DATA,
        )
        assert r > 0.0

    def test_reward_better_with_good_policy(self):
        """Good policy (low poverty, high food, good GW) should yield higher reward."""
        zone_states_good = make_zone_states({"zone_a": 20.0, "zone_b": 22.0, "zone_c": 24.0})
        reward_good = self.calc.compute(
            prev_avg_gw_depth=23.0,
            new_avg_gw_depth=22.0,   # Improving
            zone_states=zone_states_good,
            zone_data=ZONE_DATA,
            food_security_ratio=1.2,
            poverty_fraction=0.15,
            shannon_diversity=1.5,
            any_zone_collapsed=False,
            consecutive_food_failures=0,
        )

        zone_states_bad = make_zone_states({"zone_a": 38.0, "zone_b": 39.0, "zone_c": 39.5})
        reward_bad = self.calc.compute(
            prev_avg_gw_depth=35.0,
            new_avg_gw_depth=39.0,   # Worsening
            zone_states=zone_states_bad,
            zone_data=ZONE_DATA,
            food_security_ratio=0.7,
            poverty_fraction=0.65,
            shannon_diversity=0.2,
            any_zone_collapsed=False,
            consecutive_food_failures=2,
        )

        assert reward_good > reward_bad