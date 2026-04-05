# tests/test_models.py
"""Tests for AquaGuard-RL data models (Action, Observation, State)."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import AquaGuardAction, AquaGuardState, ZoneObservation, CropObservation


class TestAquaGuardAction:
    """Tests for AquaGuardAction validation."""

    def test_default_action_is_valid(self):
        """Default action should be valid."""
        action = AquaGuardAction()
        assert action is not None
        total = sum(action.crop_allocation.values())
        assert total <= 1.001

    def test_allocation_sum_valid(self):
        """Valid allocation summing to exactly 1.0."""
        action = AquaGuardAction(
            crop_allocation={
                "rice": 0.20, "wheat": 0.20, "millet": 0.20,
                "pulses": 0.15, "oilseeds": 0.15, "vegetables": 0.10,
            }
        )
        assert abs(sum(action.crop_allocation.values()) - 1.0) < 0.01

    def test_allocation_sum_too_high_raises(self):
        """Allocation sum > 1.0 should raise ValueError."""
        with pytest.raises(Exception):  # pydantic ValidationError
            AquaGuardAction(
                crop_allocation={
                    "rice": 0.40, "wheat": 0.40, "millet": 0.30,
                    "pulses": 0.10, "oilseeds": 0.05, "vegetables": 0.03,
                }
            )

    def test_negative_allocation_raises(self):
        """Negative allocation should raise ValueError."""
        with pytest.raises(Exception):
            AquaGuardAction(
                crop_allocation={"rice": -0.1, "wheat": 0.5}
            )

    def test_invalid_crop_raises(self):
        """Unknown crop in allocation should raise ValueError."""
        with pytest.raises(Exception):
            AquaGuardAction(crop_allocation={"corn": 0.5, "rice": 0.3})

    def test_water_quota_bounds(self):
        """Water quota must be in [0, 2000]."""
        # Valid
        action = AquaGuardAction(water_quotas={"zone_a": 0, "zone_b": 2000, "zone_c": 500})
        assert action.water_quotas["zone_b"] == 2000

        # Invalid
        with pytest.raises(Exception):
            AquaGuardAction(water_quotas={"zone_a": 2001})

    def test_invalid_irrigation_method_raises(self):
        """Invalid irrigation method should raise ValueError."""
        with pytest.raises(Exception):
            AquaGuardAction(irrigation_methods={"zone_a": "canal"})

    def test_valid_irrigation_methods(self):
        """All valid irrigation methods should be accepted."""
        for method in ["flood", "sprinkler", "drip"]:
            action = AquaGuardAction(
                irrigation_methods={"zone_a": method, "zone_b": method, "zone_c": method}
            )
            assert action.irrigation_methods["zone_a"] == method

    def test_extraction_limit_bounds(self):
        """Extraction limit must be in [0, 60]."""
        with pytest.raises(Exception):
            AquaGuardAction(extraction_limits={"zone_a": 61.0})

    def test_subsidy_adjustment_bounds(self):
        """Subsidy adjustment must be in [-1, 1]."""
        with pytest.raises(Exception):
            AquaGuardAction(subsidy_adjustments={"rice": 1.5})
        with pytest.raises(Exception):
            AquaGuardAction(subsidy_adjustments={"rice": -1.5})

    def test_justification_max_length(self):
        """Justification cannot exceed 2000 characters."""
        # Valid: empty
        action = AquaGuardAction(justification="")
        assert action.justification == ""

        # Valid: long
        action = AquaGuardAction(justification="x" * 2000)
        assert len(action.justification) == 2000

    def test_partial_allocation_valid(self):
        """Allocation summing to less than 1.0 (fallow land) is valid."""
        action = AquaGuardAction(
            crop_allocation={"rice": 0.30, "wheat": 0.20}
        )
        assert sum(action.crop_allocation.values()) < 1.0


class TestAquaGuardState:
    """Tests for AquaGuardState model."""

    def test_default_state_valid(self):
        """Default state should be valid."""
        state = AquaGuardState()
        assert state.task_name == "baseline"
        assert state.max_steps == 10
        assert state.cumulative_reward == 0.0

    def test_state_with_episode_id(self):
        """State with explicit episode_id."""
        state = AquaGuardState(episode_id="test-123", step_count=5)
        assert state.episode_id == "test-123"
        assert state.step_count == 5


class TestZoneObservation:
    """Tests for ZoneObservation model."""

    def test_valid_zone_obs(self):
        """Valid zone observation should be created."""
        zone = ZoneObservation(
            zone_id="zone_a",
            groundwater_depth_m=25.0,
            groundwater_recharge_rate_mm_yr=180.0,
            soil_fertility=0.75,
            soil_salinity=0.12,
            arable_land_ha=80000.0,
            active_irrigation_method="flood",
            water_used_mm=900.0,
            is_in_danger_zone=False,
            is_collapsed=False,
        )
        assert zone.zone_id == "zone_a"
        assert zone.groundwater_depth_m == 25.0

    def test_fertility_bounds(self):
        """Soil fertility must be in [0, 1]."""
        with pytest.raises(Exception):
            ZoneObservation(
                zone_id="zone_a",
                groundwater_depth_m=25.0,
                groundwater_recharge_rate_mm_yr=180.0,
                soil_fertility=1.5,  # Invalid
                soil_salinity=0.12,
                arable_land_ha=80000.0,
                active_irrigation_method="flood",
                water_used_mm=900.0,
                is_in_danger_zone=False,
            )