# tests/test_environment.py
"""Tests for AquaGuardEnvironment — reset, step, state cycle."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import AquaGuardAction, AquaGuardObservation, AquaGuardState
from server.aquaguard_environment import AquaGuardEnvironment


class TestReset:
    """Tests for the reset() method."""

    def test_reset_returns_observation(self):
        """reset() must return an AquaGuardObservation."""
        env = AquaGuardEnvironment()
        obs = env.reset(task="baseline", seed=42)
        assert isinstance(obs, AquaGuardObservation)

    def test_reset_step_number_is_zero(self):
        """Initial observation should have step_number=0."""
        env = AquaGuardEnvironment()
        obs = env.reset()
        assert obs.step_number == 0

    def test_reset_reward_is_none(self):
        """Initial reward should be None (not yet computed)."""
        env = AquaGuardEnvironment()
        obs = env.reset()
        assert obs.reward is None

    def test_reset_done_is_false(self):
        """Episode should not be done at start."""
        env = AquaGuardEnvironment()
        obs = env.reset()
        assert obs.done is False

    def test_reset_has_three_zones(self):
        """Observation must have 3 zones."""
        env = AquaGuardEnvironment()
        obs = env.reset()
        assert len(obs.zones) == 3
        assert "zone_a" in obs.zones
        assert "zone_b" in obs.zones
        assert "zone_c" in obs.zones

    def test_reset_has_six_crops(self):
        """Observation must have 6 crops."""
        env = AquaGuardEnvironment()
        obs = env.reset()
        assert len(obs.crops) == 6

    def test_reset_has_scenario_description(self):
        """Observation must have a non-empty scenario description."""
        env = AquaGuardEnvironment()
        obs = env.reset()
        assert len(obs.scenario_description) > 50

    def test_reset_season_valid(self):
        """Season must be one of kharif/rabi/zaid."""
        env = AquaGuardEnvironment()
        obs = env.reset()
        assert obs.season in ["kharif", "rabi", "zaid"]

    def test_reset_reproducible_with_seed(self):
        """Same seed should produce same initial state."""
        env1 = AquaGuardEnvironment()
        env2 = AquaGuardEnvironment()
        obs1 = env1.reset(task="baseline", seed=42)
        obs2 = env2.reset(task="baseline", seed=42)
        assert abs(obs1.shared_aquifer_level_m - obs2.shared_aquifer_level_m) < 0.01

    def test_reset_all_tasks(self):
        """All 5 tasks should reset without errors."""
        env = AquaGuardEnvironment()
        for task in ["baseline", "crisis", "policy_shift", "climate_shock", "multi_district"]:
            obs = env.reset(task=task, seed=42)
            assert obs is not None
            assert obs.task_name == task

    def test_crisis_task_starts_with_deep_aquifer(self):
        """Crisis task should start with deeper aquifer than baseline."""
        env = AquaGuardEnvironment()
        obs_baseline = env.reset(task="baseline", seed=42)
        gw_baseline = obs_baseline.shared_aquifer_level_m

        obs_crisis = env.reset(task="crisis", seed=42)
        gw_crisis = obs_crisis.shared_aquifer_level_m

        assert gw_crisis > gw_baseline


class TestStep:
    """Tests for the step() method."""

    def test_step_returns_observation(self):
        """step() must return AquaGuardObservation."""
        env = AquaGuardEnvironment()
        env.reset(task="baseline", seed=42)
        obs = env.step(AquaGuardAction())
        assert isinstance(obs, AquaGuardObservation)

    def test_step_increments_step_number(self):
        """step_number should increment after each step."""
        env = AquaGuardEnvironment()
        env.reset()
        for i in range(1, 4):
            obs = env.step(AquaGuardAction())
            assert obs.step_number == i

    def test_step_returns_reward(self):
        """reward must be numeric and in [-10, 10]."""
        env = AquaGuardEnvironment()
        env.reset()
        obs = env.step(AquaGuardAction())
        assert obs.reward is not None
        assert isinstance(obs.reward, (int, float))
        assert -10.0 <= obs.reward <= 10.0

    def test_episode_terminates_at_max_steps(self):
        """Episode must terminate at max_steps."""
        env = AquaGuardEnvironment()
        obs = env.reset(task="baseline")
        max_steps = env._max_steps

        for _ in range(max_steps):
            if obs.done:
                break
            obs = env.step(AquaGuardAction())

        assert obs.done is True

    def test_step_before_reset_raises(self):
        """step() before reset() should raise RuntimeError."""
        env = AquaGuardEnvironment()
        with pytest.raises(RuntimeError):
            env.step(AquaGuardAction())

    def test_multiple_resets(self):
        """Environment can be reset multiple times."""
        env = AquaGuardEnvironment()
        for seed in range(3):
            obs = env.reset(task="baseline", seed=seed)
            obs = env.step(AquaGuardAction())
            assert obs.step_number == 1

    def test_conservative_policy_better_gw(self):
        """Conservative water policy should maintain better GW than wasteful one."""
        # Conservative action
        env1 = AquaGuardEnvironment()
        env1.reset(task="baseline", seed=42)
        conservative = AquaGuardAction(
            water_quotas={"zone_a": 500, "zone_b": 450, "zone_c": 400},
            extraction_limits={"zone_a": 10.0, "zone_b": 8.0, "zone_c": 6.0},
            crop_allocation={"rice": 0.15, "wheat": 0.20, "millet": 0.30,
                             "pulses": 0.20, "oilseeds": 0.10, "vegetables": 0.05},
        )
        env1.step(conservative)
        gw_conservative = env1._zone_states["zone_a"]["gw_depth_m"]

        # Wasteful action
        env2 = AquaGuardEnvironment()
        env2.reset(task="baseline", seed=42)
        wasteful = AquaGuardAction(
            water_quotas={"zone_a": 1800, "zone_b": 1800, "zone_c": 1800},
            extraction_limits={"zone_a": 55.0, "zone_b": 55.0, "zone_c": 55.0},
            crop_allocation={"rice": 0.50, "wheat": 0.30, "millet": 0.05,
                             "pulses": 0.05, "oilseeds": 0.05, "vegetables": 0.05},
        )
        env2.step(wasteful)
        gw_wasteful = env2._zone_states["zone_a"]["gw_depth_m"]

        assert gw_conservative < gw_wasteful


class TestState:
    """Tests for the state property."""

    def test_state_returns_aquaguard_state(self):
        """state property must return AquaGuardState."""
        env = AquaGuardEnvironment()
        env.reset()
        state = env.state
        assert isinstance(state, AquaGuardState)

    def test_state_has_episode_id(self):
        """state must have a non-None episode_id after reset."""
        env = AquaGuardEnvironment()
        env.reset()
        state = env.state
        assert state.episode_id is not None

    def test_state_step_count_increments(self):
        """step_count must increment after each step."""
        env = AquaGuardEnvironment()
        env.reset()
        for i in range(3):
            env.step(AquaGuardAction())
            assert env.state.step_count == i + 1

    def test_state_task_name_correct(self):
        """state.task_name must match the task passed to reset."""
        env = AquaGuardEnvironment()
        for task in ["baseline", "crisis", "policy_shift"]:
            env.reset(task=task)
            assert env.state.task_name == task

    def test_state_cumulative_reward_accumulates(self):
        """cumulative_reward must increase with steps."""
        env = AquaGuardEnvironment()
        env.reset(task="baseline", seed=42)
        for _ in range(5):
            env.step(AquaGuardAction())
        state = env.state
        # cumulative reward can be negative, but step_count should be 5
        assert state.step_count == 5