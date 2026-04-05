# src/server/grader/programmatic.py
"""
Programmatic grader: 12 automated checks for AquaGuard-RL environment quality.

Checks verify:
1. API compliance (reset/step/state work correctly)
2. Reward properties (bounds, correlation)
3. Domain-specific thresholds (groundwater, food security, farmer income)
4. Episode management (done, step count)
5. Observation completeness

Each check has an associated weight; the final score is weighted_passed / total_weight.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Result of a single programmatic check."""
    name: str
    passed: bool
    weight: float
    details: str = ""


@dataclass
class ProgrammaticGradeResult:
    """Aggregate result from all 12 programmatic checks."""
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Weighted score: sum of passed weights / total weight. Range [0, 1]."""
        total_weight = sum(c.weight for c in self.checks)
        if total_weight == 0:
            return 0.0
        passed_weight = sum(c.weight for c in self.checks if c.passed)
        return passed_weight / total_weight

    @property
    def passed_count(self) -> int:
        """Number of passed checks."""
        return sum(1 for c in self.checks if c.passed)

    @property
    def total_count(self) -> int:
        """Total number of checks."""
        return len(self.checks)

    @property
    def summary(self) -> str:
        """Human-readable summary of all check results."""
        lines = [
            f"Programmatic Grade: {self.score:.3f} "
            f"({self.passed_count}/{self.total_count} checks passed)"
        ]
        for c in self.checks:
            icon = "✓" if c.passed else "✗"
            lines.append(f"  {icon} [{c.weight:.2f}] {c.name}: {c.details}")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "score": self.score,
            "passed_count": self.passed_count,
            "total_count": self.total_count,
            "checks": [
                {"name": c.name, "passed": c.passed, "weight": c.weight, "details": c.details}
                for c in self.checks
            ],
        }


class ProgrammaticGrader:
    """
    Runs 12 automated quality checks on AquaGuard-RL environment behavior.

    Check overview:
    | # | Name                    | Weight | What it tests                             |
    |---|-------------------------|--------|-------------------------------------------|
    | 1 | reset_returns_obs       | 0.10   | reset() returns valid AquaGuardObservation |
    | 2 | step_returns_obs        | 0.10   | step() returns valid AquaGuardObservation  |
    | 3 | state_valid             | 0.05   | state returns valid AquaGuardState         |
    | 4 | reward_bounds           | 0.10   | rewards in [−10, 10], no NaN/Inf          |
    | 5 | reward_correlation      | 0.10   | higher food security ↔ higher reward       |
    | 6 | groundwater_conserved   | 0.15   | ≥70% steps keep all zones below critical  |
    | 7 | food_security           | 0.10   | ≥70% steps meet food requirement          |
    | 8 | farmer_income           | 0.10   | ≥70% steps below 40% poverty              |
    | 9 | done_condition          | 0.05   | episode terminates correctly               |
    | 10| diversity_achievable    | 0.05   | Shannon diversity > 0.5 achievable         |
    | 11| step_count_monotonic    | 0.10   | step_count strictly increments             |
    | 12| observation_complete    | 0.10   | all required fields populated              |
    """

    def evaluate_episode(
        self,
        episode_observations: List,
        episode_actions: List,
        episode_states: List,
    ) -> ProgrammaticGradeResult:
        """
        Run all 12 checks on a complete or partial episode.

        Args:
            episode_observations: List of AquaGuardObservation objects (includes reset obs).
            episode_actions: List of AquaGuardAction objects taken.
            episode_states: List of AquaGuardState objects after each step.

        Returns:
            ProgrammaticGradeResult with all check results and composite score.
        """
        result = ProgrammaticGradeResult()
        obs = episode_observations
        acts = episode_actions
        states = episode_states

        result.checks.append(self._check_reset_returns_observation(obs))
        result.checks.append(self._check_step_returns_observation(obs, acts))
        result.checks.append(self._check_state_valid(states))
        result.checks.append(self._check_reward_bounds(obs))
        result.checks.append(self._check_reward_correlation(obs))
        result.checks.append(self._check_groundwater_conservation(obs))
        result.checks.append(self._check_food_security(obs))
        result.checks.append(self._check_farmer_income(obs))
        result.checks.append(self._check_done_condition(obs, states))
        result.checks.append(self._check_diversity_achievable(obs))
        result.checks.append(self._check_step_count_monotonic(states))
        result.checks.append(self._check_observation_completeness(obs))

        logger.info(f"Programmatic grade: {result.score:.3f} ({result.passed_count}/{result.total_count})")
        return result

    # ─── Individual check implementations ────────────────────────────────────

    def _check_reset_returns_observation(self, obs: List) -> CheckResult:
        """Check 1: reset() returns a valid AquaGuardObservation."""
        if not obs:
            return CheckResult("reset_returns_obs", False, 0.10, "No observations recorded")

        first = obs[0]
        try:
            valid = (
                hasattr(first, "step_number")
                and first.step_number == 0
                and hasattr(first, "season")
                and first.season in ["kharif", "rabi", "zaid"]
                and hasattr(first, "zones")
                and len(first.zones) == 3
                and hasattr(first, "crops")
                and len(first.crops) == 6
                and hasattr(first, "scenario_description")
                and len(first.scenario_description) > 10
            )
            return CheckResult(
                "reset_returns_obs", valid, 0.10,
                f"Initial obs: season={first.season}, zones={len(first.zones)}, crops={len(first.crops)}"
                if valid else f"Invalid initial observation: type={type(first).__name__}"
            )
        except Exception as e:
            return CheckResult("reset_returns_obs", False, 0.10, f"Error: {e}")

    def _check_step_returns_observation(self, obs: List, acts: List) -> CheckResult:
        """Check 2: step() returns a valid AquaGuardObservation with step_number=1."""
        if len(obs) < 2 or not acts:
            return CheckResult("step_returns_obs", False, 0.10, "Not enough steps recorded (need ≥2 obs)")
        try:
            step_obs = obs[1]
            valid = (
                hasattr(step_obs, "step_number")
                and step_obs.step_number == 1
                and hasattr(step_obs, "reward")
                and step_obs.reward is not None
                and isinstance(step_obs.reward, (int, float))
            )
            return CheckResult(
                "step_returns_obs", valid, 0.10,
                f"Step 1 obs: step_number={step_obs.step_number}, reward={step_obs.reward:.3f}"
                if valid else "Step observation missing or invalid"
            )
        except Exception as e:
            return CheckResult("step_returns_obs", False, 0.10, f"Error: {e}")

    def _check_state_valid(self, states: List) -> CheckResult:
        """Check 3: state property returns valid AquaGuardState."""
        if not states:
            return CheckResult("state_valid", False, 0.05, "No states recorded")
        try:
            s = states[-1]
            valid_tasks = {"baseline", "crisis", "policy_shift", "climate_shock", "multi_district"}
            valid = (
                hasattr(s, "episode_id")
                and s.episode_id is not None
                and hasattr(s, "step_count")
                and s.step_count >= 0
                and hasattr(s, "task_name")
                and s.task_name in valid_tasks
            )
            return CheckResult(
                "state_valid", valid, 0.05,
                f"State: episode_id={s.episode_id[:8]}..., step={s.step_count}, task={s.task_name}"
                if valid else f"Invalid state: {type(s).__name__}"
            )
        except Exception as e:
            return CheckResult("state_valid", False, 0.05, f"Error: {e}")

    def _check_reward_bounds(self, obs: List) -> CheckResult:
        """Check 4: All rewards are numeric, non-NaN/Inf, and in [−10, 10]."""
        rewards = [o.reward for o in obs if hasattr(o, "reward") and o.reward is not None]
        if not rewards:
            return CheckResult("reward_bounds", False, 0.10, "No rewards recorded (all None)")
        try:
            invalid = [
                r for r in rewards
                if not isinstance(r, (int, float))
                or math.isnan(r)
                or math.isinf(r)
                or r < -10.0
                or r > 10.0
            ]
            passed = len(invalid) == 0
            return CheckResult(
                "reward_bounds", passed, 0.10,
                f"All {len(rewards)} rewards valid ∈ [−10, 10]: "
                f"min={min(rewards):.3f}, max={max(rewards):.3f}"
                if passed else f"{len(invalid)} invalid rewards: {invalid[:3]}"
            )
        except Exception as e:
            return CheckResult("reward_bounds", False, 0.10, f"Error: {e}")

    def _check_reward_correlation(self, obs: List) -> CheckResult:
        """
        Check 5: Reward correlates positively with food security.

        Simplified check: average reward in top-half food security steps
        should be ≥ average reward in bottom-half steps.
        """
        step_obs = [o for o in obs if hasattr(o, "reward") and o.reward is not None]
        if len(step_obs) < 4:
            return CheckResult("reward_correlation", True, 0.10,
                               f"Too few steps ({len(step_obs)}) — pass by default")
        try:
            sorted_by_food = sorted(step_obs, key=lambda o: o.food_security_ratio)
            n = len(sorted_by_food)
            low_food_reward = sum(o.reward for o in sorted_by_food[:n // 2]) / max(n // 2, 1)
            high_food_reward = sum(o.reward for o in sorted_by_food[n // 2:]) / max(n - n // 2, 1)
            correlated = high_food_reward >= low_food_reward - 0.5  # allow small tolerance
            return CheckResult(
                "reward_correlation", correlated, 0.10,
                f"High-food avg reward={high_food_reward:.2f} vs low-food avg={low_food_reward:.2f}"
            )
        except Exception as e:
            return CheckResult("reward_correlation", False, 0.10, f"Error: {e}")

    def _check_groundwater_conservation(self, obs: List) -> CheckResult:
        """Check 6: ≥70% of steps maintain aquifer below critical threshold."""
        if not obs:
            return CheckResult("groundwater_conserved", False, 0.15, "No observations")
        try:
            danger_steps = sum(1 for o in obs if hasattr(o, "aquifer_danger_zone") and o.aquifer_danger_zone)
            total = len(obs)
            pass_rate = (total - danger_steps) / total
            passed = pass_rate >= 0.70
            return CheckResult(
                "groundwater_conserved", passed, 0.15,
                f"{total - danger_steps}/{total} steps in safe zone ({pass_rate:.0%})"
            )
        except Exception as e:
            return CheckResult("groundwater_conserved", False, 0.15, f"Error: {e}")

    def _check_food_security(self, obs: List) -> CheckResult:
        """Check 7: ≥70% of steps meet food security requirement (ratio ≥ 1.0)."""
        step_obs = [o for o in obs if hasattr(o, "reward") and o.reward is not None]
        if not step_obs:
            return CheckResult("food_security", False, 0.10, "No step observations recorded")
        try:
            meeting = sum(1 for o in step_obs if o.food_security_ratio >= 1.0)
            rate = meeting / len(step_obs)
            passed = rate >= 0.70
            return CheckResult(
                "food_security", passed, 0.10,
                f"{meeting}/{len(step_obs)} steps meet food requirement (ratio≥1.0) = {rate:.0%}"
            )
        except Exception as e:
            return CheckResult("food_security", False, 0.10, f"Error: {e}")

    def _check_farmer_income(self, obs: List) -> CheckResult:
        """Check 8: ≥70% of steps have <40% of farmers in poverty."""
        step_obs = [o for o in obs if hasattr(o, "reward") and o.reward is not None]
        if not step_obs:
            return CheckResult("farmer_income", False, 0.10, "No step observations recorded")
        try:
            passing = sum(1 for o in step_obs if o.percent_farmers_below_poverty < 40.0)
            rate = passing / len(step_obs)
            passed = rate >= 0.70
            return CheckResult(
                "farmer_income", passed, 0.10,
                f"{passing}/{len(step_obs)} steps below 40% poverty = {rate:.0%}"
            )
        except Exception as e:
            return CheckResult("farmer_income", False, 0.10, f"Error: {e}")

    def _check_done_condition(self, obs: List, states: List) -> CheckResult:
        """Check 9: Episode terminates correctly (done=True in last observation)."""
        if not obs:
            return CheckResult("done_condition", False, 0.05, "No observations")
        try:
            final_obs = obs[-1]
            done_triggered = hasattr(final_obs, "done") and final_obs.done
            return CheckResult(
                "done_condition", done_triggered, 0.05,
                f"Episode terminated (done=True) at step {final_obs.step_number}"
                if done_triggered else f"Episode did not terminate (done={final_obs.done})"
            )
        except Exception as e:
            return CheckResult("done_condition", False, 0.05, f"Error: {e}")

    def _check_diversity_achievable(self, obs: List) -> CheckResult:
        """Check 10: At least one step achieves Shannon diversity index > 0.5."""
        if not obs:
            return CheckResult("diversity_achievable", False, 0.05, "No observations")
        try:
            max_diversity = max((o.shannon_diversity_index for o in obs if hasattr(o, "shannon_diversity_index")), default=0.0)
            achieved = max_diversity > 0.5
            return CheckResult(
                "diversity_achievable", achieved, 0.05,
                f"Max Shannon diversity achieved: {max_diversity:.3f} (threshold: 0.5)"
            )
        except Exception as e:
            return CheckResult("diversity_achievable", False, 0.05, f"Error: {e}")

    def _check_step_count_monotonic(self, states: List) -> CheckResult:
        """Check 11: step_count strictly increments after each step."""
        if len(states) < 2:
            return CheckResult("step_count_monotonic", True, 0.10,
                               f"Too few states ({len(states)}) — pass by default")
        try:
            monotonic = all(
                states[i].step_count > states[i - 1].step_count
                for i in range(1, len(states))
            )
            step_counts = [s.step_count for s in states]
            return CheckResult(
                "step_count_monotonic", monotonic, 0.10,
                f"Step counts: {step_counts[:5]}{'...' if len(step_counts) > 5 else ''}"
                if monotonic else f"Non-monotonic step counts: {step_counts}"
            )
        except Exception as e:
            return CheckResult("step_count_monotonic", False, 0.10, f"Error: {e}")

    def _check_observation_completeness(self, obs: List) -> CheckResult:
        """Check 12: All observations have required fields populated."""
        if not obs:
            return CheckResult("observation_complete", False, 0.10, "No observations")
        try:
            incomplete = []
            for o in obs:
                if not (
                    hasattr(o, "scenario_description") and len(o.scenario_description) > 10
                    and hasattr(o, "zones") and len(o.zones) == 3
                    and hasattr(o, "crops") and len(o.crops) == 6
                    and hasattr(o, "season") and o.season in ["kharif", "rabi", "zaid"]
                    and hasattr(o, "food_security_ratio")
                    and hasattr(o, "shannon_diversity_index")
                ):
                    incomplete.append(o.step_number if hasattr(o, "step_number") else "?")

            passed = len(incomplete) == 0
            return CheckResult(
                "observation_complete", passed, 0.10,
                f"All {len(obs)} observations fully populated"
                if passed else f"Incomplete observations at steps: {incomplete}"
            )
        except Exception as e:
            return CheckResult("observation_complete", False, 0.10, f"Error: {e}")