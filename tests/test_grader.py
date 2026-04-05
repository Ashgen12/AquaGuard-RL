# tests/test_grader.py
"""Tests for programmatic and LLM graders."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import AquaGuardAction, AquaGuardState
from server.grader.programmatic import ProgrammaticGrader, CheckResult
from server.grader.llm_grader import LLMGrader
from server.aquaguard_environment import AquaGuardEnvironment


def run_episode(task="baseline", steps=3, seed=42):
    """Helper to run a short episode and collect data."""
    env = AquaGuardEnvironment()
    obs = env.reset(task=task, seed=seed)
    observations = [obs]
    actions = []
    states = []

    for _ in range(steps):
        if obs.done:
            break
        action = AquaGuardAction()
        obs = env.step(action)
        observations.append(obs)
        actions.append(action)
        states.append(env.state)

    return observations, actions, states


class TestProgrammaticGrader:
    """Tests for ProgrammaticGrader."""

    def setup_method(self):
        self.grader = ProgrammaticGrader()

    def test_grader_runs_without_error(self):
        """Grader should run without exceptions on a complete episode."""
        obs, acts, states = run_episode(task="baseline", steps=5)
        result = self.grader.evaluate_episode(obs, acts, states)
        assert result is not None

    def test_grader_returns_12_checks(self):
        """Grader must run exactly 12 checks."""
        obs, acts, states = run_episode(steps=5)
        result = self.grader.evaluate_episode(obs, acts, states)
        assert result.total_count == 12

    def test_grader_score_in_range(self):
        """Grader score must be in [0, 1]."""
        obs, acts, states = run_episode(steps=5)
        result = self.grader.evaluate_episode(obs, acts, states)
        assert 0.0 <= result.score <= 1.0

    def test_grader_check_reset_obs_passes(self):
        """Check 1 (reset returns observation) should pass for valid episode."""
        obs, acts, states = run_episode(steps=1)
        result = self.grader.evaluate_episode(obs, acts, states)
        reset_check = next(c for c in result.checks if c.name == "reset_returns_obs")
        assert reset_check.passed

    def test_grader_check_step_obs_passes(self):
        """Check 2 (step returns observation) should pass after 1 step."""
        obs, acts, states = run_episode(steps=2)
        result = self.grader.evaluate_episode(obs, acts, states)
        step_check = next(c for c in result.checks if c.name == "step_returns_obs")
        assert step_check.passed

    def test_grader_check_reward_bounds_passes(self):
        """Check 4 (reward bounds) should pass for all valid rewards."""
        obs, acts, states = run_episode(steps=5)
        result = self.grader.evaluate_episode(obs, acts, states)
        bounds_check = next(c for c in result.checks if c.name == "reward_bounds")
        assert bounds_check.passed

    def test_grader_check_step_count_monotonic(self):
        """Check 11 (step count monotonic) should pass."""
        obs, acts, states = run_episode(steps=5)
        result = self.grader.evaluate_episode(obs, acts, states)
        mono_check = next(c for c in result.checks if c.name == "step_count_monotonic")
        assert mono_check.passed

    def test_grader_summary_non_empty(self):
        """Grader summary should be a non-empty string."""
        obs, acts, states = run_episode(steps=3)
        result = self.grader.evaluate_episode(obs, acts, states)
        assert len(result.summary) > 50

    def test_grader_to_dict(self):
        """Grader result should serialize to dict."""
        obs, acts, states = run_episode(steps=3)
        result = self.grader.evaluate_episode(obs, acts, states)
        d = result.to_dict()
        assert "score" in d
        assert "checks" in d
        assert len(d["checks"]) == 12

    def test_grader_with_no_observations(self):
        """Grader should handle empty observation list gracefully."""
        result = self.grader.evaluate_episode([], [], [])
        assert result.score >= 0.0  # Should not crash

    def test_done_check_passes_at_end(self):
        """Done condition check should pass when episode terminates."""
        env = AquaGuardEnvironment()
        obs = env.reset(task="baseline", seed=42)
        observations = [obs]
        actions = []
        states = []

        max_s = env._max_steps
        while not obs.done:
            action = AquaGuardAction()
            obs = env.step(action)
            observations.append(obs)
            actions.append(action)
            states.append(env.state)

        result = self.grader.evaluate_episode(observations, actions, states)
        done_check = next(c for c in result.checks if c.name == "done_condition")
        assert done_check.passed


class TestLLMGrader:
    """Tests for LLMGrader (heuristic fallback only — no LLM API needed)."""

    def setup_method(self):
        self.grader = LLMGrader()

    def test_empty_justification_gives_low_score(self):
        """Empty or very short justification should score low."""
        obs, acts, states = run_episode(steps=1)
        obs_single = obs[1]

        action = AquaGuardAction(justification="")
        result = self.grader.score_justification("", obs_single, action)
        assert result.overall_score <= 2.0

    def test_heuristic_scores_with_keywords(self):
        """Justification with domain keywords should score higher than generic."""
        obs, acts, states = run_episode(steps=1)
        obs_single = obs[1]
        action = AquaGuardAction()

        # Generic justification
        generic_result = self.grader._heuristic_score(
            "I think this is a good decision.", obs_single, action
        )

        # Domain-rich justification
        domain_result = self.grader._heuristic_score(
            "Reducing rice allocation because it requires 1200mm per season "
            "while millet only needs 350mm, relieving aquifer extraction pressure. "
            "Deploying drip irrigation to save water. This trade-off between food "
            "security and groundwater sustainability is managed by increasing "
            "millet MSP subsidy to compensate farmers for income reduction.",
            obs_single, action,
        )

        assert domain_result.mean_score > generic_result.mean_score

    def test_heuristic_result_scores_in_range(self):
        """All heuristic scores should be in [0, 1] range (normalized)."""
        obs, acts, states = run_episode(steps=1)
        obs_single = obs[1]
        action = AquaGuardAction()

        result = self.grader._heuristic_score(
            "Aquifer is declining. Reduce rice and use drip irrigation because "
            "groundwater depletion is the main risk. Trade-off: food security "
            "may decrease slightly but this avoids collapse.",
            obs_single, action,
        )
        for score in [result.causal_reasoning, result.domain_knowledge,
                      result.policy_coherence, result.risk_awareness]:
            assert 0.0 <= score <= 1.0

    def test_parse_response_valid_json(self):
        """Valid JSON response should parse correctly (normalized to 0-1)."""
        valid_response = '''
        {
            "causal_reasoning": 8,
            "tradeoff_acknowledgment": 7,
            "domain_knowledge": 9,
            "policy_coherence": 8,
            "risk_awareness": 7,
            "overall_score": 7.8,
            "critique": "Good reasoning with clear causal logic."
        }'''
        result = self.grader._parse_response(valid_response)
        assert result.causal_reasoning == 0.8  # 8/10 normalized
        assert result.domain_knowledge == 0.9  # 9/10 normalized
        assert len(result.critique) > 0

    def test_parse_response_invalid_json_returns_neutral(self):
        """Invalid JSON should return neutral score without crashing."""
        result = self.grader._parse_response("This is not JSON {{{{")
        assert result.overall_score == 0.5  # neutral on 0-1 scale
        assert result.error is not None