# src/server/grader/llm_grader.py
"""
LLM-based grader for AquaGuard-RL.

Evaluates the quality of an agent's policy justification using an LLM.
Scores the reasoning across 5 dimensions (1-10 each, normalized to 0.0-1.0):
    1. Causal Reasoning  — correct cause→effect chains
    2. Trade-off Acknowledgment — recognizes competing objectives
    3. Domain Knowledge — understands Indian agri-economics
    4. Policy Coherence — actions match stated strategy
    5. Risk Awareness — identifies and mitigates risks

Supports:
    - NVIDIA NIM API (primary, free — https://build.nvidia.com)
    - OpenAI API (alternative)
    - Rule-based heuristic fallback (if no LLM available)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# ─── Prompt Templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert agricultural policy evaluator with deep domain knowledge in:
- India's groundwater crisis and Central Ground Water Board (CGWB) guidelines
- Agricultural economics: MSP policy, farm subsidies, crop diversification incentives
- Hydrology: aquifer dynamics, recharge rates, sustainable extraction principles
- Multi-objective policy optimization and trade-off management
- Farmer welfare, poverty measurement, and rural India economics (NSSO data)

You will be given:
1. The current state of a multi-district agricultural simulation
2. The policy decisions made by an AI agent
3. The agent's natural-language justification for those decisions

Evaluate the justification on 5 dimensions, scoring each 1-10.
Respond ONLY with valid JSON in this exact format:
{
  "causal_reasoning": <int 1-10>,
  "tradeoff_acknowledgment": <int 1-10>,
  "domain_knowledge": <int 1-10>,
  "policy_coherence": <int 1-10>,
  "risk_awareness": <int 1-10>,
  "overall_score": <float>,
  "critique": "<2-3 sentence evaluation>"
}

Scoring guide:
- 9-10: Expert-level reasoning that could come from a domain specialist
- 7-8: Good reasoning with correct causal logic and minor gaps
- 5-6: Adequate but superficial; some correct points, misses key trade-offs
- 3-4: Generic statements without domain understanding; incorrect assumptions
- 1-2: Absent, contradictory, or factually wrong reasoning"""

USER_PROMPT_TEMPLATE = """Current Simulation State:
- Season: {season}, Year {year}
- Shared aquifer depth: {gw_depth:.1f}m (critical threshold: {critical:.1f}m)
- Food security ratio: {food_ratio:.2f} (1.0 = exactly meeting requirements)
- Farmers below poverty line: {poverty:.1f}%
- Crop diversity (Shannon index): {shannon:.3f} (max=1.79 for 6 equal crops)
- Rainfall forecast: {rainfall:.0f}mm (season historical average: {season_avg:.0f}mm)
- Zones in danger zone: {danger_zones}

Agent's Policy Decisions:
- Crop allocation: {crop_alloc}
- Water quotas per zone (mm/season): {water_quotas}
- Irrigation methods: {irrigation}
- Groundwater extraction limits (m/season): {extraction}
- Subsidy adjustments: {subsidies}

Agent's Justification:
"{justification}"

Evaluate this justification using the scoring rubric."""


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class LLMGradeResult:
    """Result from LLM grader evaluation. All scores normalized to [0.0, 1.0]."""
    causal_reasoning: float = 0.0
    tradeoff_acknowledgment: float = 0.0
    domain_knowledge: float = 0.0
    policy_coherence: float = 0.0
    risk_awareness: float = 0.0
    overall_score: float = 0.0
    critique: str = ""
    error: Optional[str] = None

    @property
    def mean_score(self) -> float:
        """Mean of the 5 dimension scores (already in [0.0, 1.0])."""
        scores = [
            self.causal_reasoning,
            self.tradeoff_acknowledgment,
            self.domain_knowledge,
            self.policy_coherence,
            self.risk_awareness,
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "causal_reasoning": self.causal_reasoning,
            "tradeoff_acknowledgment": self.tradeoff_acknowledgment,
            "domain_knowledge": self.domain_knowledge,
            "policy_coherence": self.policy_coherence,
            "risk_awareness": self.risk_awareness,
            "overall_score": self.overall_score,
            "mean_score": self.mean_score,
            "critique": self.critique,
        }


# ─── Grader Class ─────────────────────────────────────────────────────────────

class LLMGrader:
    """
    Evaluates agent policy justifications using an LLM.

    Provider priority:
    1. NVIDIA NIM (default — free, OpenAI-compatible at integrate.api.nvidia.com/v1)
    2. OpenAI (if API_BASE_URL points to OpenAI)
    3. Heuristic fallback (keyword-based scoring, always available)
    """

    def __init__(self) -> None:
        """Initialize with provider detection from environment."""
        self._client = None
        self._model = os.getenv("LLM_GRADER_MODEL", "nvidia/nemotron-3-super-120b-a12b")
        self._provider = os.getenv("LLM_GRADER_PROVIDER", "nvidia")  # nvidia, openai, or heuristic

    def _get_openai_client(self):
        """Lazily initialize OpenAI-compatible client for grading.

        Uses LLM_GRADER_API_KEY first (dedicated grader key),
        falls back to OPENAI_API_KEY then HF_TOKEN.
        """
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # type: ignore
            api_key = (
                os.getenv("LLM_GRADER_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or os.getenv("HF_TOKEN")
            )
            base_url = os.getenv("API_BASE_URL", "https://integrate.api.nvidia.com/v1")
            if not api_key:
                logger.warning("No API key found (LLM_GRADER_API_KEY/OPENAI_API_KEY/HF_TOKEN). LLM grader disabled.")
                return None
            self._client = OpenAI(api_key=api_key, base_url=base_url)
            return self._client
        except (ImportError, Exception) as e:
            logger.warning(f"OpenAI-compatible client unavailable: {e}")
            return None

    def score_justification(
        self,
        justification: str,
        observation,
        action,
    ) -> LLMGradeResult:
        """
        Score the agent's policy justification.

        Args:
            justification: Agent's free-text explanation (from action.justification).
            observation: Current AquaGuardObservation (for context).
            action: AquaGuardAction taken (for context).

        Returns:
            LLMGradeResult with dimension scores and critique text.
        """
        if not justification or len(justification.strip()) < 20:
            return LLMGradeResult(
                overall_score=1.0,
                critique="No meaningful justification provided. Agent should explain reasoning.",
            )

        # Build context strings
        danger_zones = [
            zid for zid, zobs in observation.zones.items()
            if hasattr(zobs, "is_in_danger_zone") and zobs.is_in_danger_zone
        ]
        season_avg = {"kharif": 800, "rabi": 120, "zaid": 30}.get(observation.season, 500)

        user_prompt = USER_PROMPT_TEMPLATE.format(
            season=observation.season,
            year=observation.year,
            gw_depth=observation.shared_aquifer_level_m,
            critical=observation.critical_aquifer_threshold_m,
            food_ratio=observation.food_security_ratio,
            poverty=observation.percent_farmers_below_poverty,
            shannon=observation.shannon_diversity_index,
            rainfall=observation.rainfall_forecast_mm,
            season_avg=season_avg,
            danger_zones=", ".join(danger_zones) if danger_zones else "None",
            crop_alloc=json.dumps(action.crop_allocation),
            water_quotas=json.dumps(action.water_quotas),
            irrigation=json.dumps(action.irrigation_methods),
            extraction=json.dumps(action.extraction_limits),
            subsidies=json.dumps(action.subsidy_adjustments),
            justification=justification[:600],  # Truncate for token efficiency
        )

        # Try LLM providers in order
        if self._provider in ("openai", "nvidia"):
            result = self._score_with_openai(user_prompt)
            if result is not None:
                return result

        # Fallback: heuristic scoring
        logger.info("LLM unavailable — using heuristic fallback grader")
        return self._heuristic_score(justification, observation, action)

    def _score_with_openai(self, user_prompt: str) -> Optional[LLMGradeResult]:
        """Attempt to score using OpenAI-compatible API (NVIDIA NIM, OpenAI, etc.)."""
        client = self._get_openai_client()
        if client is None:
            return None
        try:
            # Build request kwargs — some providers don't support response_format
            request_kwargs = {
                "model": os.getenv("LLM_GRADER_MODEL", self._model),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 500,
            }
            # Try with JSON mode first, fall back without it
            try:
                request_kwargs["response_format"] = {"type": "json_object"}
                resp = client.chat.completions.create(**request_kwargs)
            except Exception:
                del request_kwargs["response_format"]
                resp = client.chat.completions.create(**request_kwargs)
            return self._parse_response(resp.choices[0].message.content)
        except Exception as e:
            logger.warning(f"LLM grader failed: {e}")
            return None


    def _parse_response(self, text: str) -> LLMGradeResult:
        """Parse LLM JSON response into LLMGradeResult. Normalizes 1-10 scores to [0.0, 1.0]."""
        try:
            # Strip markdown code blocks if present
            text = re.sub(r"```(?:json)?\s*", "", text).strip()
            data = json.loads(text)
            # LLM returns 1-10 scores; normalize to [0.0, 1.0]
            result = LLMGradeResult(
                causal_reasoning=float(data.get("causal_reasoning", 5)) / 10.0,
                tradeoff_acknowledgment=float(data.get("tradeoff_acknowledgment", 5)) / 10.0,
                domain_knowledge=float(data.get("domain_knowledge", 5)) / 10.0,
                policy_coherence=float(data.get("policy_coherence", 5)) / 10.0,
                risk_awareness=float(data.get("risk_awareness", 5)) / 10.0,
                overall_score=0.0,  # will be recomputed
                critique=str(data.get("critique", "")),
            )
            # Overall score = mean of normalized dimension scores
            result.overall_score = result.mean_score
            return result
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse LLM response: {e}\nRaw response: {text[:200]}")
            return LLMGradeResult(
                overall_score=0.5,
                critique="Grader parsing error — defaulting to neutral score.",
                error=str(e),
            )

    def _heuristic_score(
        self,
        justification: str,
        observation,
        action,
    ) -> LLMGradeResult:
        """
        Keyword-based heuristic fallback grader.

        Checks for presence of domain-relevant terms and reasoning quality signals
        when no LLM is available.
        """
        text = justification.lower()

        # Domain knowledge keywords
        domain_keywords = [
            "aquifer", "groundwater", "recharge", "monsoon", "kharif", "rabi",
            "msp", "subsidy", "irrigation", "drip", "sprinkler", "flood",
            "food security", "poverty", "diversity", "rice", "wheat", "millet",
            "water stress", "depletion", "cgwb", "extraction",
        ]
        trade_off_keywords = [
            "however", "but", "trade-off", "balance", "compromise", "despite",
            "while", "although", "at the cost", "to avoid", "in order to",
        ]
        causal_keywords = [
            "because", "therefore", "since", "as a result", "which means",
            "leads to", "causes", "due to", "will reduce", "will increase",
        ]
        risk_keywords = [
            "risk", "danger", "threshold", "critical", "prevent", "avoid",
            "forecast", "shortage", "crisis", "collapse", "drought",
        ]

        def keyword_score(keywords: list, text: str, max_score: float = 1.0) -> float:
            """Keyword-based scoring, normalized to [0.0, 1.0]."""
            matches = sum(1 for kw in keywords if kw in text)
            score = min(max_score, 0.2 + matches * 0.1)  # base 0.2, +0.1 per keyword
            return score

        domain_score = keyword_score(domain_keywords, text)
        tradeoff_score = keyword_score(trade_off_keywords, text, 0.8)
        causal_score = keyword_score(causal_keywords, text, 0.8)
        risk_score = keyword_score(risk_keywords, text, 0.8)

        # Policy coherence: check if justification mentions what was actually done
        alloc_mentioned = any(
            crop in text for crop in action.crop_allocation.keys()
        )
        policy_score = 0.6 if alloc_mentioned else 0.3

        result = LLMGradeResult(
            causal_reasoning=causal_score,
            tradeoff_acknowledgment=tradeoff_score,
            domain_knowledge=domain_score,
            policy_coherence=policy_score,
            risk_awareness=risk_score,
        )
        result.overall_score = result.mean_score
        result.critique = (
            f"Heuristic evaluation: {len(justification.split())} words. "
            f"Domain terms: {sum(1 for kw in domain_keywords if kw in text)}/24. "
            "LLM grader unavailable — using keyword analysis."
        )
        return result