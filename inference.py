#!/usr/bin/env python3
"""
AquaGuard-RL Inference Script — MANDATORY ROOT-LEVEL FILE
Meta PyTorch OpenEnv Hackathon Submission

Uses the OpenAI Client to run an LLM agent through multiple AquaGuard-RL tasks.
The LLM agent reads the natural language observation (scenario_description) and
decides on a policy action (crop allocation, water quotas, irrigation methods, etc.)

Environment variables (required):
    API_BASE_URL   — Base URL of the LLM API (e.g., "https://integrate.api.nvidia.com/v1")
    MODEL_NAME     — LLM model to use (e.g., "meta/llama-3.3-70b-instruct")
    HF_TOKEN       — Hugging Face token (used when API_BASE_URL points to HF Inference API)

Environment variables (optional):
    ENV_SERVER_URL — URL of running AquaGuard-RL server (default: http://localhost:8000)
    OPENAI_API_KEY — API key for LLM provider (falls back to HF_TOKEN)

Usage:
    # With NVIDIA NIM API (free — recommended):
    export API_BASE_URL="https://integrate.api.nvidia.com/v1"
    export MODEL_NAME="meta/llama-3.3-70b-instruct"
    export OPENAI_API_KEY="nvapi-..."
    python inference.py

    # With Hugging Face Inference API:
    export API_BASE_URL="https://api-inference.huggingface.co/v1"
    export MODEL_NAME="meta-llama/Llama-3-8B-Instruct"
    export HF_TOKEN="hf_..."
    python inference.py

    # With local server (no LLM — heuristic fallback):
    export ENV_SERVER_URL="http://localhost:8000"
    python inference.py --heuristic
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv(override=True)
from dataclasses import dataclass
from typing import Dict, List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from models import AquaGuardAction

# ─── Setup path ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Fix Windows console encoding for unicode characters
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("aquaguard.inference")


# ─── Configuration ────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/nemotron-3-super-120b-a12b")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or HF_TOKEN  # HF_TOKEN as fallback
ENV_SERVER_URL = os.getenv("ENV_SERVER_URL", "http://localhost:8000")

# Tasks to run (in order)
TASKS_TO_RUN = ["baseline", "crisis", "policy_shift"]

# Maximum steps per task (for inference — may differ from task max_steps)
MAX_STEPS_PER_TASK = {
    "baseline": 10,
    "crisis": 12,
    "policy_shift": 8,
    "climate_shock": 6,
    "multi_district": 15,
}

# Random seed for reproducibility
INFERENCE_SEED = 42


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    """Results from running one task."""
    task_name: str
    total_reward: float
    steps_completed: int
    final_food_ratio: float
    final_gw_depth: float
    final_poverty_pct: float
    final_shannon: float
    food_failures: int
    crisis_triggered: bool


# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert agricultural policy advisor for India's District Agricultural Commission.
You have deep knowledge of:
- India's groundwater crisis: aquifer depletion, CGWB guidelines, sustainable extraction
- Agricultural economics: MSP policies, crop subsidies, farmer income support
- Hydrology: monsoon patterns, rainfall variability, irrigation efficiency
- Crop science: rice/wheat water requirements, millet/pulse drought tolerance

You will receive a description of the current agricultural simulation state and must respond
with a JSON policy decision. Your goal is to balance four competing objectives:
1. Groundwater sustainability (prevent aquifer depletion)
2. Food security (maintain food production ratio ≥ 1.0)
3. Farmer welfare (keep poverty fraction below 35%)
4. Crop diversity (Shannon diversity index ≥ 1.0)

RESPOND ONLY WITH VALID JSON in this exact format:
{
  "crop_allocation": {
    "rice": <0.0-1.0>,
    "wheat": <0.0-1.0>,
    "millet": <0.0-1.0>,
    "pulses": <0.0-1.0>,
    "oilseeds": <0.0-1.0>,
    "vegetables": <0.0-1.0>
  },
  "water_quotas": {
    "zone_a": <0-2000>,
    "zone_b": <0-2000>,
    "zone_c": <0-2000>
  },
  "irrigation_methods": {
    "zone_a": "<flood|sprinkler|drip>",
    "zone_b": "<flood|sprinkler|drip>",
    "zone_c": "<flood|sprinkler|drip>"
  },
  "extraction_limits": {
    "zone_a": <0-60>,
    "zone_b": <0-60>,
    "zone_c": <0-60>
  },
  "subsidy_adjustments": {
    "rice": <-1.0 to 1.0>,
    "wheat": <-1.0 to 1.0>,
    "millet": <-1.0 to 1.0>,
    "pulses": <-1.0 to 1.0>,
    "oilseeds": <-1.0 to 1.0>,
    "vegetables": <-1.0 to 1.0>
  },
  "justification": "<150-400 word explanation of your policy decisions>"
}

CRITICAL CONSTRAINTS:
- crop_allocation values must sum to ≤ 1.0
- Water-intensive rice requires 1200mm/season; millet only 350mm
- Groundwater depth > 40m is critical; > 50m is catastrophic collapse
- A good justification explains the causal reasoning behind each decision"""


# ─── LLM Interface ────────────────────────────────────────────────────────────

class LLMAgent:
    """
    Agent that uses an LLM to decide policy actions based on scenario descriptions.
    """

    def __init__(self, use_heuristic: bool = False) -> None:
        """
        Initialize LLM agent.

        Args:
            use_heuristic: If True, skip LLM and use rule-based heuristic.
        """
        self._use_heuristic = use_heuristic
        self._client = None

        if not use_heuristic:
            self._client = self._init_client()

    def _init_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            api_key = OPENAI_API_KEY or HF_TOKEN
            kwargs: Dict[str, Any] = {}
            if api_key:
                kwargs["api_key"] = api_key
            if API_BASE_URL and API_BASE_URL != "https://api.openai.com/v1":
                kwargs["base_url"] = API_BASE_URL
            client = OpenAI(**kwargs)
            logger.info(f"OpenAI client initialized: model={MODEL_NAME}, base={API_BASE_URL}")
            return client
        except ImportError:
            logger.warning("openai package not installed. Install with: pip install openai")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            return None

    def decide(self, observation) -> "AquaGuardAction":
        """
        Decide on a policy action given the current observation.

        Args:
            observation: AquaGuardObservation from the environment.

        Returns:
            AquaGuardAction with policy decisions.
        """
        if self._use_heuristic or self._client is None:
            return self._heuristic_action(observation)

        try:
            action = self._llm_action(observation)
            if action is None:
                logger.warning("LLM returned unparseable action, falling back to heuristic")
                return self._heuristic_action(observation)
            return action
        except Exception as e:
            logger.warning(f"LLM action failed ({e}), falling back to heuristic")
            return self._heuristic_action(observation)

    def _llm_action(self, observation) -> "AquaGuardAction":
        """Call the LLM to get a policy action."""
        # Build user message from observation
        user_message = self._build_user_message(observation)

        request_kwargs = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.2,
            "max_tokens": 2048,
        }
        # Try JSON mode first (cleaner output), fall back without it
        try:
            request_kwargs["response_format"] = {"type": "json_object"}
            response = self._client.chat.completions.create(**request_kwargs)
        except Exception:
            del request_kwargs["response_format"]
            response = self._client.chat.completions.create(**request_kwargs)

        response_text = response.choices[0].message.content.strip()
        logger.debug(f"LLM response: {response_text[:200]}")

        return self._parse_action(response_text)

    def _build_user_message(self, observation) -> str:
        """Build the user message for the LLM from the observation."""
        return f"""Current simulation state:

{observation.scenario_description}

Additional details:
- Average groundwater depth: {observation.shared_aquifer_level_m:.1f}m
- Food security ratio: {observation.food_security_ratio:.3f}
- Farmers below poverty: {observation.percent_farmers_below_poverty:.1f}%
- Shannon diversity: {observation.shannon_diversity_index:.3f}
- Season: {observation.season}, Year: {observation.year}
- Step: {observation.step_number}

Please provide your policy action as JSON."""

    def _parse_action(self, text: str) -> "AquaGuardAction":
        """Parse LLM JSON response into an AquaGuardAction."""
        from models import AquaGuardAction
        import re

        # Extract JSON from response (handle markdown code blocks)
        text = re.sub(r"```(?:json)?\s*", "", text).strip()

        # Find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

        # Strip thinking tags (nemotron/qwen thinking models)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<\|think\|>.*?<\|/think\|>", "", text, flags=re.DOTALL)

        # Fix common LLM JSON issues
        text = re.sub(r",\s*}", "}", text)      # trailing comma before }
        text = re.sub(r",\s*]", "]", text)      # trailing comma before ]
        text = re.sub(r"//.*?\n", "\n", text)    # single-line comments
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)  # block comments
        # Replace single quotes with double quotes (but not within strings)
        text = text.replace("'", '"')

        try:
            data = json.loads(text)
            # Normalize allocation if it sums > 1.0
            alloc = data.get("crop_allocation", {})
            total = sum(alloc.values())
            if total > 1.001:
                alloc = {k: v / total * 0.99 for k, v in alloc.items()}
                data["crop_allocation"] = alloc

            return AquaGuardAction(**data)
        except Exception as e:
            logger.warning(f"Failed to parse LLM action: {e}. Using heuristic fallback.")
            return None  # Will trigger heuristic fallback in decide()

    def _heuristic_action(self, observation) -> "AquaGuardAction":
        """
        Rule-based heuristic action optimized for sustainable water balance.

        Key insight: Water quotas must be LOW enough that groundwater extraction
        does not exceed seasonal recharge. With natural recharge ~60mm/season and
        rainfall recharge ~12% of rainfall, total recharge is 60-156mm/season.
        Water quota * water_use_fraction * gwi_fraction must stay below this.

        Decision rules (3 tiers based on aquifer stress):
        - Tier 1 (stress > 80%): Emergency - minimize extraction, max diversity
        - Tier 2 (stress > 60%): Active conservation - low water, drip irrigation
        - Tier 3 (healthy): Proactive sustainability - moderate water, balanced crops
        """
        from models import AquaGuardAction

        gw = observation.shared_aquifer_level_m
        critical = observation.critical_aquifer_threshold_m
        stress_level = gw / critical if critical > 0 else 0.0

        food_ratio = observation.food_security_ratio
        poverty = observation.percent_farmers_below_poverty

        if stress_level > 0.80:
            # TIER 1: Emergency water conservation
            action = AquaGuardAction(
                crop_allocation={
                    "rice": 0.08, "wheat": 0.10, "millet": 0.25,
                    "pulses": 0.28, "oilseeds": 0.17, "vegetables": 0.12,
                },
                water_quotas={"zone_a": 200, "zone_b": 180, "zone_c": 150},
                irrigation_methods={"zone_a": "drip", "zone_b": "drip", "zone_c": "drip"},
                extraction_limits={"zone_a": 3.0, "zone_b": 2.5, "zone_c": 2.0},
                subsidy_adjustments={
                    "rice": -0.25, "wheat": -0.15, "millet": 0.25,
                    "pulses": 0.25, "oilseeds": 0.15, "vegetables": 0.10,
                },
                justification=(
                    f"EMERGENCY: Aquifer at {gw:.1f}m is {stress_level:.0%} of critical "
                    f"threshold ({critical:.0f}m). Implementing maximum water conservation. "
                    f"Slashing rice to 8% (saves ~1100mm/ha vs 30% allocation). Boosting "
                    f"pulses to 28% (high MSP at INR 71,500/t, only 300mm water needed) and "
                    f"millet to 25% (drought-tolerant, 350mm requirement). Water quotas reduced "
                    f"to 150-200mm across zones to keep extraction below recharge rate. "
                    f"Drip irrigation deployed for 45% water savings. Subsidy shifts make "
                    f"water-efficient crops economically attractive for farmers, reducing poverty "
                    f"while protecting the aquifer. Zone C gets lowest quota due to low storage "
                    f"coefficient (0.08) amplifying extraction impact. "
                    f"Current food ratio {food_ratio:.2f} provides buffer for crop transition."
                ),
            )

        elif stress_level > 0.60:
            # TIER 2: Active conservation
            action = AquaGuardAction(
                crop_allocation={
                    "rice": 0.12, "wheat": 0.14, "millet": 0.22,
                    "pulses": 0.25, "oilseeds": 0.15, "vegetables": 0.12,
                },
                water_quotas={"zone_a": 320, "zone_b": 280, "zone_c": 220},
                irrigation_methods={"zone_a": "drip", "zone_b": "drip", "zone_c": "drip"},
                extraction_limits={"zone_a": 5.0, "zone_b": 4.0, "zone_c": 3.0},
                subsidy_adjustments={
                    "rice": -0.15, "wheat": -0.08, "millet": 0.18,
                    "pulses": 0.20, "oilseeds": 0.12, "vegetables": 0.05,
                },
                justification=(
                    f"Aquifer at {gw:.1f}m ({stress_level:.0%} of {critical:.0f}m critical). "
                    f"Active conservation: reducing rice/wheat to 26% combined (from typical "
                    f"60%) and boosting water-efficient crops. Pulses at 25% provide high "
                    f"income (INR 71,500/t MSP) with minimal water (300mm). Water quotas "
                    f"capped at 220-320mm to maintain positive recharge balance. Drip "
                    f"irrigation across all zones reduces effective water use by 45%. "
                    f"Zone C (Rajasthan-type, storage coefficient 0.08) gets strictest limits "
                    f"because depth changes are amplified 12.5x per mm of water deficit. "
                    f"Subsidy adjustments incentivize farmer transition to drought-tolerant crops. "
                    f"Food security maintained via high-yield vegetables (20 t/ha base). "
                    f"Current poverty at {poverty:.0f}% -- targeting reduction through "
                    f"higher-value crop mix."
                ),
            )

        else:
            # TIER 3: Proactive sustainability (healthy aquifer)
            diversity = observation.shannon_diversity_index
            action = AquaGuardAction(
                crop_allocation={
                    "rice": 0.15, "wheat": 0.16, "millet": 0.20,
                    "pulses": 0.22, "oilseeds": 0.14, "vegetables": 0.13,
                },
                water_quotas={"zone_a": 400, "zone_b": 350, "zone_c": 280},
                irrigation_methods={
                    "zone_a": "sprinkler",
                    "zone_b": "drip",
                    "zone_c": "drip",
                },
                extraction_limits={"zone_a": 8.0, "zone_b": 6.0, "zone_c": 4.0},
                subsidy_adjustments={
                    "rice": -0.08, "wheat": -0.04, "millet": 0.12,
                    "pulses": 0.15, "oilseeds": 0.08, "vegetables": 0.04,
                },
                justification=(
                    f"Aquifer at {gw:.1f}m ({stress_level:.0%} of critical) -- proactive "
                    f"sustainability mode. Diversified allocation: rice/wheat at 31% combined "
                    f"(vs Green Revolution 60%), with pulses (22%), millet (20%), oilseeds "
                    f"(14%), vegetables (13%). This maximizes farmer income through high-value "
                    f"pulses (INR 71,500/t) and oilseeds (INR 58,500/t) while minimizing "
                    f"water demand. Water quotas set at 280-400mm -- below seasonal recharge "
                    f"rates to allow gradual aquifer recovery. Zone A uses sprinkler (30% "
                    f"water saving); Zones B/C use drip (45% saving) due to higher stress. "
                    f"Extraction limits binding at 4-8m/season to prevent overdraft. "
                    f"Shannon diversity at {diversity:.3f} -- targeting >1.2 for ecosystem "
                    f"resilience. Food ratio {food_ratio:.2f} maintained through vegetable "
                    f"allocation (20 t/ha yield). Poverty at {poverty:.0f}% expected to "
                    f"decrease as high-value crop income flows through."
                ),
            )

        return action


# ─── Main inference loop ──────────────────────────────────────────────────────

def run_task(
    agent: LLMAgent,
    server_url: str,
    task_name: str,
    seed: int = INFERENCE_SEED,
) -> TaskResult:
    """
    Run one full episode for a given task.

    Args:
        agent: LLM or heuristic agent.
        server_url: URL of running AquaGuard-RL server.
        task_name: Task to run.
        seed: Random seed.

    Returns:
        TaskResult with episode statistics.
    """
    from client import AquaGuardEnv

    logger.info(f"\n{'='*60}")
    logger.info(f"Starting task: {task_name} (seed={seed})")
    logger.info(f"{'='*60}")

    env = AquaGuardEnv(server_url)

    try:
        obs = env.reset(task=task_name, seed=seed)
    except Exception as e:
        logger.error(f"Failed to reset environment for task '{task_name}': {e}")
        raise

    # Structured START log (required by hackathon submission validator)
    print(f"[START] task={task_name} seed={seed}", flush=True)

    logger.info(f"Initial state: GW={obs.shared_aquifer_level_m:.1f}m | "
                f"food={obs.food_security_ratio:.2f} | "
                f"poverty={obs.percent_farmers_below_poverty:.1f}%")
    logger.info(f"Season: {obs.season} | Task: {obs.task_name}")
    logger.info(f"Scenario: {obs.scenario_description[:500]}...")

    total_reward = 0.0
    step = 0
    max_steps = MAX_STEPS_PER_TASK.get(task_name, 10)

    while not obs.done and step < max_steps:
        step += 1

        # Agent decides action
        t0 = time.time()
        action = agent.decide(obs)
        decision_time = time.time() - t0

        # Execute action
        obs = env.step(action)
        total_reward += obs.reward or 0.0

        # Structured STEP log (required by hackathon submission validator)
        print(f"[STEP] step={step} reward={obs.reward:+.4f} done={obs.done}", flush=True)

        logger.info(
            f"Step {step:2d} [{obs.season:6s}]: "
            f"reward={obs.reward:+.2f} | "
            f"GW={obs.shared_aquifer_level_m:.1f}m | "
            f"food={obs.food_security_ratio:.2f} | "
            f"poverty={obs.percent_farmers_below_poverty:.1f}% | "
            f"H={obs.shannon_diversity_index:.3f} | "
            f"LLM={decision_time:.1f}s"
        )

        if obs.grader_feedback:
            logger.info(f"   Grader: {obs.grader_feedback[:120]}")

    # Final state
    try:
        state = env.state
        food_failures = state.food_security_failures
        crisis_triggered = state.groundwater_crisis_triggered
    except Exception:
        food_failures = 0
        crisis_triggered = False

    result = TaskResult(
        task_name=task_name,
        total_reward=total_reward,
        steps_completed=step,
        final_food_ratio=obs.food_security_ratio,
        final_gw_depth=obs.shared_aquifer_level_m,
        final_poverty_pct=obs.percent_farmers_below_poverty,
        final_shannon=obs.shannon_diversity_index,
        food_failures=food_failures,
        crisis_triggered=crisis_triggered,
    )

    # Compute a normalized score (0.0 to 1.0) from total_reward for the validator
    max_possible = max_steps * 10.0  # theoretical max reward
    score = max(0.0, min(1.0, (total_reward + max_possible) / (2 * max_possible)))

    # Structured END log (required by hackathon submission validator)
    print(f"[END] task={task_name} score={score:.4f} steps={step}", flush=True)

    logger.info(f"\nTask '{task_name}' complete:")
    logger.info(f"  Total reward: {total_reward:.2f}")
    logger.info(f"  Steps completed: {step}")
    logger.info(f"  Final GW depth: {obs.shared_aquifer_level_m:.1f}m")
    logger.info(f"  Food security failures: {food_failures}")
    logger.info(f"  Crisis triggered: {crisis_triggered}")

    env.close()
    return result


def main(args: argparse.Namespace) -> None:
    """Main inference entry point."""
    logger.info("AquaGuard-RL Inference — Meta PyTorch OpenEnv Hackathon")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"API Base: {API_BASE_URL}")
    logger.info(f"Server: {ENV_SERVER_URL}")
    logger.info(f"Tasks: {TASKS_TO_RUN}")

    # Check server availability
    try:
        import httpx
        resp = httpx.get(f"{ENV_SERVER_URL}/health", timeout=5.0)
        if resp.status_code != 200:
            logger.error(f"Server health check failed: {resp.status_code}")
            sys.exit(1)
        logger.info(f"Server health: {resp.json()}")
    except Exception as e:
        logger.error(
            f"Cannot connect to environment server at {ENV_SERVER_URL}: {e}\n"
            "Start the server first:\n"
            "  python -m uvicorn server.app:app --host 0.0.0.0 --port 8000\n"
            "  OR: docker run -p 8000:8000 aquaguard-env:latest"
        )
        sys.exit(1)

    # Initialize agent
    agent = LLMAgent(use_heuristic=args.heuristic)

    # Run tasks
    all_results: List[TaskResult] = []
    for task_name in TASKS_TO_RUN:
        try:
            result = run_task(agent, ENV_SERVER_URL, task_name, seed=INFERENCE_SEED)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Task '{task_name}' failed: {e}", exc_info=True)

    # Print final summary to stderr (keeps stdout clean for validator)
    total_cumulative = 0.0
    for r in all_results:
        total_cumulative += r.total_reward

    logger.info("")
    logger.info("=" * 70)
    logger.info("INFERENCE COMPLETE -- FINAL SCORES")
    logger.info("=" * 70)
    logger.info(f"{'Task':<20} {'Reward':>10} {'Steps':>7} {'GW(m)':>8} {'Food':>7} {'Poverty%':>10} {'Shannon':>9}")
    logger.info("-" * 70)
    for r in all_results:
        crisis_marker = " [!]" if r.crisis_triggered else ""
        logger.info(
            f"{r.task_name:<20} {r.total_reward:>10.2f} {r.steps_completed:>7} "
            f"{r.final_gw_depth:>8.1f} {r.final_food_ratio:>7.2f} "
            f"{r.final_poverty_pct:>10.1f} {r.final_shannon:>9.3f}{crisis_marker}"
        )
    logger.info("-" * 70)
    logger.info(f"{'TOTAL':<20} {total_cumulative:>10.2f}")
    logger.info("=" * 70)

    # Save results
    results_path = os.path.join(os.path.dirname(__file__), "inference_results.json")
    try:
        with open(results_path, "w") as f:
            json.dump(
                {
                    "model": MODEL_NAME,
                    "api_base": API_BASE_URL,
                    "tasks": [
                        {
                            "task": r.task_name,
                            "total_reward": r.total_reward,
                            "steps": r.steps_completed,
                            "final_gw_depth_m": r.final_gw_depth,
                            "final_food_ratio": r.final_food_ratio,
                            "final_poverty_pct": r.final_poverty_pct,
                            "final_shannon": r.final_shannon,
                            "food_failures": r.food_failures,
                            "crisis_triggered": r.crisis_triggered,
                        }
                        for r in all_results
                    ],
                    "total_cumulative_reward": total_cumulative,
                },
                f,
                indent=2,
            )
        logger.info(f"Results saved to: {results_path}")
    except Exception as e:
        logger.warning(f"Failed to save results: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AquaGuard-RL Inference — Run LLM agent through agricultural policy tasks"
    )
    parser.add_argument(
        "--heuristic",
        action="store_true",
        help="Use rule-based heuristic instead of LLM (for testing without API access)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=TASKS_TO_RUN,
        help=f"Tasks to run (default: {TASKS_TO_RUN})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=INFERENCE_SEED,
        help=f"Random seed (default: {INFERENCE_SEED})",
    )
    args = parser.parse_args()

    if args.tasks:
        TASKS_TO_RUN[:] = args.tasks

    main(args)