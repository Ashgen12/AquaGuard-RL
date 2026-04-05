# src/server/aquaguard_environment.py
"""
Main environment class for AquaGuard-RL.

Implements the OpenEnv Environment interface with three core methods:
    - reset(seed, episode_id, task) → AquaGuardObservation
    - step(action) → AquaGuardObservation
    - state → AquaGuardState

Coordinates all simulation components:
    GroundwaterModel + CropGrowthModel + EconomicModel + SeasonManager → RewardCalculator
"""

from __future__ import annotations

import math
import uuid
import logging
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)

# ─── OpenEnv compatibility layer ──────────────────────────────────────────────
try:
    from openenv_core import Environment as _BaseEnvironment  # type: ignore
    _OPENENV_AVAILABLE = True
except ImportError:
    _OPENENV_AVAILABLE = False

    class _BaseEnvironment:
        """Standalone environment base class."""
        pass

from models import (
    AquaGuardAction,
    AquaGuardObservation,
    AquaGuardState,
    ZoneObservation,
    CropObservation,
)
from constants import (
    ZONE_DATA,
    CROP_DATA,
    IRRIGATION_EFFICIENCY,
    SEASONS,
    POVERTY_LINE_INR_PER_YEAR,
    DEFAULT_REWARD_WEIGHTS,
    RAINFALL_RECHARGE_FRACTION,
    FOOD_REQUIREMENT_KG_PER_PERSON_YEAR,
    HOUSEHOLD_SIZE,
    CATASTROPHIC_GW_DEPTH_M,
    FAMINE_CONSECUTIVE_FAILURES,
)
from server.simulation.groundwater import GroundwaterModel
from server.simulation.crop_growth import CropGrowthModel
from server.simulation.economic import EconomicModel
from server.simulation.season import SeasonManager
from server.reward import RewardCalculator
from server.tasks.task_definitions import TASK_CONFIGS, get_task_config
from server.utils.description_builder import build_scenario_description


class AquaGuardEnvironment(_BaseEnvironment):
    """
    AquaGuard-RL: India Groundwater & Agricultural Resource Management Environment.

    An LLM agent plays the role of a District Agricultural Commissioner making
    seasonal policy decisions across a 3-zone agricultural system. The agent must
    balance four competing objectives:
        1. Groundwater sustainability (prevent aquifer depletion)
        2. Food security (maintain production ≥ district requirement)
        3. Farmer welfare (keep income above poverty line)
        4. Crop diversity (encourage water-efficient crop mix)

    The environment is TEXT-BASED with natural language observations.
    All state is described in the AquaGuardObservation.scenario_description field.

    API:
        obs = env.reset(task="baseline", seed=42)
        obs = env.step(AquaGuardAction(...))
        state = env.state  # episode metadata

    See README.md for complete documentation.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        """Initialize environment with all simulation components."""
        if _OPENENV_AVAILABLE:
            super().__init__()

        # Episode state
        self._episode_id: Optional[str] = None
        self._step_count: int = 0
        self._task_name: str = "baseline"
        self._task_config: Dict[str, Any] = {}
        self._max_steps: int = 10
        self._cumulative_reward: float = 0.0

        # Simulation components
        self._gw_model = GroundwaterModel()
        self._crop_model = CropGrowthModel()
        self._eco_model = EconomicModel()
        self._season_mgr = SeasonManager()
        self._reward_calc = RewardCalculator()

        # Mutable simulation state
        self._zone_states: Dict[str, Dict] = {}
        self._crop_states: Dict[str, Dict] = {}
        self._irrigation_methods: Dict[str, str] = {}
        self._irrigation_transition_countdown: Dict[str, int] = {}

        # Episode metrics
        self._food_security_failures: int = 0
        self._income_failures: int = 0
        self._crisis_triggered: bool = False
        self._best_shannon_diversity: float = 0.0
        self._aquifer_recovery_steps: int = 0
        self._consecutive_food_failures: int = 0

        # Previous observation (for reward delta computation)
        self._prev_obs: Optional[AquaGuardObservation] = None

    # ─── Public API ────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: str = "baseline",
        **kwargs: Any,
    ) -> AquaGuardObservation:
        """
        Initialize a new episode.

        Args:
            seed: Random seed for reproducibility.
            episode_id: Optional explicit episode ID (UUID4 generated if None).
            task: Task configuration name. One of:
                'baseline', 'crisis', 'policy_shift', 'climate_shock', 'multi_district'.

        Returns:
            Initial AquaGuardObservation with step_number=0, reward=None, done=False.
        """
        import random, numpy as np

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Initialize episode state
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._task_name = task
        self._task_config = get_task_config(task)
        self._max_steps = self._task_config.get("max_steps", 10)
        self._cumulative_reward = 0.0
        self._food_security_failures = 0
        self._income_failures = 0
        self._crisis_triggered = False
        self._best_shannon_diversity = 0.0
        self._aquifer_recovery_steps = 0
        self._consecutive_food_failures = 0

        tc = self._task_config

        # Initialize zone states from task configuration
        self._zone_states = {}
        for zone_id, zone_def in ZONE_DATA.items():
            initial_depth = tc.get(f"{zone_id}_gw_depth", zone_def["initial_gw_depth_m"])
            self._zone_states[zone_id] = {
                "gw_depth_m": float(initial_depth),
                "soil_fertility": zone_def["baseline_soil_fertility"],
                "soil_salinity": zone_def["baseline_soil_salinity"],
                "arable_land_ha": float(zone_def["arable_land_ha"]),
                "water_used_mm": 0.0,
                "is_collapsed": False,
                "yield_t_per_ha": {},  # per-crop yield cache
            }
            self._irrigation_methods[zone_id] = "flood"
            self._irrigation_transition_countdown[zone_id] = 0

        # Initialize crop states
        initial_alloc = tc.get("initial_allocation", {
            "rice": 0.30, "wheat": 0.30, "millet": 0.15,
            "pulses": 0.15, "oilseeds": 0.07, "vegetables": 0.03,
        })
        self._crop_states = {}
        for crop_id, crop_def in CROP_DATA.items():
            self._crop_states[crop_id] = {
                "allocated_fraction": float(initial_alloc.get(crop_id, 0.0)),
                "subsidy_multiplier": 1.0,
                "market_demand_index": 1.0,
                "yield_t_per_ha": crop_def["base_yield_t_per_ha"],  # start at base yield
            }

        # Initialize economic model
        self._eco_model.initialize(CROP_DATA)
        self._eco_model.last_income_result = (
            POVERTY_LINE_INR_PER_YEAR * tc.get("farmer_income_ratio", 1.8),
            max(0.0, 1.0 - tc.get("farmer_income_ratio", 1.8) * 0.25),  # approximate initial poverty
        )

        # Initialize season manager
        self._season_mgr.reset(
            seed=seed,
            climate_shock=tc.get("special_conditions", {}).get("drought_active", False),
            climate_shock_factors=tc.get("rainfall_shock_by_season", {}),
        )

        # Build initial observation
        obs = self._build_observation(reward=None, done=False)
        self._prev_obs = obs

        logger.info(
            f"Episode {self._episode_id[:8]}... started | "
            f"Task: {task} | Max steps: {self._max_steps}"
        )
        return obs

    def step(
        self,
        action: AquaGuardAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> AquaGuardObservation:
        """
        Execute one policy step (one growing season ~4 months).

        Pipeline:
            1. Validate action (Pydantic does this automatically)
            2. Update irrigation methods (with transition lag)
            3. Realize rainfall for this season
            4. Update crop allocations and subsidies
            5. Advance groundwater per zone (extraction + recharge + lateral flow)
            6. Compute crop yields (water stress × soil × temperature)
            7. Update economic model (income + poverty)
            8. Compute reward (multi-objective weighted sum)
            9. Advance season counter
            10. Check done conditions
            11. Build and return observation

        Args:
            action: AquaGuardAction with all policy decisions.

        Returns:
            AquaGuardObservation with next state, reward, and done flag.
        """
        if self._episode_id is None:
            raise RuntimeError("Call reset() before step(). Environment not initialized.")

        self._step_count += 1
        tc = self._task_config

        # ── 1. Update irrigation methods (transition lag) ──────────────────
        for zone_id in ZONE_DATA:
            new_method = action.irrigation_methods.get(zone_id, self._irrigation_methods[zone_id])
            if new_method != self._irrigation_methods[zone_id]:
                # Starting a new method — set transition countdown
                self._irrigation_transition_countdown[zone_id] = (
                    IRRIGATION_EFFICIENCY[new_method]["transition_seasons"]
                )

            countdown = self._irrigation_transition_countdown[zone_id]
            if countdown > 0:
                # Still transitioning — old method still active
                self._irrigation_transition_countdown[zone_id] -= 1
            else:
                # Transition complete (or immediate for flood)
                self._irrigation_methods[zone_id] = new_method

        # ── 2. Realize rainfall ─────────────────────────────────────────────
        actual_rainfall = self._season_mgr.realize_rainfall()

        # Additional shock factor from task (climate_shock task)
        rainfall_shock = tc.get("rainfall_shock_factor", 1.0)
        if rainfall_shock != 1.0:
            actual_rainfall *= rainfall_shock

        # ── 3. Update crop states from action ──────────────────────────────
        for crop_id in CROP_DATA:
            new_alloc = action.crop_allocation.get(crop_id, 0.0)
            new_subsidy_adj = action.subsidy_adjustments.get(crop_id, 0.0)

            # Policy shift task: enforce max transition speed constraint
            if tc.get("max_rice_allocation_reduction_per_step") and crop_id == "rice":
                max_drop = tc["max_rice_allocation_reduction_per_step"]
                old_alloc = self._crop_states[crop_id]["allocated_fraction"]
                new_alloc = max(old_alloc - max_drop, new_alloc)
            if tc.get("max_wheat_allocation_reduction_per_step") and crop_id == "wheat":
                max_drop = tc["max_wheat_allocation_reduction_per_step"]
                old_alloc = self._crop_states[crop_id]["allocated_fraction"]
                new_alloc = max(old_alloc - max_drop, new_alloc)

            self._crop_states[crop_id]["allocated_fraction"] = new_alloc

            # Subsidy multiplier update: compound, bounded [0.2, 3.0]
            current_mult = self._crop_states[crop_id]["subsidy_multiplier"]
            new_mult = current_mult * (1.0 + new_subsidy_adj)
            self._crop_states[crop_id]["subsidy_multiplier"] = max(0.2, min(3.0, new_mult))

        # ── 4. Record pre-step aquifer average for reward delta ─────────────
        total_gw_depth_before = sum(z["gw_depth_m"] for z in self._zone_states.values()) / len(self._zone_states)

        # ── 5. Update groundwater per zone ──────────────────────────────────
        for zone_id, zone_state in self._zone_states.items():
            if zone_state["is_collapsed"]:
                continue

            zone_def = ZONE_DATA[zone_id]
            irr_method = self._irrigation_methods[zone_id]
            irr_eff = IRRIGATION_EFFICIENCY[irr_method]

            water_quota = action.water_quotas.get(zone_id, 900.0)
            extraction_limit = action.extraction_limits.get(zone_id, 30.0)

            # Total fraction of land allocated to crops
            total_alloc = sum(self._crop_states[c]["allocated_fraction"] for c in CROP_DATA)
            
            # Effective irrigation water applied to the planted fields
            effective_irrigation_mm = water_quota * irr_eff["water_use_fraction"]
            
            # Weighted groundwater irrigation fraction (by crop allocation)
            gwi_weighted = sum(
                CROP_DATA[c]["gwi_irrigation_fraction"] * self._crop_states[c]["allocated_fraction"]
                for c in CROP_DATA
            )
            if total_alloc > 0:
                gwi_weighted /= total_alloc
            else:
                gwi_weighted = 0.4  # default

            new_depth, water_extracted_m = self._gw_model.advance(
                current_depth_m=zone_state["gw_depth_m"],
                extraction_limit_m=extraction_limit,
                rainfall_mm=actual_rainfall,
                recharge_rate_mm_yr=zone_def["natural_recharge_mm_per_year"],
                storage_coefficient=zone_def["storage_coefficient"],
                irrigation_demand_mm=effective_irrigation_mm * min(1.0, total_alloc),
                gwi_fraction=gwi_weighted,
                rainfall_recharge_fraction=RAINFALL_RECHARGE_FRACTION,
            )

            zone_state["gw_depth_m"] = new_depth
            zone_state["water_used_mm"] = effective_irrigation_mm
            zone_state["actual_extracted_m"] = water_extracted_m

            # Check for collapse
            if new_depth > zone_def["collapse_threshold_m"]:
                if not zone_state["is_collapsed"]:
                    zone_state["is_collapsed"] = True
                    zone_state["arable_land_ha"] *= 0.30  # 70% land permanently lost
                    self._crisis_triggered = True
                    logger.warning(
                        f"AQUIFER COLLAPSE: {zone_id} at {new_depth:.1f}m "
                        f"(threshold: {zone_def['collapse_threshold_m']}m)"
                    )

            # Update soil health
            # Salinity accumulation depends on irrigation method
            salinity_increase = irr_eff["salinity_accumulation_per_season"]
            zone_state["soil_salinity"] = min(1.0, zone_state["soil_salinity"] + salinity_increase)

            # Fertility: slight baseline degradation + diversity improvement
            shannon = self._compute_shannon_diversity()
            fertility_gain = 0.005 * (shannon / 1.8)  # diversity improves fertility
            zone_state["soil_fertility"] = min(1.0, max(0.0,
                zone_state["soil_fertility"] - 0.003 + fertility_gain
            ))

        # Apply inter-zone lateral flow
        self._gw_model.apply_lateral_flow(self._zone_states, ZONE_DATA)

        # ── 6. Compute crop yields ──────────────────────────────────────────
        total_food_production_tons = 0.0
        total_land_ha = sum(z["arable_land_ha"] for z in self._zone_states.values())
        all_zones_avg_fertility = sum(
            z["soil_fertility"] for z in self._zone_states.values()
        ) / max(len(self._zone_states), 1)

        for crop_id, crop_state in self._crop_states.items():
            if crop_state["allocated_fraction"] < 0.001:
                crop_state["yield_t_per_ha"] = 0.0
                continue

            crop_def = CROP_DATA[crop_id]
            crop_land_ha = crop_state["allocated_fraction"] * total_land_ha

            # Average water available across zones
            avg_water_mm = sum(z["water_used_mm"] for z in self._zone_states.values()) / max(len(self._zone_states), 1)

            yield_t_per_ha = self._crop_model.compute_yield(
                base_yield=crop_def["base_yield_t_per_ha"],
                water_available_mm=avg_water_mm + actual_rainfall * 0.5,  # combine irrigation + rain
                water_requirement_mm=crop_def["water_requirement_mm_per_season"],
                optimal_water_mm=crop_def["optimal_water_mm"],
                wilting_point_mm=crop_def["wilting_point_mm"],
                soil_fertility=all_zones_avg_fertility,
                temperature_anomaly=self._season_mgr.current_temperature_anomaly,
                irrigation_method=self._irrigation_methods.get("zone_a", "flood"),
            )

            # Update market demand (mean-reverting)
            self._crop_states[crop_id]["market_demand_index"] = (
                self._eco_model.update_market_demand(crop_state["market_demand_index"])
            )

            crop_state["yield_t_per_ha"] = yield_t_per_ha
            total_food_production_tons += crop_land_ha * yield_t_per_ha

        # Update MSP prices annually
        if self._season_mgr.season_index == 0:  # start of year (kharif)
            self._eco_model.update_msp_prices(self._season_mgr.current_year)

        # ── 7. Economic model update ────────────────────────────────────────
        avg_income, poverty_fraction = self._eco_model.compute_farmer_income(
            crop_states=self._crop_states,
            zone_states=self._zone_states,
            zone_data=ZONE_DATA,
            crop_data=CROP_DATA,
        )

        # ── 8. Food security calculation ────────────────────────────────────
        total_farmers = sum(ZONE_DATA[z]["farmer_households"] for z in ZONE_DATA)
        total_population_served = total_farmers * HOUSEHOLD_SIZE
        food_req_tons = (
            total_population_served * FOOD_REQUIREMENT_KG_PER_PERSON_YEAR / 1000
            * tc.get("food_requirement_multiplier", 1.0)
        )
        food_security_ratio = total_food_production_tons / max(food_req_tons, 1.0)

        # ── 9. Update episode metrics ───────────────────────────────────────
        shannon_diversity = self._compute_shannon_diversity()
        if shannon_diversity > self._best_shannon_diversity:
            self._best_shannon_diversity = shannon_diversity

        current_avg_gw_depth = sum(z["gw_depth_m"] for z in self._zone_states.values()) / len(self._zone_states)
        if current_avg_gw_depth < total_gw_depth_before:
            self._aquifer_recovery_steps += 1

        if food_security_ratio < 1.0:
            self._food_security_failures += 1
            self._consecutive_food_failures += 1
        else:
            self._consecutive_food_failures = 0

        if poverty_fraction > 0.35:
            self._income_failures += 1

        # ── 10. Compute reward ──────────────────────────────────────────────
        reward = self._reward_calc.compute(
            prev_avg_gw_depth=total_gw_depth_before,
            new_avg_gw_depth=current_avg_gw_depth,
            zone_states=self._zone_states,
            zone_data=ZONE_DATA,
            food_security_ratio=food_security_ratio,
            poverty_fraction=poverty_fraction,
            shannon_diversity=shannon_diversity,
            any_zone_collapsed=any(z["is_collapsed"] for z in self._zone_states.values()),
            consecutive_food_failures=self._consecutive_food_failures,
            reward_weights=tc.get("reward_weights", DEFAULT_REWARD_WEIGHTS),
        )
        self._cumulative_reward += reward

        # ── 11. Advance season ──────────────────────────────────────────────
        self._season_mgr.advance()

        # ── 12. Check done conditions ───────────────────────────────────────
        done = False
        done_reason = ""

        if self._step_count >= self._max_steps:
            done = True
            done_reason = f"Episode completed after {self._step_count} seasons"
        elif current_avg_gw_depth > CATASTROPHIC_GW_DEPTH_M:
            done = True
            done_reason = "Catastrophic aquifer collapse — average depth exceeded 50m"
        elif self._consecutive_food_failures >= FAMINE_CONSECUTIVE_FAILURES:
            done = True
            done_reason = f"Famine triggered — {FAMINE_CONSECUTIVE_FAILURES} consecutive food security failures"

        if done:
            logger.info(
                f"Episode {self._episode_id[:8]}... ended. "
                f"Reason: {done_reason}. "
                f"Cumulative reward: {self._cumulative_reward:.2f}"
            )

        # ── 12.5 Run grader on action justification ─────────────────────────
        step_llm_score = None
        step_grader_feedback = None
        try:
            from server.grader import LLMGrader
            grader = LLMGrader()
            if hasattr(action, 'justification') and action.justification:
                grade_result = grader.score_justification(
                    justification=action.justification,
                    observation=self._prev_obs,
                    action=action,
                )
                step_llm_score = grade_result.overall_score  # already [0.0, 1.0]
                step_grader_feedback = grade_result.critique
                logger.debug(
                    f"LLM grader score: {step_llm_score:.3f} | "
                    f"Feedback: {step_grader_feedback[:80]}..."
                )
        except Exception as e:
            logger.debug(f"Grader skipped this step: {e}")

        # ── 13. Build and return observation ────────────────────────────────
        obs = self._build_observation(
            reward=reward,
            done=done,
            llm_score=step_llm_score,
            grader_feedback=step_grader_feedback,
        )
        self._prev_obs = obs
        return obs

    @property
    def state(self) -> AquaGuardState:
        """
        Return current episode state metadata.

        This property can be called at any time to get episode-level statistics.
        Useful for graders and external monitoring.
        """
        return AquaGuardState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_name=self._task_name,
            task_config=self._task_config,
            max_steps=self._max_steps,
            cumulative_reward=self._cumulative_reward,
            seasons_completed=self._step_count,
            groundwater_crisis_triggered=self._crisis_triggered,
            food_security_failures=self._food_security_failures,
            income_failures=self._income_failures,
            best_shannon_diversity=self._best_shannon_diversity,
            aquifer_recovery_steps=self._aquifer_recovery_steps,
        )

    # ─── Private Helpers ───────────────────────────────────────────────────────

    def _compute_shannon_diversity(self) -> float:
        """Compute Shannon entropy of current crop allocation."""
        allocations = [
            self._crop_states[c]["allocated_fraction"]
            for c in self._crop_states
            if self._crop_states[c]["allocated_fraction"] > 0.001
        ]
        if not allocations:
            return 0.0
        total = sum(allocations)
        if total <= 0:
            return 0.0
        return -sum((p / total) * math.log(p / total) for p in allocations)

    def _build_observation(
        self,
        reward: Optional[float],
        done: bool,
        llm_score: Optional[float] = None,
        grader_feedback: Optional[str] = None,
    ) -> AquaGuardObservation:
        """Construct the full AquaGuardObservation from current simulation state."""
        # Build ZoneObservation objects
        zones = {}
        for zone_id, zone_state in self._zone_states.items():
            zone_def = ZONE_DATA[zone_id]
            zones[zone_id] = ZoneObservation(
                zone_id=zone_id,
                groundwater_depth_m=zone_state["gw_depth_m"],
                groundwater_recharge_rate_mm_yr=zone_def["natural_recharge_mm_per_year"],
                soil_fertility=zone_state["soil_fertility"],
                soil_salinity=zone_state["soil_salinity"],
                arable_land_ha=zone_state["arable_land_ha"],
                active_irrigation_method=self._irrigation_methods[zone_id],
                water_used_mm=zone_state["water_used_mm"],
                is_in_danger_zone=zone_state["gw_depth_m"] > zone_def["critical_threshold_m"],
                is_collapsed=zone_state["is_collapsed"],
            )

        # Build CropObservation objects
        crops = {}
        total_land = sum(z["arable_land_ha"] for z in self._zone_states.values())
        avg_water = sum(z["water_used_mm"] for z in self._zone_states.values()) / max(len(self._zone_states), 1)

        for crop_id, crop_state in self._crop_states.items():
            crop_def = CROP_DATA[crop_id]
            water_stress = self._crop_model.compute_water_stress(
                water_available_mm=avg_water,
                water_requirement_mm=crop_def["water_requirement_mm_per_season"],
                wilting_point_mm=crop_def["wilting_point_mm"],
            )
            crops[crop_id] = CropObservation(
                crop_type=crop_id,
                allocated_fraction=crop_state["allocated_fraction"],
                water_requirement_mm=crop_def["water_requirement_mm_per_season"],
                actual_yield_t_per_ha=crop_state.get("yield_t_per_ha", crop_def["base_yield_t_per_ha"]),
                base_yield_t_per_ha=crop_def["base_yield_t_per_ha"],
                msp_price_inr_per_ton=self._eco_model.current_msp.get(
                    crop_id, crop_def["msp_inr_per_ton"]
                ),
                subsidy_multiplier=crop_state["subsidy_multiplier"],
                market_demand_index=crop_state["market_demand_index"],
                water_stress_factor=water_stress,
            )

        # Compute aggregate values
        total_farmers = sum(ZONE_DATA[z]["farmer_households"] for z in ZONE_DATA)
        avg_income, poverty_frac = self._eco_model.last_income_result

        total_prod = sum(
            self._crop_states[c].get("yield_t_per_ha", 0) *
            self._crop_states[c]["allocated_fraction"] * total_land
            for c in self._crop_states
        )
        pop_served = total_farmers * HOUSEHOLD_SIZE
        food_req = pop_served * FOOD_REQUIREMENT_KG_PER_PERSON_YEAR / 1000
        food_ratio = total_prod / max(food_req, 1.0)

        avg_gw = sum(z["gw_depth_m"] for z in self._zone_states.values()) / max(len(self._zone_states), 1)
        critical_thresh = min(ZONE_DATA[z]["critical_threshold_m"] for z in ZONE_DATA)
        shannon = self._compute_shannon_diversity()
        rainfall_forecast = self._season_mgr.forecast_rainfall()

        # Build natural language scenario description
        scenario_desc = build_scenario_description(
            season=self._season_mgr.current_season,
            year=self._season_mgr.current_year,
            zones=zones,
            crops=crops,
            food_ratio=food_ratio,
            poverty_frac=poverty_frac,
            shannon=shannon,
            rainfall_forecast=rainfall_forecast,
            task_name=self._task_name,
            step_number=self._step_count,
            reward=reward,
        )

        return AquaGuardObservation(
            # Episode context
            season=self._season_mgr.current_season,
            year=self._season_mgr.current_year,
            step_number=self._step_count,
            task_name=self._task_name,
            # Zone and crop data
            zones=zones,
            crops=crops,
            # Farmer welfare
            total_farmer_population=total_farmers,
            average_farmer_income_inr=avg_income,
            income_poverty_line_inr=POVERTY_LINE_INR_PER_YEAR,
            percent_farmers_below_poverty=poverty_frac * 100,
            # Food security
            national_food_requirement_tons=food_req,
            current_production_tons=total_prod,
            food_security_ratio=food_ratio,
            # Hydrology
            shared_aquifer_level_m=avg_gw,
            critical_aquifer_threshold_m=critical_thresh,
            aquifer_danger_zone=avg_gw > critical_thresh,
            # Climate
            rainfall_forecast_mm=rainfall_forecast,
            rainfall_probability_distribution=self._season_mgr.rainfall_distribution_description(),
            temperature_anomaly_c=self._season_mgr.current_temperature_anomaly,
            # Diversity
            shannon_diversity_index=shannon,
            # Grader scores
            llm_score=llm_score,
            grader_feedback=grader_feedback,
            # Reward/done
            reward=reward,
            done=done,
            # Natural language description
            scenario_description=scenario_desc,
        )