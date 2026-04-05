# src/models.py
"""
Data models for AquaGuard-RL environment.

All models use Pydantic v2 with full validation.
Designed for OpenEnv compatibility — inherits from openenv_core base classes when available,
falls back to standalone Pydantic BaseModel for local development.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Any

try:
    from pydantic import Field, field_validator, model_validator, computed_field, BaseModel, ConfigDict
except ImportError:
    raise ImportError("pydantic>=2.0 required: pip install pydantic>=2.0")

# ─── OpenEnv compatibility layer ───────────────────────────────────────────────
# When openenv-core is installed, we inherit from its base classes.
# When running standalone (local dev), we use plain Pydantic BaseModel.

try:
    from openenv_core import Action as _OEAction, Observation as _OEObservation, State as _OEState  # type: ignore
    _Action = _OEAction
    _Observation = _OEObservation
    _State = _OEState
    _OPENENV_AVAILABLE = True
except ImportError:
    _OPENENV_AVAILABLE = False

    class _BaseOpenEnvModel(BaseModel):
        """Standalone base with reward/done fields matching OpenEnv convention."""
        model_config = ConfigDict(extra="allow", populate_by_name=True)
        reward: Optional[float] = Field(default=None, description="Step reward (None on reset)")
        done: bool = Field(default=False, description="Whether the episode has ended")

    class _Action(BaseModel):
        """Standalone Action base."""
        model_config = ConfigDict(extra="allow")

    class _Observation(_BaseOpenEnvModel):
        """Standalone Observation base with reward + done."""
        pass

    class _State(BaseModel):
        """Standalone State base."""
        model_config = ConfigDict(extra="allow")
        episode_id: Optional[str] = Field(default=None)
        step_count: int = Field(default=0)


# ─── Sub-observation models ────────────────────────────────────────────────────

class ZoneObservation(_Action):
    """
    Groundwater and soil status for a single district zone.

    Attributes:
        zone_id: Identifier (zone_a, zone_b, zone_c).
        groundwater_depth_m: Depth to water table in meters (higher = more depleted).
        groundwater_recharge_rate_mm_yr: Natural annual aquifer recharge in mm/yr.
        soil_fertility: Soil fertility index [0, 1] where 1 = pristine.
        soil_salinity: Soil salinity index [0, 1] where 1 = severely saline.
        arable_land_ha: Usable arable land in hectares.
        active_irrigation_method: Currently deployed method (flood/sprinkler/drip).
        water_used_mm: Irrigation water applied this season in mm.
        is_in_danger_zone: True if depth exceeds zone's critical threshold.
        is_collapsed: True if aquifer has irreversibly collapsed (depth > collapse threshold).
    """

    model_config = ConfigDict(extra="allow")

    zone_id: str = Field(..., description="Zone identifier: zone_a, zone_b, or zone_c")
    groundwater_depth_m: float = Field(..., ge=0.0, le=200.0,
                                       description="Depth to water table in meters")
    groundwater_recharge_rate_mm_yr: float = Field(..., ge=0.0,
                                                    description="Annual recharge rate mm/yr")
    soil_fertility: float = Field(..., ge=0.0, le=1.0,
                                  description="Soil fertility index (0=exhausted, 1=pristine)")
    soil_salinity: float = Field(..., ge=0.0, le=1.0,
                                 description="Soil salinity index (0=none, 1=severely saline)")
    arable_land_ha: float = Field(..., gt=0.0, description="Usable arable land in hectares")
    active_irrigation_method: str = Field(..., description="Deployed method: flood/sprinkler/drip")
    water_used_mm: float = Field(..., ge=0.0, description="Irrigation water used this season in mm")
    is_in_danger_zone: bool = Field(..., description="True if depth > critical threshold")
    is_collapsed: bool = Field(default=False, description="True if aquifer irreversibly collapsed")


class CropObservation(_Action):
    """
    Agricultural production data for one crop type in the current season.

    Attributes:
        crop_type: Crop identifier (rice, wheat, millet, pulses, oilseeds, vegetables).
        allocated_fraction: Land fraction [0,1] allocated to this crop.
        water_requirement_mm: Full-season water requirement in mm.
        actual_yield_t_per_ha: Current yield given water/soil/temperature conditions.
        base_yield_t_per_ha: Potential yield under ideal conditions.
        msp_price_inr_per_ton: Current Minimum Support Price in INR/ton.
        subsidy_multiplier: Income multiplier applied to MSP (1.0 = baseline).
        market_demand_index: Relative market demand (1.0 = balanced).
        water_stress_factor: Water availability ratio [0,1] affecting yield.
    """

    model_config = ConfigDict(extra="allow")

    crop_type: str = Field(..., description="Crop identifier")
    allocated_fraction: float = Field(..., ge=0.0, le=1.0,
                                      description="Land fraction allocated to this crop")
    water_requirement_mm: float = Field(..., ge=0.0,
                                        description="Full season water requirement in mm")
    actual_yield_t_per_ha: float = Field(..., ge=0.0,
                                         description="Actual yield given current conditions")
    base_yield_t_per_ha: float = Field(..., ge=0.0,
                                       description="Baseline potential yield")
    msp_price_inr_per_ton: float = Field(..., ge=0.0,
                                         description="Current Minimum Support Price in INR/ton")
    subsidy_multiplier: float = Field(..., ge=0.0,
                                      description="Subsidy multiplier on MSP (1.0 = baseline)")
    market_demand_index: float = Field(..., ge=0.0,
                                       description="Relative market demand (1.0 = balanced)")
    water_stress_factor: float = Field(..., ge=0.0, le=1.0,
                                       description="Water stress factor [0,1] affecting yield")


# ─── Main Action Model ─────────────────────────────────────────────────────────

class AquaGuardAction(_Action):
    """
    Policy decisions made by the District Agricultural Commissioner for the upcoming season.

    The agent specifies crop area allocations, water quotas, irrigation methods,
    extraction limits, subsidy adjustments, and a natural-language justification
    for the policy choices (evaluated by the LLM grader).

    All allocation values are fractions (0.0–1.0) summing to ≤ 1.0.
    Water quotas are in mm/season. Extraction limits in meters/season.
    Subsidy adjustments are relative (−1.0 to +1.0).
    """

    model_config = ConfigDict(extra="allow")

    crop_allocation: Dict[str, float] = Field(
        default_factory=lambda: {
            "rice": 0.30, "wheat": 0.30, "millet": 0.15,
            "pulses": 0.15, "oilseeds": 0.07, "vegetables": 0.03,
        },
        description=(
            "Fraction of total arable land allocated to each crop type. "
            "Must sum to ≤ 1.0. Valid keys: rice, wheat, millet, pulses, oilseeds, vegetables."
        ),
    )

    water_quotas: Dict[str, float] = Field(
        default_factory=lambda: {"zone_a": 900.0, "zone_b": 900.0, "zone_c": 900.0},
        description="Maximum irrigation water per zone in mm/season (range: 0–2000 mm).",
    )

    irrigation_methods: Dict[str, str] = Field(
        default_factory=lambda: {"zone_a": "flood", "zone_b": "flood", "zone_c": "flood"},
        description="Irrigation method per zone: 'flood', 'sprinkler', or 'drip'.",
    )

    extraction_limits: Dict[str, float] = Field(
        default_factory=lambda: {"zone_a": 30.0, "zone_b": 30.0, "zone_c": 30.0},
        description="Maximum groundwater extraction per zone in meters/season (range: 0–60 m).",
    )

    subsidy_adjustments: Dict[str, float] = Field(
        default_factory=lambda: {
            "rice": 0.0, "wheat": 0.0, "millet": 0.0,
            "pulses": 0.0, "oilseeds": 0.0, "vegetables": 0.0,
        },
        description=(
            "Relative MSP subsidy adjustment per crop (range: −1.0 to +1.0). "
            "Negative values reduce subsidy; positive values increase it."
        ),
    )

    justification: str = Field(
        default="",
        max_length=2000,
        description=(
            "Agent's natural-language reasoning for these policy decisions. "
            "Evaluated by LLM grader for causal logic, domain knowledge, and trade-off awareness."
        ),
    )

    @field_validator("crop_allocation")
    @classmethod
    def validate_allocation(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate crop allocations: valid keys, [0,1] fractions, sum ≤ 1.0."""
        valid = {"rice", "wheat", "millet", "pulses", "oilseeds", "vegetables"}
        for crop, frac in v.items():
            if crop not in valid:
                raise ValueError(f"Unknown crop '{crop}'. Valid crops: {sorted(valid)}")
            if not (0.0 <= frac <= 1.0):
                raise ValueError(
                    f"Crop allocation for '{crop}' must be in [0.0, 1.0], got {frac:.3f}"
                )
        total = sum(v.values())
        if total > 1.001:
            raise ValueError(
                f"Crop allocations sum to {total:.3f} which exceeds 1.0 (max allowed). "
                "Reduce allocations so they sum to ≤ 1.0."
            )
        return v

    @field_validator("water_quotas")
    @classmethod
    def validate_quotas(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate water quotas are non-negative and ≤ 2000 mm/season."""
        for zone, q in v.items():
            if not (0.0 <= q <= 2000.0):
                raise ValueError(
                    f"Water quota for '{zone}' must be in [0, 2000] mm/season, got {q}"
                )
        return v

    @field_validator("irrigation_methods")
    @classmethod
    def validate_irrigation(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate irrigation methods are flood, sprinkler, or drip."""
        valid = {"flood", "sprinkler", "drip"}
        for zone, m in v.items():
            if m not in valid:
                raise ValueError(f"Invalid irrigation method '{m}' for '{zone}'. Valid: {valid}")
        return v

    @field_validator("extraction_limits")
    @classmethod
    def validate_extraction(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate extraction limits are in [0, 60] m/season."""
        for zone, limit in v.items():
            if not (0.0 <= limit <= 60.0):
                raise ValueError(
                    f"Extraction limit for '{zone}' must be in [0, 60] m/season, got {limit}"
                )
        return v

    @field_validator("subsidy_adjustments")
    @classmethod
    def validate_subsidies(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate subsidy adjustments are in [-1.0, 1.0]."""
        for crop, adj in v.items():
            if not (-1.0 <= adj <= 1.0):
                raise ValueError(
                    f"Subsidy adjustment for '{crop}' must be in [-1.0, 1.0], got {adj}"
                )
        return v


# ─── Main Observation Model ────────────────────────────────────────────────────

class AquaGuardObservation(_Observation):
    """
    Full observation returned after each step() or reset() call.

    Describes the current state of the multi-district agricultural system for
    the upcoming season planning cycle. Includes groundwater levels, crop data,
    farmer welfare, food security, and a natural-language scenario description.

    The `reward` and `done` fields are None after reset() and populated after step().
    The `scenario_description` provides an LLM-readable summary of the current state.
    """

    model_config = ConfigDict(extra="allow")

    # Episode context
    season: str = Field(..., description="Growing season: 'kharif', 'rabi', or 'zaid'")
    year: int = Field(..., ge=1, description="Simulation year (starts at 1)")
    step_number: int = Field(..., ge=0, description="Step number within episode (0 = after reset)")
    task_name: str = Field(..., description="Active task name")

    # Zone states (3 zones sharing an aquifer)
    zones: Dict[str, ZoneObservation] = Field(
        ..., description="Per-zone groundwater and soil status (keys: zone_a, zone_b, zone_c)"
    )

    # Crop states (6 crop types)
    crops: Dict[str, CropObservation] = Field(
        ..., description="Per-crop agricultural production data (6 crops)"
    )

    # Farmer welfare aggregates
    total_farmer_population: int = Field(
        ..., gt=0, description="Total farming households across all zones"
    )
    average_farmer_income_inr: float = Field(
        ..., description="Average annual farmer household income in INR"
    )
    income_poverty_line_inr: float = Field(
        ..., description="Rural poverty line threshold in INR/year"
    )
    percent_farmers_below_poverty: float = Field(
        ..., ge=0.0, le=100.0,
        description="Percentage of farming households below poverty line"
    )

    # Food security
    national_food_requirement_tons: float = Field(
        ..., gt=0.0, description="Minimum food grain requirement for served population in tons"
    )
    current_production_tons: float = Field(
        ..., ge=0.0, description="Total food grain production this season in tons"
    )
    food_security_ratio: float = Field(
        ..., ge=0.0,
        description="Production / requirement ratio (1.0 = exactly met, >1.0 = surplus)"
    )

    # Hydrology
    shared_aquifer_level_m: float = Field(
        ..., ge=0.0,
        description="Shared aquifer depth in meters (higher value = more depleted)"
    )
    critical_aquifer_threshold_m: float = Field(
        ..., description="Depth beyond which extraction causes permanent damage (meters)"
    )
    aquifer_danger_zone: bool = Field(
        ..., description="True if shared aquifer is below critical threshold"
    )

    # Climate
    rainfall_forecast_mm: float = Field(
        ..., ge=0.0, description="Forecasted rainfall for the upcoming season in mm"
    )
    rainfall_probability_distribution: str = Field(
        ..., description="Textual description of rainfall uncertainty"
    )
    temperature_anomaly_c: float = Field(
        ..., description="Temperature deviation from historical average in °C"
    )

    # Crop diversity
    shannon_diversity_index: float = Field(
        ..., ge=0.0,
        description="Shannon entropy of crop allocation (0 = monoculture, log(6)≈1.79 = equal)"
    )

    # Grader results (populated after evaluation, None initially)
    programmatic_score: Optional[float] = Field(
        default=None, description="Programmatic grader score [0.0, 1.0]"
    )
    llm_score: Optional[float] = Field(
        default=None, description="LLM grader overall score [0.0, 1.0]"
    )
    grader_feedback: Optional[str] = Field(
        default=None, description="LLM grader textual critique of the agent justification"
    )

    # Natural language summary for LLM-based agents
    scenario_description: str = Field(
        ...,
        description=(
            "Natural language description of the current situation for LLM-based agents. "
            "Includes season, groundwater status, food security, and key challenges."
        ),
    )

    @property
    def composite_grader_score(self) -> Optional[float]:
        """
        Combined grader score: 0.60 × programmatic + 0.40 × LLM.
        Both scores are already in [0.0, 1.0]. Returns None if no grader has run yet.
        """
        if self.programmatic_score is None and self.llm_score is None:
            return None
        p = self.programmatic_score or 0.0
        lm = self.llm_score or 0.0  # already normalized to [0.0, 1.0]
        if self.programmatic_score is not None and self.llm_score is not None:
            return 0.60 * p + 0.40 * lm
        if self.programmatic_score is not None:
            return p
        return lm


# ─── State Model ──────────────────────────────────────────────────────────────

class AquaGuardState(_State):
    """
    Episode-level metadata and cumulative statistics.

    Returned by the `state` property at any point during the episode.
    Contains aggregate metrics for evaluation and progress tracking.
    """

    model_config = ConfigDict(extra="allow")

    task_name: str = Field(default="baseline", description="Active task configuration name")
    task_config: Dict[str, Any] = Field(
        default_factory=dict, description="Task-specific parameter dictionary"
    )
    max_steps: int = Field(default=10, description="Maximum steps allowed in this episode")
    cumulative_reward: float = Field(
        default=0.0, description="Sum of all rewards received so far"
    )
    seasons_completed: int = Field(default=0, description="Number of seasons simulated")
    groundwater_crisis_triggered: bool = Field(
        default=False,
        description="True if any zone exceeded collapse threshold during episode"
    )
    food_security_failures: int = Field(
        default=0,
        description="Number of steps where food_security_ratio < 1.0"
    )
    income_failures: int = Field(
        default=0,
        description="Number of steps where poverty_fraction exceeded 35%"
    )
    best_shannon_diversity: float = Field(
        default=0.0,
        description="Highest Shannon diversity index achieved during episode"
    )
    aquifer_recovery_steps: int = Field(
        default=0,
        description="Number of steps where average aquifer depth improved (shallower)"
    )
    # episode_id and step_count are inherited from State base class