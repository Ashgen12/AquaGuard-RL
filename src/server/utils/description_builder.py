# src/server/utils/description_builder.py
"""
Natural language scenario description builder for AquaGuard-RL.

Generates human-readable (and LLM-readable) summaries of the current
agricultural system state. These descriptions are included in every
observation as `scenario_description` to enable LLM-based agents to
understand the situation without parsing raw numerical data.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

# Shannon entropy maximum for 6 crops (log(6) ≈ 1.7918)
H_MAX = math.log(6)


def build_scenario_description(
    season: str,
    year: int,
    zones: Dict,
    crops: Dict,
    food_ratio: float,
    poverty_frac: float,
    shannon: float,
    rainfall_forecast: float,
    task_name: str,
    step_number: int = 0,
    reward: Optional[float] = None,
) -> str:
    """
    Build a comprehensive natural language description of the current simulation state.

    The description is structured for LLM consumption:
    1. Context header (season, year, task)
    2. Groundwater status (per zone, danger indicators)
    3. Food security status
    4. Farmer welfare status
    5. Crop portfolio summary
    6. Climate forecast
    7. Action guidance

    Args:
        season: Current season name (kharif/rabi/zaid).
        year: Current year number.
        zones: Dict of ZoneObservation objects.
        crops: Dict of CropObservation objects.
        food_ratio: Current food security ratio.
        poverty_frac: Fraction of farmers below poverty line [0,1].
        shannon: Shannon diversity index of current allocation.
        rainfall_forecast: Forecast rainfall for this season in mm.
        task_name: Active task name.
        step_number: Current step number.
        reward: Previous step's reward (None at reset).

    Returns:
        Multi-line natural language description (200–500 words).
    """
    lines = []

    # ─── Header ───────────────────────────────────────────────────────────────
    season_label = {"kharif": "Kharif (Monsoon, June-Oct)", "rabi": "Rabi (Winter, Oct-Mar)",
                    "zaid": "Zaid (Summer, Mar-Jun)"}.get(season, season.title())
    lines.append(f"=== AquaGuard-RL: {_task_display(task_name)} ===")
    lines.append(f"Season: {season_label} | Year {year} | Step {step_number}")
    if reward is not None:
        lines.append(f"Previous step reward: {reward:+.2f}")
    lines.append("")

    # ─── Groundwater Status ───────────────────────────────────────────────────
    lines.append("GROUNDWATER STATUS:")
    avg_depth = sum(z.groundwater_depth_m for z in zones.values()) / len(zones)
    any_danger = any(z.is_in_danger_zone for z in zones.values())
    any_collapsed = any(z.is_collapsed for z in zones.values())

    for zone_id, zone_obs in sorted(zones.items()):
        depth = zone_obs.groundwater_depth_m
        critical = zone_obs.groundwater_depth_m  # use zone's actual depth for relative assessment

        if zone_obs.is_collapsed:
            status = "COLLAPSED (irreversible)"
        elif zone_obs.is_in_danger_zone:
            status = f"DANGER ZONE ({depth:.1f}m --near critical threshold)"
        elif depth > 30:
            status = f"Warning ({depth:.1f}m --approaching stress)"
        elif depth > 20:
            status = f"Moderate ({depth:.1f}m)"
        else:
            status = f"Healthy ({depth:.1f}m)"

        lines.append(
            f"  {zone_id.upper()}: {status} | "
            f"Soil fertility: {zone_obs.soil_fertility:.0%} | "
            f"Salinity: {zone_obs.soil_salinity:.0%} | "
            f"Irrigation: {zone_obs.active_irrigation_method}"
        )

    lines.append(f"  Average aquifer depth: {avg_depth:.1f}m")

    if any_collapsed:
        lines.append("  [!] AQUIFER COLLAPSE DETECTED -- Permanent land degradation active")
    elif any_danger:
        lines.append("  [!] WARNING: One or more zones in danger zone -- reduce extraction immediately")
    lines.append("")

    # ─── Food Security ────────────────────────────────────────────────────────
    lines.append("FOOD SECURITY:")
    if food_ratio >= 1.2:
        food_status = f"Excellent surplus ({food_ratio:.2f}x requirement)"
    elif food_ratio >= 1.0:
        food_status = f"Meeting requirement ({food_ratio:.2f}x)"
    elif food_ratio >= 0.85:
        food_status = f"Mild shortfall ({food_ratio:.2f}x --{(1-food_ratio)*100:.0f}% deficit)"
    elif food_ratio >= 0.7:
        food_status = f"SERIOUS SHORTFALL ({food_ratio:.2f}x --emergency)"
    else:
        food_status = f"FAMINE RISK ({food_ratio:.2f}x --critical)"
    lines.append(f"  Food production ratio: {food_status}")
    lines.append("")

    # ─── Farmer Welfare ───────────────────────────────────────────────────────
    lines.append("FARMER WELFARE:")
    poverty_pct = poverty_frac * 100
    if poverty_pct < 20:
        welfare_status = f"Good ({poverty_pct:.0f}% below poverty)"
    elif poverty_pct < 35:
        welfare_status = f"Moderate ({poverty_pct:.0f}% below poverty --attention needed)"
    elif poverty_pct < 55:
        welfare_status = f"Poor ({poverty_pct:.0f}% in poverty --intervention required)"
    else:
        welfare_status = f"CRISIS ({poverty_pct:.0f}% in poverty --social instability risk)"
    lines.append(f"  Farmer welfare: {welfare_status}")
    lines.append("")

    # ─── Current Crop Portfolio ───────────────────────────────────────────────
    lines.append("CURRENT CROP PORTFOLIO:")
    sorted_crops = sorted(crops.items(), key=lambda x: x[1].allocated_fraction, reverse=True)
    for crop_id, crop_obs in sorted_crops:
        if crop_obs.allocated_fraction < 0.01:
            continue
        alloc_pct = crop_obs.allocated_fraction * 100
        yield_str = f"{crop_obs.actual_yield_t_per_ha:.1f} t/ha"
        msp_str = f"INR {crop_obs.msp_price_inr_per_ton:,.0f}/t"
        lines.append(
            f"  {crop_id.title():<12}: {alloc_pct:.0f}% land | "
            f"Yield: {yield_str} | MSP: {msp_str} | "
            f"Water req: {crop_obs.water_requirement_mm:.0f}mm"
        )

    diversity_pct = shannon / H_MAX * 100
    diversity_label = _diversity_label(shannon)
    lines.append(f"  Shannon diversity: {shannon:.3f} ({diversity_pct:.0f}% of maximum) --{diversity_label}")
    lines.append("")

    # ─── Climate Forecast ─────────────────────────────────────────────────────
    lines.append("CLIMATE FORECAST:")
    season_normal = {"kharif": 800, "rabi": 120, "zaid": 30}.get(season, 300)
    dev_pct = (rainfall_forecast - season_normal) / season_normal * 100
    if dev_pct > 20:
        rain_desc = f"above normal (+{dev_pct:.0f}%)"
    elif dev_pct > -10:
        rain_desc = "near normal"
    elif dev_pct > -30:
        rain_desc = f"below normal ({dev_pct:.0f}%)"
    else:
        rain_desc = f"SIGNIFICANTLY BELOW NORMAL ({dev_pct:.0f}%) --drought conditions"
    lines.append(f"  {season.title()} rainfall forecast: {rainfall_forecast:.0f}mm ({rain_desc})")
    lines.append(f"  Historical season average: {season_normal:.0f}mm")
    lines.append("")

    # ─── Strategic Context ────────────────────────────────────────────────────
    lines.append(_build_strategic_context(task_name, avg_depth, food_ratio, poverty_frac, shannon, season))

    return "\n".join(lines)


def _task_display(task_name: str) -> str:
    """Convert task name to display format."""
    return {
        "baseline": "Baseline Stability Challenge",
        "crisis": "Aquifer Crisis Recovery",
        "policy_shift": "Green Revolution Policy Shift",
        "climate_shock": "El Niño Drought Management",
        "multi_district": "Multi-District Equity Coordination",
    }.get(task_name, task_name.replace("_", " ").title())


def _diversity_label(shannon: float) -> str:
    """Convert Shannon index to qualitative label."""
    if shannon >= 1.5:
        return "Excellent diversity"
    elif shannon >= 1.0:
        return "Good diversity"
    elif shannon >= 0.6:
        return "Moderate diversity"
    elif shannon >= 0.3:
        return "Low diversity --rice/wheat dominant"
    else:
        return "Near monoculture --very high water/pest risk"


def _build_strategic_context(
    task_name: str,
    avg_depth: float,
    food_ratio: float,
    poverty_frac: float,
    shannon: float,
    season: str,
) -> str:
    """
    Build task-specific strategic guidance paragraph.

    This helps LLM agents understand what they should optimize for.
    """
    context_lines = ["STRATEGIC CONTEXT:"]

    # Common observations
    warnings = []
    if avg_depth > 35:
        warnings.append(f"Aquifer critically low ({avg_depth:.1f}m avg) --reduce rice/wheat, enable drip irrigation")
    elif avg_depth > 28:
        warnings.append(f"Aquifer under stress ({avg_depth:.1f}m avg) --monitor extraction closely")

    if food_ratio < 0.9:
        warnings.append(f"Food shortfall ({food_ratio:.2f}x) --maintain grain production, avoid crop cuts")
    if poverty_frac > 0.4:
        warnings.append(f"High poverty ({poverty_frac:.0%}) --subsidy increases for water-efficient crops recommended")

    # Task-specific guidance
    task_guidance = {
        "baseline": (
            "Goal: Maintain all zones below critical thresholds for 10 seasons. "
            "Gradually shift from water-intensive rice/wheat toward millets and pulses. "
            "Target: cumulative reward > 40, GW depth <= 38m throughout."
        ),
        "crisis": (
            "EMERGENCY: Zone C near critical. Immediate aquifer intervention required. "
            "Slash rice allocation, deploy drip irrigation, reduce extraction limits. "
            "Food security may need to be temporarily compromised to prevent collapse."
        ),
        "policy_shift": (
            "Goal: Achieve Shannon diversity >= 1.2 (from ~0.85 initial) over 8 seasons. "
            "Transition speed is limited --max 8pp reduction per crop per season. "
            "Use subsidy adjustments to make water-efficient crops economically attractive."
        ),
        "climate_shock": (
            "DROUGHT ACTIVE: Only 40% of normal rainfall expected. "
            "Critical: do NOT over-extract groundwater to compensate --this causes collapse. "
            "Prioritize drought-tolerant crops (millet, pulses). Reduce rice allocation urgently."
        ),
        "multi_district": (
            "Goal: Balance all 3 zones with equity constraint (poorest zone income >= 70% of richest). "
            "Zone C needs extra protection --vulnerable dryland with stressed aquifer. "
            "Use targeted subsidies for Zone C. 15 seasons available for gradual transition."
        ),
    }

    guidance = task_guidance.get(task_name, "Optimize all four objectives simultaneously.")
    context_lines.append(f"  Task: {guidance}")

    if warnings:
        context_lines.append("  Alerts:")
        for w in warnings:
            context_lines.append(f"    - {w}")

    context_lines.append(
        "  Action format: Specify crop_allocation (sum<=1.0), water_quotas, "
        "irrigation_methods, extraction_limits, subsidy_adjustments, and justification."
    )

    return "\n".join(context_lines)