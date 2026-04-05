# src/server/simulation/groundwater.py
"""
Groundwater dynamics model for AquaGuard-RL.

Implements a simplified multi-zone confined aquifer model calibrated with
Central Ground Water Board (CGWB) data for North India.

Model features:
    - Seasonal water balance: extraction vs recharge vs rainfall
    - Inter-zone lateral flow (Darcy's Law approximation)
    - Aquifer collapse detection and permanent land degradation
    - Sustainable extraction enforcement

References:
    - CGWB Annual Report 2023: https://cgwb.gov.in/en/reports
    - Darcy's Law: Q = K × A × (dh/dl)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Fraction of depth differential transferred between adjacent zones per season
INTER_ZONE_CONDUCTIVITY = 0.15


class GroundwaterModel:
    """
    Simulates multi-zone aquifer dynamics for one growing season at a time.

    The model tracks:
    1. Natural annual recharge (CGWB district estimates)
    2. Monsoon rainfall recharge (12% of rainfall reaches aquifer — CGWB all-India)
    3. Groundwater extraction for irrigation (bounded by extraction limits)
    4. Inter-zone lateral flow driven by hydraulic head gradients
    5. Aquifer collapse detection (irreversible)

    All depth values are in meters (depth to water table, positive = deeper/more depleted).
    """

    def advance(
        self,
        current_depth_m: float,
        extraction_limit_m: float,
        rainfall_mm: float,
        recharge_rate_mm_yr: float,
        storage_coefficient: float,
        irrigation_demand_mm: float,
        gwi_fraction: float,
        rainfall_recharge_fraction: float,
    ) -> Tuple[float, float]:
        """
        Advance groundwater depth for one growing season (~4 months).

        The water balance equation:
            net_change = total_recharge - actual_extraction
            depth_change = -net_change / storage_coefficient / 1000

        Args:
            current_depth_m: Current depth to water table in meters.
            extraction_limit_m: Agent-imposed extraction limit in meters/season.
            rainfall_mm: Actual realized rainfall for this season in mm.
            recharge_rate_mm_yr: Natural annual recharge rate in mm/year.
            storage_coefficient: Aquifer storage coefficient (dimensionless).
            irrigation_demand_mm: Irrigation water demand in mm.
            gwi_fraction: Fraction of irrigation met by groundwater (0–1).
            rainfall_recharge_fraction: Fraction of rainfall recharging aquifer (0–1).

        Returns:
            Tuple of (new_depth_m, water_extracted_m) where water_extracted_m
            is the volume extracted expressed as equivalent aquifer depth change.
        """
        # Seasonal natural recharge (1/3 of annual for a ~4-month season)
        seasonal_natural_recharge_mm = recharge_rate_mm_yr * (1.0 / 3.0)

        # Rainfall recharge (12% of rainfall reaches the aquifer, CGWB estimate)
        rainfall_recharge_mm = rainfall_mm * rainfall_recharge_fraction

        # Total recharge this season
        total_recharge_mm = seasonal_natural_recharge_mm + rainfall_recharge_mm

        # Groundwater extraction from irrigation
        gw_extraction_demand_mm = irrigation_demand_mm * gwi_fraction

        # Enforce extraction limit (convert m/season → mm equivalent for water balance)
        # extraction_limit_m is max depth drawdown allowed per season
        max_extraction_mm = extraction_limit_m * 1000.0 * storage_coefficient
        actual_extraction_mm = min(gw_extraction_demand_mm, max_extraction_mm)

        # Net change in water storage (positive = more water stored = table rising)
        net_change_mm = total_recharge_mm - actual_extraction_mm

        # Convert storage change to water table depth change
        # ΔH = ΔStorage_mm / (StorageCoefficient × 1000)
        # Positive net_change → water level rises → depth decreases (negative depth change)
        depth_change_m = -(net_change_mm / 1000.0) / storage_coefficient

        new_depth_m = max(0.1, current_depth_m + depth_change_m)  # floor at 0.1m (artesian)
        water_extracted_m = actual_extraction_mm / 1000.0

        logger.debug(
            f"GW balance: depth {current_depth_m:.2f}→{new_depth_m:.2f}m | "
            f"recharge={total_recharge_mm:.1f}mm (natural={seasonal_natural_recharge_mm:.1f}, "
            f"rain={rainfall_recharge_mm:.1f}) | extraction={actual_extraction_mm:.1f}mm | "
            f"net={net_change_mm:.1f}mm"
        )
        return new_depth_m, water_extracted_m

    def apply_lateral_flow(
        self,
        zone_states: Dict[str, Dict],
        zone_data: Dict[str, Dict],
    ) -> None:
        """
        Apply inter-zone lateral groundwater flow based on Darcy's Law approximation.

        Water flows from zones with shallower tables (less depleted) to zones with
        deeper tables (more depleted), driven by hydraulic head gradients.

        This models the hydrological connectivity of shared aquifer systems —
        aggressive extraction in one zone affects neighboring zones.

        Modifies zone_states in-place.

        Args:
            zone_states: Current zone simulation states (modified in-place).
            zone_data: Static zone parameters (used for storage coefficients).
        """
        zone_ids = [zid for zid, zs in zone_states.items() if not zs.get("is_collapsed", False)]

        if len(zone_ids) < 2:
            return

        # Snapshot current depths before applying flows
        depths_before = {zid: zone_states[zid]["gw_depth_m"] for zid in zone_ids}

        for i, zone_a_id in enumerate(zone_ids):
            for zone_b_id in zone_ids[i + 1:]:
                depth_a = depths_before[zone_a_id]
                depth_b = depths_before[zone_b_id]
                depth_diff = depth_a - depth_b

                if abs(depth_diff) < 0.5:
                    continue  # No meaningful hydraulic gradient

                # Darcy-approximated lateral flow
                # Water flows FROM deeper (more depleted) zone TO shallower (less depleted)
                # depth_diff = depth_a - depth_b; positive means A is deeper
                # Flow magnitude proportional to depth differential
                flow_magnitude = abs(depth_diff) * INTER_ZONE_CONDUCTIVITY

                sc_a = zone_data[zone_a_id]["storage_coefficient"]
                sc_b = zone_data[zone_b_id]["storage_coefficient"]

                # Depth change: zones converge toward each other
                change_a = flow_magnitude * (sc_b / (sc_a + sc_b))
                change_b = flow_magnitude * (sc_a / (sc_a + sc_b))

                if depth_diff > 0:
                    # Zone A is deeper → gains water (depth decreases), Zone B loses water (depth increases)
                    zone_states[zone_a_id]["gw_depth_m"] = max(0.1, zone_states[zone_a_id]["gw_depth_m"] - change_a)
                    zone_states[zone_b_id]["gw_depth_m"] = max(0.1, zone_states[zone_b_id]["gw_depth_m"] + change_b)
                else:
                    # Zone B is deeper → gains water (depth decreases), Zone A loses water (depth increases)
                    zone_states[zone_b_id]["gw_depth_m"] = max(0.1, zone_states[zone_b_id]["gw_depth_m"] - change_b)
                    zone_states[zone_a_id]["gw_depth_m"] = max(0.1, zone_states[zone_a_id]["gw_depth_m"] + change_a)

        logger.debug(
            f"Lateral flow applied. Depths: "
            + " | ".join(f"{zid}={zone_states[zid]['gw_depth_m']:.2f}m" for zid in zone_ids)
        )

    def compute_sustainable_extraction(
        self,
        recharge_rate_mm_yr: float,
        storage_coefficient: float,
        safety_factor: float = 1.5,
    ) -> float:
        """
        Compute the maximum sustainable seasonal extraction in meters.

        Allows up to safety_factor × natural recharge, enabling some managed overdraft
        in drought years while preventing long-term depletion.

        Args:
            recharge_rate_mm_yr: Natural annual recharge rate in mm/year.
            storage_coefficient: Aquifer storage coefficient.
            safety_factor: Multiplier on natural recharge (default 1.5 = 50% overdraft).

        Returns:
            Maximum sustainable extraction in meters/season.
        """
        seasonal_recharge_mm = recharge_rate_mm_yr / 3.0
        max_extraction_mm = seasonal_recharge_mm * safety_factor
        return (max_extraction_mm / 1000.0) / storage_coefficient