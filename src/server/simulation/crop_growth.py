# src/server/simulation/crop_growth.py
"""
Crop growth model for AquaGuard-RL.

Implements yield response to water, soil fertility, and temperature
based on FAO Penman-Monteith-inspired water stress functions.

References:
    - FAO Crop Yield Response to Water (Doorenbos & Kassam):
      https://www.fao.org/3/i2800e/i2800e.pdf
    - FAO AQUASTAT crop water requirements:
      https://www.fao.org/aquastat/en/
"""

from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Temperature above which yield penalty applies (°C anomaly threshold)
TEMP_PENALTY_THRESHOLD_C = 2.0
# Yield loss per °C above threshold (Lobell et al., 6% per °C)
YIELD_LOSS_PER_DEGREE = 0.06


class CropGrowthModel:
    """
    Simulates crop yield as a function of water availability, soil health, and climate.

    The yield model:
        actual_yield = base_yield
                     × water_stress_factor
                     × soil_fertility_factor
                     × temperature_factor
                     × irrigation_efficiency_factor

    All factors are in [0, 1] (multiplicative penalties).
    """

    def compute_yield(
        self,
        base_yield: float,
        water_available_mm: float,
        water_requirement_mm: float,
        optimal_water_mm: float,
        wilting_point_mm: float,
        soil_fertility: float,
        temperature_anomaly: float,
        irrigation_method: str,
    ) -> float:
        """
        Compute crop yield for a given season and conditions.

        Args:
            base_yield: Potential yield under ideal conditions (t/ha).
            water_available_mm: Total water available to the crop this season (mm).
            water_requirement_mm: Full-season water requirement (mm).
            optimal_water_mm: Water level at which stress begins below this point (mm).
            wilting_point_mm: Minimum water for survival (below = near-zero yield).
            soil_fertility: Soil fertility index [0, 1].
            temperature_anomaly: Deviation from historical average temperature (°C).
            irrigation_method: Active irrigation method ('flood', 'sprinkler', 'drip').

        Returns:
            Estimated yield in tons per hectare.
        """
        wsf = self.compute_water_stress(water_available_mm, water_requirement_mm, wilting_point_mm)
        sff = self.compute_soil_fertility_factor(soil_fertility)
        tf = self.compute_temperature_factor(temperature_anomaly)
        ief = self.compute_irrigation_efficiency_factor(irrigation_method)

        yield_t_per_ha = base_yield * wsf * sff * tf * ief
        yield_t_per_ha = max(0.0, yield_t_per_ha)

        logger.debug(
            f"Yield: base={base_yield:.2f} × wsf={wsf:.3f} × sff={sff:.3f} × "
            f"tf={tf:.3f} × ief={ief:.3f} = {yield_t_per_ha:.3f} t/ha"
        )
        return yield_t_per_ha

    def compute_water_stress(
        self,
        water_available_mm: float,
        water_requirement_mm: float,
        wilting_point_mm: float,
    ) -> float:
        """
        Compute Penman-Monteith-inspired water stress factor.

        The stress function has three regimes:
            - Below wilting point (< 30% requirement): severe stress → factor ~0.1
            - Deficit stress (30-100% of requirement): linear interpolation
            - Adequate water (≥ requirement): no stress → factor = 1.0

        Args:
            water_available_mm: Available water this season (mm).
            water_requirement_mm: Full crop water requirement (mm).
            wilting_point_mm: Minimum water for crop survival (mm).

        Returns:
            Water stress factor in [0.05, 1.0].
        """
        w = water_available_mm
        req = water_requirement_mm
        wp = wilting_point_mm

        if w >= req:
            return 1.0  # Optimal or excess water
        elif w < wp:
            return 0.05  # Catastrophic: below wilting point
        elif w < 0.4 * req:
            # Severe stress: wilting point to 40% of requirement
            # Linearly interpolate from 0.05 to 0.25
            t = (w - wp) / (0.4 * req - wp + 1e-6)
            return 0.05 + t * 0.20
        elif w < 0.7 * req:
            # Moderate stress: 40-70% of requirement
            # Linearly interpolate from 0.25 to 0.60
            t = (w - 0.4 * req) / (0.3 * req + 1e-6)
            return 0.25 + t * 0.35
        else:
            # Mild stress: 70-100% of requirement
            # Linearly interpolate from 0.60 to 1.0
            t = (w - 0.7 * req) / (0.3 * req + 1e-6)
            return 0.60 + t * 0.40

    def compute_soil_fertility_factor(self, soil_fertility: float) -> float:
        """
        Compute yield multiplier based on soil fertility.

        Uses a sublinear response: degraded soils (fertility < 0.5) have
        disproportionately lower yields due to nutrient limitations.

        Args:
            soil_fertility: Soil fertility index [0, 1].

        Returns:
            Fertility factor in [0.1, 1.0].
        """
        fertility = max(0.0, min(1.0, soil_fertility))
        if fertility >= 0.7:
            # Good soil: near-linear response
            return 0.80 + (fertility - 0.7) / 0.3 * 0.20
        elif fertility >= 0.4:
            # Moderate soil: reduced yield
            return 0.50 + (fertility - 0.4) / 0.3 * 0.30
        else:
            # Poor soil: severe yield limitation
            return 0.10 + fertility / 0.4 * 0.40

    def compute_temperature_factor(self, temperature_anomaly: float) -> float:
        """
        Compute yield multiplier based on temperature deviation from historical average.

        No penalty for anomalies < +2°C. Above +2°C, applies 6% yield loss per degree
        (calibrated from Lobell et al. 2011, Nature Climate Change).

        Args:
            temperature_anomaly: Deviation from historical average in °C.

        Returns:
            Temperature factor in [0.1, 1.0].
        """
        if temperature_anomaly <= TEMP_PENALTY_THRESHOLD_C:
            return 1.0
        excess_degrees = temperature_anomaly - TEMP_PENALTY_THRESHOLD_C
        penalty = excess_degrees * YIELD_LOSS_PER_DEGREE
        return max(0.1, 1.0 - penalty)

    def compute_irrigation_efficiency_factor(self, irrigation_method: str) -> float:
        """
        Compute yield multiplier from irrigation method efficiency.

        Drip and sprinkler irrigation provide more uniform water distribution,
        reducing waterlogging and improving nutrient uptake.

        Args:
            irrigation_method: Method name ('flood', 'sprinkler', 'drip').

        Returns:
            Irrigation efficiency factor.
        """
        factors = {
            "flood": 1.00,
            "sprinkler": 1.08,   # 8% yield improvement (uniform coverage)
            "drip": 1.15,        # 15% yield improvement (root zone delivery)
        }
        return factors.get(irrigation_method, 1.00)

    def compute_water_productivity(
        self,
        yield_t_per_ha: float,
        water_used_mm: float,
        area_ha: float,
    ) -> float:
        """
        Compute water productivity (kg/m³).

        A key metric for irrigation efficiency assessment.

        Args:
            yield_t_per_ha: Crop yield in tons per hectare.
            water_used_mm: Total water used in mm.
            area_ha: Cropped area in hectares.

        Returns:
            Water productivity in kg grain per m³ of water.
        """
        if water_used_mm <= 0 or area_ha <= 0:
            return 0.0
        total_yield_kg = yield_t_per_ha * area_ha * 1000  # t → kg
        total_water_m3 = water_used_mm / 1000 * area_ha * 10000  # mm × ha → m³
        return total_yield_kg / total_water_m3