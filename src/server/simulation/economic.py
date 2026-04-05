# src/server/simulation/economic.py
"""
Economic model for AquaGuard-RL.

Simulates farmer income, poverty dynamics, MSP price evolution,
and market demand fluctuations.

References:
    - NSSO Situation Assessment Survey 2021: https://mospi.gov.in/web/mospi
    - CACP MSP schedule: https://cacp.dacnet.nic.in
    - Log-normal income distribution: calibrated from NSSO household survey data
"""

from __future__ import annotations

import math
import logging
import random
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Average holding size per farm household (operational holding, Census 2011)
AVG_FARM_SIZE_HA = 1.8

# Groundwater pumping energy cost (INR per m³ equivalent)
PUMP_COST_INR_PER_M3 = 3.5

# Log-normal sigma for income distribution (calibrated from NSSO 2021)
INCOME_DISTRIBUTION_SIGMA = 0.45

# Rural poverty line (NSSO 2021, inflation-adjusted)
POVERTY_LINE_INR = 125000

# MSP random walk parameters
MSP_ANNUAL_DRIFT = 0.04       # ~4% annual increase (historical trend)
MSP_VOLATILITY = 0.05         # ±5% standard deviation

# Market demand parameters
MARKET_DEMAND_MEAN_REVERSION = 0.15   # Speed of reversion to 1.0
MARKET_DEMAND_VOLATILITY = 0.08       # Noise amplitude


class EconomicModel:
    """
    Simulates the agricultural economic system including:
    - Farm household gross income from crop sales
    - Input cost deductions (seeds, fertilizers, labor, irrigation)
    - MSP price evolution (random walk with drift)
    - Market demand fluctuations (mean-reverting process)
    - Poverty fraction estimation via log-normal income distribution
    """

    def __init__(self) -> None:
        """Initialize economic model with baseline MSP prices and market state."""
        self.current_msp: Dict[str, float] = {}
        self._market_demand: Dict[str, float] = {}
        self.last_income_result: Tuple[float, float] = (0.0, 0.0)  # (avg_income, poverty_frac)

    def initialize(self, crop_data: Dict[str, Dict]) -> None:
        """
        Initialize MSP prices and market demand from crop constants.

        Args:
            crop_data: Crop data dictionary from constants.py.
        """
        for crop_id, crop_def in crop_data.items():
            self.current_msp[crop_id] = crop_def["msp_inr_per_ton"]
            self._market_demand[crop_id] = 1.0  # Start at balanced demand

        logger.debug(f"Economic model initialized with {len(self.current_msp)} crops")

    def update_msp_prices(self, year: int) -> None:
        """
        Update MSP prices with random walk + annual drift.

        MSP prices follow a log-normal random walk with:
        - Annual drift of ~4% (historical GoI trend)
        - Seasonal volatility of ±5%

        Args:
            year: Current simulation year (for annual adjustment).
        """
        for crop_id in self.current_msp:
            # Log-normal increment: drift + noise
            log_change = MSP_ANNUAL_DRIFT / 3 + random.gauss(0, MSP_VOLATILITY / 3)
            multiplier = math.exp(log_change)
            self.current_msp[crop_id] = self.current_msp[crop_id] * multiplier

    def update_market_demand(self, current_demand: float) -> float:
        """
        Update market demand with mean-reverting random walk (Ornstein-Uhlenbeck).

        The process reverts toward 1.0 (balanced supply-demand) with noise.

        Args:
            current_demand: Current market demand index for a crop.

        Returns:
            Updated market demand index, clamped to [0.3, 2.0].
        """
        # Mean-reverting: ΔD = κ(μ - D) + σε
        kappa = MARKET_DEMAND_MEAN_REVERSION
        sigma = MARKET_DEMAND_VOLATILITY
        mean = 1.0

        delta = kappa * (mean - current_demand) + sigma * random.gauss(0, 1)
        new_demand = current_demand + delta
        return max(0.3, min(2.0, new_demand))

    def compute_farmer_income(
        self,
        crop_states: Dict[str, Dict],
        zone_states: Dict[str, Dict],
        zone_data: Dict[str, Dict],
        crop_data: Dict[str, Dict],
    ) -> Tuple[float, float]:
        """
        Compute average farm household income and poverty fraction.

        Income Model (per average 1.8-ha farm household):
            gross_income = Σ crops (allocated_fraction × farm_size × yield × msp × subsidy)
            water_cost = extraction_volume × pump_cost
            net_income = gross_income - input_costs - water_cost

        Poverty fraction estimated via log-normal CDF:
            P(income < poverty_line) = Φ((ln(poverty_line) - ln(net_income)) / σ)

        Args:
            crop_states: Current crop state dictionaries.
            zone_states: Current zone state dictionaries (for water extraction).
            zone_data: Static zone parameter dictionaries.
            crop_data: Static crop parameter dictionaries.

        Returns:
            Tuple of (average_income_inr, poverty_fraction).
        """
        if not self.current_msp:
            # Model not initialized — return reasonable defaults
            return (150000.0, 0.30)

        total_land_ha = sum(z["arable_land_ha"] for z in zone_states.values())
        total_farmers = sum(zd["farmer_households"] for zd in zone_data.values())

        if total_farmers <= 0:
            return (0.0, 1.0)

        avg_farm_size = total_land_ha / total_farmers

        # Gross income from crop sales
        gross_income = 0.0
        total_input_costs = 0.0

        for crop_id, crop_state in crop_states.items():
            alloc = crop_state.get("allocated_fraction", 0.0)
            if alloc < 0.001:
                continue

            crop_def = crop_data.get(crop_id, {})
            yield_t_per_ha = crop_state.get("yield_t_per_ha", crop_def.get("base_yield_t_per_ha", 0))
            msp = self.current_msp.get(crop_id, crop_def.get("msp_inr_per_ton", 0))
            subsidy_mult = crop_state.get("subsidy_multiplier", 1.0)
            market_demand = crop_state.get("market_demand_index", 1.0)
            input_cost_per_ha = crop_def.get("input_cost_inr_per_ha", 30000)

            # Effective price: MSP adjusted by subsidy and market demand
            effective_price = msp * subsidy_mult * min(1.5, max(0.8, market_demand))

            crop_land = alloc * avg_farm_size
            gross_income += crop_land * yield_t_per_ha * effective_price
            total_input_costs += crop_land * input_cost_per_ha

        # Water pumping cost (proportional to average actual extraction applied)
        avg_water_extracted_m = sum(z.get("actual_extracted_m", 0) for z in zone_states.values()) / max(len(zone_states), 1)
        # avg_water_extracted_m is volume per unit area. total volume = depth (m) * farm size (m²)
        water_volume_m3 = avg_water_extracted_m * avg_farm_size * 10000
        water_cost = water_volume_m3 * PUMP_COST_INR_PER_M3

        # Net income (per season)
        net_income = max(10000.0, gross_income - total_input_costs - water_cost)

        # Poverty fraction via log-normal CDF
        # Compare annualized net income against the annual poverty line
        annualized_income = net_income * 3.0
        poverty_fraction = self._lognormal_poverty_fraction(annualized_income)

        self.last_income_result = (annualized_income, poverty_fraction)

        logger.debug(
            f"Farmer income: gross={gross_income:.0f} - costs={total_input_costs:.0f} "
            f"- water={water_cost:.0f} = net={net_income:.0f} INR | "
            f"poverty_frac={poverty_fraction:.3f}"
        )
        return net_income, poverty_fraction

    def _lognormal_poverty_fraction(self, mean_income: float) -> float:
        """
        Estimate fraction of farmers below poverty line using log-normal distribution.

        The income distribution is modeled as log-normal with:
        - Location parameter: ln(mean_income) - σ²/2 (to maintain mean)
        - Scale parameter: σ = 0.45 (NSSO 2021 calibration)

        Args:
            mean_income: Average farm household income in INR.

        Returns:
            Fraction of farmers with income below poverty line [0, 1].
        """
        if mean_income <= 0:
            return 1.0

        sigma = INCOME_DISTRIBUTION_SIGMA
        poverty_line = POVERTY_LINE_INR

        # Log-normal: if X ~ LN(μ, σ²), then E[X] = exp(μ + σ²/2)
        # So μ = ln(mean_income) - σ²/2
        mu = math.log(max(mean_income, 1.0)) - (sigma ** 2) / 2

        # P(X < poverty_line) = Φ((ln(poverty_line) - μ) / σ)
        z = (math.log(poverty_line) - mu) / sigma

        # Standard normal CDF approximation (Abramowitz & Stegun)
        return self._standard_normal_cdf(z)

    @staticmethod
    def _standard_normal_cdf(z: float) -> float:
        """
        Compute standard normal CDF using math.erfc.

        Args:
            z: Z-score.

        Returns:
            CDF value in [0, 1].
        """
        return 0.5 * math.erfc(-z / math.sqrt(2))