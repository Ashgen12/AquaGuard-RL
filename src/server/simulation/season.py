# src/server/simulation/season.py
"""
Season progression manager for AquaGuard-RL.

Manages the agricultural calendar (kharif/rabi/zaid cycle),
rainfall sampling, and temperature anomaly generation.

Indian agricultural seasons:
    - Kharif: June–October (monsoon; rice, millets, pulses, oilseeds)
    - Rabi: October–March (winter; wheat, mustard, vegetables)
    - Zaid: March–June (summer; vegetables, melons)

References:
    - IMD Historical Rainfall: India Meteorological Department
    - CGWB Groundwater Year Book (seasonal recharge patterns)
"""

from __future__ import annotations

import logging
import random
import math
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Season sequence (cycles: kharif → rabi → zaid → kharif of next year)
SEASON_ORDER = ["kharif", "rabi", "zaid"]

# Rainfall parameters per season (calibrated from IMD normals)
SEASON_RAINFALL_PARAMS: Dict[str, Dict] = {
    "kharif": {
        "mean_mm": 800,
        "std_mm": 150,
        "description": "Monsoon season (June–October)",
    },
    "rabi": {
        "mean_mm": 120,
        "std_mm": 40,
        "description": "Winter/western disturbance season (October–March)",
    },
    "zaid": {
        "mean_mm": 30,
        "std_mm": 20,
        "description": "Summer/pre-monsoon season (March–June)",
    },
}

# Temperature anomaly parameters (21st-century trend: warming bias)
TEMP_ANOMALY_MEAN = 0.5   # +0.5°C above historical baseline
TEMP_ANOMALY_STD = 0.8


class SeasonManager:
    """
    Manages season progression, rainfall sampling, and climate parameter generation.

    The seasonal cycle:
        kharif (year 1) → rabi (year 1) → zaid (year 1) → kharif (year 2) → ...

    Rainfall is sampled from a Gaussian distribution with seasonal parameters.
    Temperature anomaly is sampled once per season from a correlated random walk.
    """

    def __init__(self) -> None:
        """Initialize with default state."""
        self._season_index: int = 0  # Index into SEASON_ORDER
        self._year: int = 1
        self._temperature_anomaly: float = 0.0
        self._prev_temperature: float = 0.0
        self._rng_seed: Optional[int] = None
        self._forecasted_rainfall: float = 0.0
        self._climate_shock_active: bool = False
        self._climate_shock_factors: Dict[str, float] = {}

    def reset(
        self,
        seed: Optional[int] = None,
        start_season: str = "kharif",
        climate_shock: bool = False,
        climate_shock_factors: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Reset the season manager for a new episode.

        Args:
            seed: Random seed for reproducibility.
            start_season: Season to start the episode on.
            climate_shock: If True, applies rainfall reduction factors.
            climate_shock_factors: Per-season rainfall reduction factors.
        """
        if seed is not None:
            random.seed(seed)

        self._season_index = SEASON_ORDER.index(start_season) if start_season in SEASON_ORDER else 0
        self._year = 1
        self._temperature_anomaly = random.gauss(TEMP_ANOMALY_MEAN, TEMP_ANOMALY_STD * 0.3)
        self._prev_temperature = self._temperature_anomaly
        self._climate_shock_active = climate_shock
        self._climate_shock_factors = climate_shock_factors or {}

        # Generate initial forecast
        self._forecasted_rainfall = self._sample_rainfall(self.current_season)

        logger.debug(
            f"SeasonManager reset: season={self.current_season}, year={self._year}, "
            f"temp_anomaly={self._temperature_anomaly:.2f}°C"
        )

    def advance(self) -> None:
        """
        Advance to the next growing season.

        Cycles through: kharif → rabi → zaid → kharif (incrementing year on kharif).
        Updates temperature anomaly and pre-computes next season's rainfall forecast.
        """
        next_idx = (self._season_index + 1) % len(SEASON_ORDER)

        # Increment year when cycling back to kharif
        if next_idx == 0:
            self._year += 1

        self._season_index = next_idx

        # Update temperature anomaly with autocorrelation (AR(1) process)
        # Temperature has persistence: current ≈ 0.7×previous + 0.3×new draw
        new_temp = random.gauss(TEMP_ANOMALY_MEAN, TEMP_ANOMALY_STD)
        self._temperature_anomaly = 0.7 * self._prev_temperature + 0.3 * new_temp
        self._prev_temperature = self._temperature_anomaly

        # Pre-compute next season's rainfall forecast
        self._forecasted_rainfall = self._sample_rainfall(self.current_season)

        logger.debug(
            f"Season advanced to {self.current_season} year {self._year} | "
            f"rainfall_forecast={self._forecasted_rainfall:.0f}mm | "
            f"temp_anomaly={self._temperature_anomaly:.2f}°C"
        )

    def realize_rainfall(self) -> float:
        """
        Realize actual rainfall for the current season.

        In the model, the forecasted rainfall IS the realized rainfall (perfect forecast
        model). For a noisy model, this could add an additional realization draw.
        In practice, the forecast is generated at season start and the agent must plan
        with this value — there's no additional hidden realization.

        Returns:
            Actual rainfall this season in mm.
        """
        actual = self._forecasted_rainfall

        # Apply climate shock if active
        if self._climate_shock_active:
            shock_factor = self._climate_shock_factors.get(self.current_season, 1.0)
            actual *= shock_factor
            logger.info(f"Climate shock applied: rainfall reduced to {actual:.0f}mm")

        return max(0.0, actual)

    def forecast_rainfall(self) -> float:
        """
        Get the rainfall forecast for the upcoming season.

        Returns:
            Forecasted rainfall in mm (available to the agent at the start of each step).
        """
        return max(0.0, self._forecasted_rainfall)

    def rainfall_distribution_description(self) -> str:
        """
        Generate a natural language description of rainfall uncertainty.

        Returns:
            Human-readable description of the rainfall forecast and uncertainty.
        """
        season = self.current_season
        params = SEASON_RAINFALL_PARAMS[season]
        forecast = self._forecasted_rainfall
        mean = params["mean_mm"]
        std = params["std_mm"]

        # Categorize the forecast relative to normal
        deviation_pct = (forecast - mean) / mean * 100

        if deviation_pct > 25:
            category = "significantly above normal"
        elif deviation_pct > 10:
            category = "above normal"
        elif deviation_pct > -10:
            category = "near normal"
        elif deviation_pct > -25:
            category = "below normal"
        else:
            category = "significantly below normal (drought risk)"

        return (
            f"{season.capitalize()} rainfall forecast: {forecast:.0f}mm ({category}). "
            f"Historical average: {mean:.0f}mm ± {std:.0f}mm. "
            f"Season: {params['description']}."
        )

    def _sample_rainfall(self, season: str) -> float:
        """
        Sample rainfall from the seasonal distribution.

        Args:
            season: Season name.

        Returns:
            Sampled rainfall in mm (non-negative).
        """
        params = SEASON_RAINFALL_PARAMS.get(season, SEASON_RAINFALL_PARAMS["kharif"])
        sample = random.gauss(params["mean_mm"], params["std_mm"])
        return max(0.0, sample)

    @property
    def current_season(self) -> str:
        """Current season name."""
        return SEASON_ORDER[self._season_index]

    @property
    def current_year(self) -> int:
        """Current simulation year (starts at 1)."""
        return self._year

    @property
    def current_temperature_anomaly(self) -> float:
        """Current temperature anomaly in °C (deviation from historical average)."""
        return self._temperature_anomaly

    @property
    def season_index(self) -> int:
        """Index of current season in SEASON_ORDER."""
        return self._season_index

    def crops_for_season(self, season: Optional[str] = None) -> List[str]:
        """
        Get list of crops appropriate for a given season.

        Args:
            season: Season name (defaults to current season).

        Returns:
            List of crop identifiers suitable for this season.
        """
        s = season or self.current_season
        season_crops = {
            "kharif": ["rice", "millet", "pulses", "vegetables"],
            "rabi": ["wheat", "oilseeds", "vegetables"],
            "zaid": ["vegetables"],
        }
        return season_crops.get(s, ["vegetables"])