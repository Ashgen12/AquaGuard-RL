# src/constants.py
"""
Real-world calibrated constants for AquaGuard-RL simulation.

Data sources:
    - FAO AQUASTAT (crop water requirements): https://www.fao.org/aquastat/en/
    - CGWB Annual Report 2023 (groundwater recharge rates): https://cgwb.gov.in/en/reports
    - Government of India MSP FY2024-25 (crop prices): https://cacp.dacnet.nic.in
    - NSSO Situation Assessment Survey 2021 (farmer income): https://mospi.gov.in/web/mospi
    - CACP Input Cost Estimates (cultivation costs): https://cacp.dacnet.nic.in
"""

from __future__ import annotations

# ─── Crop Definitions ─────────────────────────────────────────────────────────

CROP_DATA: dict = {
    "rice": {
        "base_yield_t_per_ha": 4.5,              # FAOSTAT India average yield
        "water_requirement_mm_per_season": 1200,  # FAO AQUASTAT — kharif crop
        "optimal_water_mm": 1000,
        "wilting_point_mm": 400,
        "msp_inr_per_ton": 23100,                # Kharif 2024-25, GoI notification
        "input_cost_inr_per_ha": 42000,          # CACP estimate
        "season": "kharif",
        "water_intensity": "very_high",
        "gwi_irrigation_fraction": 0.60,         # 60% from groundwater (CGWB)
    },
    "wheat": {
        "base_yield_t_per_ha": 3.8,
        "water_requirement_mm_per_season": 450,
        "optimal_water_mm": 400,
        "wilting_point_mm": 150,
        "msp_inr_per_ton": 22150,                # Rabi 2024-25, GoI notification
        "input_cost_inr_per_ha": 35000,
        "season": "rabi",
        "water_intensity": "high",
        "gwi_irrigation_fraction": 0.55,
    },
    "millet": {
        "base_yield_t_per_ha": 2.1,
        "water_requirement_mm_per_season": 350,
        "optimal_water_mm": 300,
        "wilting_point_mm": 100,
        "msp_inr_per_ton": 21150,                # Bajra, Kharif 2024-25
        "input_cost_inr_per_ha": 18000,
        "season": "kharif",
        "water_intensity": "low",
        "gwi_irrigation_fraction": 0.20,
    },
    "pulses": {
        "base_yield_t_per_ha": 1.0,
        "water_requirement_mm_per_season": 300,
        "optimal_water_mm": 270,
        "wilting_point_mm": 90,
        "msp_inr_per_ton": 71500,                # Tur/Arhar, Kharif 2024-25
        "input_cost_inr_per_ha": 22000,
        "season": "kharif",
        "water_intensity": "low",
        "gwi_irrigation_fraction": 0.15,
    },
    "oilseeds": {
        "base_yield_t_per_ha": 1.5,
        "water_requirement_mm_per_season": 250,
        "optimal_water_mm": 220,
        "wilting_point_mm": 80,
        "msp_inr_per_ton": 58500,                # Mustard/Rapeseed, Rabi 2024-25
        "input_cost_inr_per_ha": 28000,
        "season": "rabi",
        "water_intensity": "low",
        "gwi_irrigation_fraction": 0.25,
    },
    "vegetables": {
        "base_yield_t_per_ha": 20.0,
        "water_requirement_mm_per_season": 500,
        "optimal_water_mm": 450,
        "wilting_point_mm": 180,
        "msp_inr_per_ton": 8000,                 # Approximate market-driven price
        "input_cost_inr_per_ha": 65000,
        "season": "all",
        "water_intensity": "medium",
        "gwi_irrigation_fraction": 0.35,
    },
}

VALID_CROPS = set(CROP_DATA.keys())
FOOD_GRAIN_CROPS = {"rice", "wheat", "millet", "pulses"}

# ─── Zone Definitions ─────────────────────────────────────────────────────────

ZONE_DATA: dict = {
    "zone_a": {
        "name": "Northern Plains (Punjab-type)",
        "description": "High-productivity Indo-Gangetic alluvial plain with deep aquifer",
        "arable_land_ha": 80000,
        "farmer_households": 50000,
        "avg_farm_size_ha": 1.6,
        "natural_recharge_mm_per_year": 180,      # CGWB: Indo-Gangetic plain recharge
        "hydraulic_conductivity_m_per_day": 12.0,
        "storage_coefficient": 0.15,
        "initial_gw_depth_m": 22.0,
        "critical_threshold_m": 40.0,
        "collapse_threshold_m": 50.0,
        "baseline_soil_fertility": 0.75,
        "baseline_soil_salinity": 0.12,
    },
    "zone_b": {
        "name": "Central Plains (Haryana-type)",
        "description": "Moderate productivity plains with declining groundwater table",
        "arable_land_ha": 65000,
        "farmer_households": 42000,
        "avg_farm_size_ha": 1.55,
        "natural_recharge_mm_per_year": 120,
        "hydraulic_conductivity_m_per_day": 8.0,
        "storage_coefficient": 0.12,
        "initial_gw_depth_m": 28.0,
        "critical_threshold_m": 40.0,
        "collapse_threshold_m": 50.0,
        "baseline_soil_fertility": 0.65,
        "baseline_soil_salinity": 0.18,
    },
    "zone_c": {
        "name": "Semi-Arid Fringe (Rajasthan-type)",
        "description": "Low rainfall dryland farming zone with stressed aquifer",
        "arable_land_ha": 45000,
        "farmer_households": 30000,
        "avg_farm_size_ha": 2.1,
        "natural_recharge_mm_per_year": 60,
        "hydraulic_conductivity_m_per_day": 4.5,
        "storage_coefficient": 0.08,
        "initial_gw_depth_m": 32.0,
        "critical_threshold_m": 35.0,           # Stricter: arid zone can't recover easily
        "collapse_threshold_m": 45.0,
        "baseline_soil_fertility": 0.55,
        "baseline_soil_salinity": 0.22,
    },
}

VALID_ZONES = set(ZONE_DATA.keys())

# ─── Irrigation Efficiency Factors ────────────────────────────────────────────

IRRIGATION_EFFICIENCY: dict = {
    "flood": {
        "water_use_fraction": 1.0,               # Baseline — highest water use
        "yield_multiplier": 1.0,
        "salinity_accumulation_per_season": 0.025,
        "setup_cost_inr_per_ha": 0,
        "transition_seasons": 0,
    },
    "sprinkler": {
        "water_use_fraction": 0.70,              # 30% water saving vs flood
        "yield_multiplier": 1.08,                # 8% yield improvement
        "salinity_accumulation_per_season": 0.010,
        "setup_cost_inr_per_ha": 35000,
        "transition_seasons": 1,
    },
    "drip": {
        "water_use_fraction": 0.55,              # 45% water saving vs flood
        "yield_multiplier": 1.15,                # 15% yield improvement
        "salinity_accumulation_per_season": 0.005,
        "setup_cost_inr_per_ha": 85000,
        "transition_seasons": 2,
    },
}

VALID_IRRIGATION_METHODS = set(IRRIGATION_EFFICIENCY.keys())

# ─── Economic Constants ────────────────────────────────────────────────────────

POVERTY_LINE_INR_PER_YEAR = 125000           # Rural poverty line, NSSO 2021
INCOME_DISTRIBUTION_SIGMA = 0.45            # Log-normal sigma (NSSO calibrated)
GROUNDWATER_PUMP_COST_INR_PER_M3 = 3.5     # Energy cost for pumping
FOOD_REQUIREMENT_KG_PER_PERSON_YEAR = 175   # GoI food security norm
INDIA_POPULATION_SERVED_FRACTION = 0.15     # District serves 15% of national target
HOUSEHOLD_SIZE = 4.5                        # Average rural household size (Census 2011)
AVG_FARM_SIZE_HA = 1.8                      # Average operational holding size

# ─── Reward Weights (default) ─────────────────────────────────────────────────

DEFAULT_REWARD_WEIGHTS: dict = {
    "groundwater": 0.35,
    "food_security": 0.30,
    "farmer_income": 0.25,
    "crop_diversity": 0.10,
}

# ─── Season Configuration ─────────────────────────────────────────────────────

SEASONS = ["kharif", "rabi", "zaid"]

SEASON_RAINFALL: dict = {
    "kharif": {
        "mean_mm": 800,
        "std_mm": 150,
        "distribution": "normal",
        "description": "June–October monsoon season",
    },
    "rabi": {
        "mean_mm": 120,
        "std_mm": 40,
        "distribution": "normal",
        "description": "October–March winter season",
    },
    "zaid": {
        "mean_mm": 30,
        "std_mm": 20,
        "distribution": "normal",
        "description": "March–June summer season",
    },
}

# Fraction of rainfall that recharges the aquifer (CGWB all-India estimate)
RAINFALL_RECHARGE_FRACTION = 0.12

# Crops viable per season
SEASON_CROPS: dict = {
    "kharif": ["rice", "millet", "pulses", "vegetables"],
    "rabi": ["wheat", "oilseeds", "vegetables"],
    "zaid": ["vegetables"],
}

# ─── Temperature Anomaly ──────────────────────────────────────────────────────

TEMPERATURE_ANOMALY_MEAN = 0.5              # +0.5°C above historical baseline (21st century trend)
TEMPERATURE_ANOMALY_STD = 0.8

# Temperature yield penalty: per °C above optimal (Lobell et al. calibration)
TEMPERATURE_YIELD_PENALTY_PER_DEGREE = 0.06  # 6% yield loss per °C above +2°C

# ─── Inter-Zone Hydrology ─────────────────────────────────────────────────────

INTER_ZONE_CONDUCTIVITY = 0.15             # Fraction of depth differential transferred per season

# ─── Groundwater Physics ──────────────────────────────────────────────────────

SEASONAL_FRACTION = 1.0 / 3.0             # One season = 1/3 of a year (4 months)
MIN_GW_DEPTH_M = 0.1                      # Minimum groundwater depth (spring/artesian)

# ─── Soil Health ──────────────────────────────────────────────────────────────

SOIL_SALINITY_DRIP_REDUCTION = 0.70       # Drip reduces salinity accumulation by 70%
SOIL_FERTILITY_MONOCULTURE_LOSS = 0.003   # Per-season fertility loss from monoculture
SOIL_FERTILITY_DIVERSITY_GAIN_PER_SHANNON = 0.005  # Per 0.1 Shannon unit improvement

# ─── Market Dynamics ─────────────────────────────────────────────────────────

MSP_RANDOM_WALK_SIGMA = 0.05              # Annual MSP change: ±5% standard deviation
MARKET_DEMAND_MEAN_REVERSION = 0.15      # Speed of market demand reversion to mean
MARKET_DEMAND_VOLATILITY = 0.08          # Market demand noise

# ─── Episode Constants ───────────────────────────────────────────────────────

DEFAULT_MAX_STEPS = 10
CATASTROPHIC_GW_DEPTH_M = 50.0           # Episode terminates if avg GW crosses this
FAMINE_CONSECUTIVE_FAILURES = 3          # Episodes ends if food security fails 3x in a row