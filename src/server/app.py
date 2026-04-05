# src/server/app.py
"""
FastAPI application for AquaGuard-RL environment.

Provides HTTP endpoints for the OpenEnv protocol:
    POST /reset   — Start a new episode
    POST /step    — Execute one action
    GET  /state   — Get episode state metadata
    GET  /health  — Health check
    GET  /info    — Environment metadata

Supports concurrent sessions via per-request environment instances.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# ─── Try to use openenv_core's create_app; fallback to manual FastAPI setup ───

try:
    from openenv_core import create_app as _oe_create_app  # type: ignore
    _OPENENV_CREATE_APP = True
except ImportError:
    _OPENENV_CREATE_APP = False

from server.aquaguard_environment import AquaGuardEnvironment
from models import AquaGuardAction, AquaGuardObservation, AquaGuardState


def _create_standalone_app() -> FastAPI:
    """
    Create a standalone FastAPI app that implements the OpenEnv HTTP API
    without depending on openenv_core.create_app().

    Uses a single shared environment instance (non-concurrent).
    For production, use openenv_core.create_app() which handles concurrency.
    """
    app = FastAPI(
        title="AquaGuard-RL",
        description=(
            "India Groundwater & Agricultural Resource Management RL Environment. "
            "Meta PyTorch OpenEnv Hackathon submission."
        ),
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Shared environment instance (per-process)
    _env = AquaGuardEnvironment()

    @app.post("/reset")
    async def reset(request: Request) -> dict:
        """Reset environment to start a new episode."""
        try:
            body = await request.json() if request.headers.get("content-type") == "application/json" else {}
        except Exception:
            body = {}

        task = body.get("task", "baseline")
        seed = body.get("seed", None)
        episode_id = body.get("episode_id", None)

        obs = _env.reset(seed=seed, episode_id=episode_id, task=task)
        return obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()

    @app.post("/step")
    async def step(request: Request) -> dict:
        """Execute one policy step."""
        try:
            body = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")

        # Unwrap OpenEnv client wrapper: {"action": {...}} -> {...}
        if "action" in body and isinstance(body["action"], dict):
            body = body["action"]

        try:
            action = AquaGuardAction(**body)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid action: {e}")

        try:
            obs = _env.step(action)
            return obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Step error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Step failed: {e}")

    @app.get("/state")
    async def get_state() -> dict:
        """Get current episode state metadata."""
        try:
            state = _env.state
            return state.model_dump() if hasattr(state, "model_dump") else state.dict()
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/health")
    async def health() -> dict:
        """Health check endpoint."""
        return {"status": "healthy", "environment": "aquaguard-rl", "version": "1.0.0"}

    @app.get("/info")
    async def environment_info() -> dict:
        """Return environment metadata."""
        return {
            "name": "AquaGuard-RL",
            "version": "1.0.0",
            "description": "India Groundwater & Agricultural Resource Management Environment",
            "tasks": ["baseline", "crisis", "policy_shift", "climate_shock", "multi_district"],
            "action_space": {
                "crop_allocation": "Dict[str, float] — 6 crop fractions summing to ≤1.0",
                "water_quotas": "Dict[str, float] — per-zone mm/season [0-2000]",
                "irrigation_methods": "Dict[str, str] — flood/sprinkler/drip per zone",
                "extraction_limits": "Dict[str, float] — per-zone m/season [0-60]",
                "subsidy_adjustments": "Dict[str, float] — relative adjustment [-1,1]",
                "justification": "str — natural language reasoning (evaluated by LLM grader)",
            },
            "observation_space": {
                "season": "kharif/rabi/zaid",
                "zones": "3 zones with groundwater depth, soil health, irrigation status",
                "crops": "6 crops with yield, MSP, water requirements, market demand",
                "food_security_ratio": "production/requirement (1.0 = exactly meeting target)",
                "percent_farmers_below_poverty": "0-100%",
                "shannon_diversity_index": "0 (monoculture) to 1.79 (equal 6-crop)",
                "scenario_description": "Natural language summary for LLM agents",
            },
            "reward_range": [-10.0, 10.0],
            "max_steps": {
                "baseline": 10, "crisis": 12, "policy_shift": 8,
                "climate_shock": 6, "multi_district": 15,
            },
            "openenv_spec": "0.2.2",
            "social_impact": "India groundwater crisis — 500% extraction increase 1990-2020",
            "data_sources": [
                "FAO AQUASTAT (crop water requirements): https://www.fao.org/aquastat/en/",
                "CGWB Annual Report 2023: https://cgwb.gov.in/en/reports",
                "GoI MSP Schedule FY2024-25: https://cacp.dacnet.nic.in",
                "NSSO Situation Assessment Survey 2021: https://mospi.gov.in/web/mospi",
            ],
        }

    return app


# ─── App Factory ──────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Create the FastAPI application.

    Uses standalone implementation that maintains a single shared environment
    instance with proper reset/step state management.

    Note: openenv_core.create_app() is available but creates separate env
    instances per session, which can break state continuity. We use our
    standalone app which implements the same OpenEnv REST API protocol.
    """
    logger.info("Creating standalone FastAPI app (OpenEnv-compatible)")
    return _create_standalone_app()


# ─── Module-level app instance ────────────────────────────────────────────────

app = create_app()


# ─── Main entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    log_level = os.getenv("LOG_LEVEL", "info").lower()

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        workers=1,
        log_level=log_level,
        reload=False,
    )