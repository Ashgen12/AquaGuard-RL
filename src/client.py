# src/client.py
"""
HTTP client for AquaGuard-RL environment.

Provides a synchronous client that connects to a running AquaGuard-RL
server (local or remote) and provides the standard RL interface:
    env.reset(task, seed) → AquaGuardObservation
    env.step(action)      → AquaGuardObservation
    env.state             → AquaGuardState
    env.close()           → stops Docker container if started via from_docker_image()

Usage:
    # Connect to a running server
    env = AquaGuardEnv("http://localhost:8000")
    obs = env.reset(task="baseline", seed=42)
    while not obs.done:
        action = ...
        obs = env.step(action)
    env.close()

    # Start from Docker image
    env = AquaGuardEnv.from_docker_image("aquaguard-env:latest")
    obs = env.reset(task="crisis")
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

from models import AquaGuardAction, AquaGuardObservation, AquaGuardState


def _parse_observation(data: dict) -> AquaGuardObservation:
    """Parse server response into AquaGuardObservation.

    Handles both flat responses and openenv_core's nested format:
        {"observation": {...}, "reward": ..., "done": ...}
    """
    if "observation" in data and isinstance(data["observation"], dict):
        # Nested format from openenv_core Environment
        obs_data = data["observation"]
        obs_data["reward"] = data.get("reward")
        obs_data["done"] = data.get("done", False)
        return AquaGuardObservation(**obs_data)
    return AquaGuardObservation(**data)


class AquaGuardEnv:
    """
    Synchronous HTTP client for AquaGuard-RL environment.

    Implements the standard RL interface (reset/step/state) backed by
    HTTP requests to a running FastAPI environment server.
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        """
        Initialize client connected to a running environment server.

        Args:
            base_url: Base URL of the environment server.
        """
        try:
            import httpx
            self._http = httpx.Client(timeout=120.0)
        except ImportError:
            raise ImportError("httpx required: pip install httpx")

        self._base_url = base_url.rstrip("/")
        self._container: Optional[str] = None

        logger.debug(f"AquaGuardEnv client initialized: {self._base_url}")

    @classmethod
    def from_docker_image(
        cls,
        image: str = "aquaguard-env:latest",
        port: int = 8000,
        timeout: int = 60,
    ) -> "AquaGuardEnv":
        """
        Start AquaGuard-RL in a Docker container and return connected client.

        Args:
            image: Docker image name/tag.
            port: Host port to bind (default 8000).
            timeout: Seconds to wait for server startup.

        Returns:
            Connected AquaGuardEnv client.

        Raises:
            RuntimeError: If server fails to start within timeout.
        """
        import subprocess
        import httpx

        logger.info(f"Starting Docker container: {image}")
        result = subprocess.run(
            ["docker", "run", "-d", "-p", f"{port}:8000", image],
            capture_output=True, text=True, check=True,
        )
        container_id = result.stdout.strip()
        logger.info(f"Container started: {container_id[:12]}")

        base_url = f"http://localhost:{port}"
        for attempt in range(timeout):
            try:
                resp = httpx.get(f"{base_url}/health", timeout=2.0)
                if resp.status_code == 200:
                    env = cls(base_url)
                    env._container = container_id
                    logger.info(f"Server ready after {attempt + 1}s")
                    return env
            except Exception:
                pass
            time.sleep(1.0)

        # Cleanup failed container
        subprocess.run(["docker", "stop", container_id], capture_output=True)
        raise RuntimeError(
            f"Environment server failed to start within {timeout}s. "
            f"Check Docker logs: docker logs {container_id[:12]}"
        )

    def reset(
        self,
        task: str = "baseline",
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> AquaGuardObservation:
        """
        Reset environment to start a new episode.

        Args:
            task: Task name ('baseline', 'crisis', 'policy_shift', 'climate_shock', 'multi_district').
            seed: Optional random seed for reproducibility.
            episode_id: Optional explicit episode ID.

        Returns:
            Initial AquaGuardObservation (step_number=0, reward=None, done=False).
        """
        payload: dict = {"task": task}
        if seed is not None:
            payload["seed"] = seed
        if episode_id is not None:
            payload["episode_id"] = episode_id

        resp = self._http.post(f"{self._base_url}/reset", json=payload)
        resp.raise_for_status()
        return _parse_observation(resp.json())

    def step(self, action: AquaGuardAction) -> AquaGuardObservation:
        """
        Execute one policy step (one growing season ~4 months).

        Args:
            action: AquaGuardAction with crop allocation, water quotas, etc.

        Returns:
            AquaGuardObservation with updated state, reward, and done flag.
        """
        payload = action.model_dump() if hasattr(action, "model_dump") else action.dict()

        # Try openenv_core wrapped format first: {"action": {...}}, then flat fallback
        resp = self._http.post(f"{self._base_url}/step", json={"action": payload})
        if resp.status_code == 422:
            # Server may use standalone (flat) format
            resp = self._http.post(f"{self._base_url}/step", json=payload)
        resp.raise_for_status()
        return _parse_observation(resp.json())

    @property
    def state(self) -> AquaGuardState:
        """Get current episode state metadata."""
        resp = self._http.get(f"{self._base_url}/state")
        resp.raise_for_status()
        return AquaGuardState(**resp.json())

    def close(self) -> None:
        """
        Close the HTTP client and stop Docker container if started via from_docker_image().
        """
        try:
            self._http.close()
        except Exception:
            pass

        if self._container:
            import subprocess
            logger.info(f"Stopping container {self._container[:12]}")
            subprocess.run(["docker", "stop", self._container], capture_output=True)
            subprocess.run(["docker", "rm", self._container], capture_output=True)

    def __enter__(self) -> "AquaGuardEnv":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def health_check(self) -> bool:
        """Check if the server is running and healthy."""
        try:
            resp = self._http.get(f"{self._base_url}/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    def get_info(self) -> dict:
        """Get environment metadata from the server."""
        resp = self._http.get(f"{self._base_url}/info")
        resp.raise_for_status()
        return resp.json()