# src/server/tasks/__init__.py
"""Task definitions for AquaGuard-RL (5 tasks of varying difficulty)."""

from .task_definitions import TASK_CONFIGS, AVAILABLE_TASKS, TASK_DIFFICULTIES, get_task_config

__all__ = ["TASK_CONFIGS", "AVAILABLE_TASKS", "TASK_DIFFICULTIES", "get_task_config"]