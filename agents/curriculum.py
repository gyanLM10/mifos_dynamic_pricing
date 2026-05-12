"""
Curriculum Learning — Weight annealing scheduler for the multi-objective
reward function during RL training.

Implements the two-phase training strategy:
  Phase 1 (Exploration): High w1, moderate w3, artificially low w2
  Phase 2 (Calibration): Linearly increase w2 to its true value

This prevents the agent from learning a hyper-conservative "reject all"
policy during early training when exploratory defaults cause large
negative rewards.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


@dataclass
class CurriculumSchedule:
    """
    Defines the weight annealing schedule.

    Attributes
    ----------
    total_timesteps : int
        Total training timesteps.
    phase1_fraction : float
        Fraction of training spent in Phase 1 (exploration). Default 0.4.
    w1_start / w1_end : float
        Interest weight start/end values.
    w2_start / w2_end : float
        Default risk weight start/end values.
    w3_start / w3_end : float
        Client utility weight start/end values.
    """
    total_timesteps: int = 100_000

    phase1_fraction: float = 0.4

    # Phase 1 → Phase 2 weights
    w1_start: float = 1.2
    w1_end: float = 1.0

    w2_start: float = 0.3   # Artificially low to encourage exploration
    w2_end: float = 1.3     # True risk-adjusted value

    w3_start: float = 0.8
    w3_end: float = 0.65

    def get_weights(self, current_timestep: int) -> dict[str, float]:
        """Calculate interpolated weights for the current training step."""
        progress = min(current_timestep / max(self.total_timesteps, 1), 1.0)

        if progress < self.phase1_fraction:
            # Phase 1: Exploration — mostly constant, slight warm-up of w2
            phase_progress = progress / self.phase1_fraction
            w1 = self.w1_start
            w2 = self.w2_start + (self.w2_start * 0.2) * phase_progress
            w3 = self.w3_start
            phase = "exploration"
        else:
            # Phase 2: Calibration — linear interpolation to final values
            phase_progress = (progress - self.phase1_fraction) / (
                1.0 - self.phase1_fraction
            )
            w1 = self.w1_start + (self.w1_end - self.w1_start) * phase_progress
            w2 = self.w2_start + (self.w2_end - self.w2_start) * phase_progress
            w3 = self.w3_start + (self.w3_end - self.w3_start) * phase_progress
            phase = "calibration"

        return {
            "w1": w1,
            "w2": w2,
            "w3": w3,
            "phase": phase,
            "progress": progress,
        }


class CurriculumCallback(BaseCallback):
    """
    Stable Baselines 3 callback that updates the environment's reward
    weights according to the curriculum schedule at each training step.
    """

    def __init__(
        self,
        schedule: CurriculumSchedule,
        log_interval: int = 1000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.schedule = schedule
        self.log_interval = log_interval
        self._last_logged_step = -1

    def _on_step(self) -> bool:
        weights = self.schedule.get_weights(self.num_timesteps)

        # Update the environment's config
        env = self.training_env.envs[0]  # type: ignore
        if hasattr(env, "unwrapped"):
            env = env.unwrapped

        if hasattr(env, "config"):
            env.config["w1"] = weights["w1"]
            env.config["w2"] = weights["w2"]
            env.config["w3"] = weights["w3"]

        # Logging
        if (
            self.num_timesteps - self._last_logged_step >= self.log_interval
            or self._last_logged_step < 0
        ):
            self._last_logged_step = self.num_timesteps
            if self.verbose > 0:
                logger.info(
                    f"Step {self.num_timesteps}: "
                    f"Phase={weights['phase']} "
                    f"w1={weights['w1']:.3f} "
                    f"w2={weights['w2']:.3f} "
                    f"w3={weights['w3']:.3f}"
                )

            # Log to TensorBoard if available
            self.logger.record("curriculum/w1", weights["w1"])
            self.logger.record("curriculum/w2", weights["w2"])
            self.logger.record("curriculum/w3", weights["w3"])
            self.logger.record(
                "curriculum/phase",
                0.0 if weights["phase"] == "exploration" else 1.0,
            )

        return True


class EconomicShockCallback(BaseCallback):
    """
    Periodically injects economic shock events to test policy robustness.
    Increases the shock probability temporarily during training.
    """

    def __init__(
        self,
        shock_interval: int = 10000,
        shock_duration: int = 2000,
        shock_multiplier: float = 3.0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.shock_interval = shock_interval
        self.shock_duration = shock_duration
        self.shock_multiplier = shock_multiplier
        self._original_prob: float | None = None
        self._shock_active = False
        self._shock_start_step = 0

    def _on_step(self) -> bool:
        env = self.training_env.envs[0]  # type: ignore
        if hasattr(env, "unwrapped"):
            env = env.unwrapped

        if not hasattr(env, "config"):
            return True

        # Check if we should start a shock
        if (
            not self._shock_active
            and self.num_timesteps > 0
            and self.num_timesteps % self.shock_interval < 1
        ):
            self._original_prob = env.config["shock_probability"]
            env.config["shock_probability"] = min(
                0.5, self._original_prob * self.shock_multiplier
            )
            self._shock_active = True
            self._shock_start_step = self.num_timesteps
            if self.verbose > 0:
                logger.info(
                    f"💥 Economic shock at step {self.num_timesteps} — "
                    f"shock_prob = {env.config['shock_probability']:.2f}"
                )

        # Check if shock should end
        if (
            self._shock_active
            and self.num_timesteps - self._shock_start_step >= self.shock_duration
        ):
            if self._original_prob is not None:
                env.config["shock_probability"] = self._original_prob
            self._shock_active = False
            if self.verbose > 0:
                logger.info(f"📈 Economic shock ended at step {self.num_timesteps}")

        return True
