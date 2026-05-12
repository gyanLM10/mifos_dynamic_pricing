"""
Training Orchestrator — Entry point for training RL agents on the
MicroLoanEnv with curriculum learning, fairness monitoring, and
multi-modal policy networks.

Usage:
    python -m agents.train --algo ppo --total-timesteps 100000
    python -m agents.train --algo dqn --no-nlp --total-timesteps 50000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Register the environment
import env  # noqa: F401

from agents.curriculum import (
    CurriculumCallback,
    CurriculumSchedule,
    EconomicShockCallback,
)
from agents.multimodal_policy import get_policy_kwargs
from safety.fairness_metrics import FairnessMonitorCallback

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def make_env(
    use_nlp: bool = True,
    sentiment_dim: int = 768,
    use_continuous: bool = False,
) -> gym.Env:
    """Create and configure the MicroLoanEnv."""
    config = {
        "use_nlp_features": use_nlp,
        "sentiment_dim": sentiment_dim if use_nlp else 0,
        "use_continuous_actions": use_continuous,
    }
    return gym.make("MicroLoan-v0", config=config)


def train_ppo(
    env: gym.Env,
    total_timesteps: int,
    use_nlp: bool,
    output_dir: Path,
) -> None:
    """Train a PPO agent with curriculum learning."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, EvalCallback

    policy_kwargs = get_policy_kwargs(use_nlp=use_nlp)

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=str(output_dir / "tb_logs"),
    )

    # Callbacks
    schedule = CurriculumSchedule(total_timesteps=total_timesteps)
    curriculum_cb = CurriculumCallback(schedule, log_interval=2000, verbose=1)
    shock_cb = EconomicShockCallback(
        shock_interval=total_timesteps // 5,
        shock_duration=total_timesteps // 25,
        verbose=1,
    )
    fairness_cb = FairnessMonitorCallback(log_interval=5000, verbose=1)

    eval_env = make_env(use_nlp=use_nlp)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )

    callbacks = CallbackList([curriculum_cb, shock_cb, fairness_cb, eval_cb])

    logger.info(f"Starting PPO training for {total_timesteps} timesteps")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(str(output_dir / "final_model"))
    logger.info(f"Model saved to {output_dir / 'final_model'}")


def train_dqn(
    env: gym.Env,
    total_timesteps: int,
    use_nlp: bool,
    output_dir: Path,
) -> None:
    """Train a DQN agent with curriculum learning."""
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import CallbackList, EvalCallback

    policy_kwargs = get_policy_kwargs(use_nlp=use_nlp)
    # DQN uses a flat net_arch list, not a dict with pi/vf keys
    policy_kwargs["net_arch"] = [128, 64]

    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_final_eps=0.02,
        verbose=1,
        tensorboard_log=str(output_dir / "tb_logs"),
    )

    schedule = CurriculumSchedule(total_timesteps=total_timesteps)
    curriculum_cb = CurriculumCallback(schedule, log_interval=2000, verbose=1)
    fairness_cb = FairnessMonitorCallback(log_interval=5000, verbose=1)

    eval_env = make_env(use_nlp=use_nlp)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )

    callbacks = CallbackList([curriculum_cb, fairness_cb, eval_cb])

    logger.info(f"Starting DQN training for {total_timesteps} timesteps")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(str(output_dir / "final_model"))
    logger.info(f"Model saved to {output_dir / 'final_model'}")


def main():
    parser = argparse.ArgumentParser(
        description="Train RL agents on the MicroLoan Dynamic Pricing environment"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "dqn"],
        help="RL algorithm to use (default: ppo)",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps (default: 100000)",
    )
    parser.add_argument(
        "--no-nlp",
        action="store_true",
        help="Disable NLP features (tabular-only baseline)",
    )
    parser.add_argument(
        "--sentiment-dim",
        type=int,
        default=768,
        help="Dimension of sentiment embeddings (default: 768)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Output directory for models and logs",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Use continuous action space",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_nlp = not args.no_nlp

    env_instance = make_env(
        use_nlp=use_nlp,
        sentiment_dim=args.sentiment_dim,
        use_continuous=args.continuous,
    )

    logger.info(f"Environment: MicroLoan-v0")
    logger.info(f"  Observation space: {env_instance.observation_space}")
    logger.info(f"  Action space: {env_instance.action_space}")
    logger.info(f"  NLP features: {use_nlp}")
    logger.info(f"  Algorithm: {args.algo.upper()}")

    if args.algo == "ppo":
        train_ppo(env_instance, args.total_timesteps, use_nlp, output_dir)
    elif args.algo == "dqn":
        train_dqn(env_instance, args.total_timesteps, use_nlp, output_dir)


if __name__ == "__main__":
    main()
